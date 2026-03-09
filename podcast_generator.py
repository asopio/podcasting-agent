"""
podcast_generator.py
~~~~~~~~~~~~~~~~~~~~
End-to-end pipeline for turning an arXiv paper into a two-host podcast episode.

Pipeline:
  1. Fetch paper metadata from the arXiv API (title, authors, abstract).
  2. Use Qwen/Qwen3.5-9B (local, via HuggingFace Transformers) to write a
     two-host podcast script.
  3. Use Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice (local, via qwen-tts) to
     synthesise each dialogue line with a distinct speaker voice.
  4. Concatenate the audio segments (with short silences between turns) and
     write a WAV file.
"""

import logging
import os
import re
import tempfile

import arxiv
import numpy as np
import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model identifiers
# ---------------------------------------------------------------------------

SCRIPT_MODEL: str = "Qwen/Qwen3.5-9B"
TTS_MODEL: str = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"

# Two English native speakers from the Qwen3-TTS CustomVoice catalogue
HOST1_SPEAKER: str = "Ryan"   # Dynamic male, strong rhythmic drive
HOST2_SPEAKER: str = "Aiden"  # Sunny American male, clear midrange

# Host names used in the script prompt and parser
HOST1_NAME: str = "Alex"
HOST2_NAME: str = "Jordan"

# Preferred dtype for both LLM and TTS on GPU
_GPU_DTYPE = torch.bfloat16
_CPU_DTYPE = torch.float32

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are a podcast script writer. "
    "Create an engaging two-host podcast episode about the research paper below. "
    f"The hosts are {HOST1_NAME} (Host1) and {HOST2_NAME} (Host2). "
    "Their conversation should be natural, informative, and accessible to a general "
    "technical audience. "
    "Format every line strictly as either:\n"
    f"  {HOST1_NAME}: <dialogue>\n"
    f"  {HOST2_NAME}: <dialogue>\n"
    "Aim for 12 to 16 exchanges in total. "
    "Do NOT include any stage directions, markdown, or extra commentary."
)

# ---------------------------------------------------------------------------
# arXiv helpers
# ---------------------------------------------------------------------------


def _extract_arxiv_id(url: str) -> str:
    """Return the bare arXiv paper ID (e.g. '1706.03762') from a URL or ID string."""
    # Matches URLs like https://arxiv.org/abs/1706.03762 or .../pdf/1706.03762
    m = re.search(r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d+(?:v\d+)?)", url)
    if m:
        return m.group(1)
    # Bare ID passed directly (e.g. "1706.03762" or "2301.00001v2")
    m = re.match(r"^(\d{4}\.\d+(?:v\d+)?)$", url.strip())
    if m:
        return m.group(1)
    raise ValueError(f"Cannot extract an arXiv ID from: {url!r}")


def fetch_arxiv_paper(url: str) -> dict:
    """Fetch paper metadata from the arXiv API.

    Returns a dict with keys: title, authors, abstract, published, arxiv_id.
    """
    paper_id = _extract_arxiv_id(url)
    logger.info("Fetching arXiv paper %s", paper_id)
    client = arxiv.Client()
    search = arxiv.Search(id_list=[paper_id])
    paper = next(client.results(search))
    return {
        "title": paper.title,
        "authors": [str(a) for a in paper.authors],
        "abstract": paper.summary.replace("\n", " "),
        "published": paper.published.strftime("%B %d, %Y"),
        "arxiv_id": paper_id,
    }


# ---------------------------------------------------------------------------
# Script generation
# ---------------------------------------------------------------------------


def generate_script(paper: dict, model_name: str = SCRIPT_MODEL) -> list[dict]:
    """Generate a two-host podcast script from *paper* using Qwen3.5-9B.

    Returns a list of dicts: [{"speaker": "Host1"|"Host2", "text": str}, ...].
    """
    logger.info("Loading script model: %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dtype = _GPU_DTYPE if torch.cuda.is_available() else _CPU_DTYPE
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto",
    )
    model.eval()

    user_content = (
        f"Paper Title: {paper['title']}\n"
        f"Authors: {', '.join(paper['authors'][:5])}\n"
        f"Published: {paper['published']}\n\n"
        f"Abstract:\n{paper['abstract']}\n\n"
        "Write the podcast script now."
    )
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer([prompt], return_tensors="pt").to(model.device)

    logger.info("Generating podcast script…")
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=2048,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
        )

    new_ids = output_ids[0][inputs.input_ids.shape[1]:]
    script_text = tokenizer.decode(new_ids, skip_special_tokens=True)
    logger.info("Script generated (%d chars)", len(script_text))

    # Free GPU/CPU memory before loading the TTS model
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    segments = _parse_script(script_text)
    if not segments:
        raise ValueError(
            "Script generation produced no parseable dialogue lines. "
            "Raw output:\n" + script_text[:500]
        )
    return segments


def _parse_script(text: str) -> list[dict]:
    """Parse raw LLM output into structured dialogue segments."""
    segments = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        m = re.match(
            rf"^({re.escape(HOST1_NAME)}|{re.escape(HOST2_NAME)})\s*:\s*(.*)",
            line,
            re.IGNORECASE,
        )
        if m:
            host = "Host1" if m.group(1).lower() == HOST1_NAME.lower() else "Host2"
            dialogue = m.group(2).strip()
            if dialogue:
                segments.append({"speaker": host, "text": dialogue})
    return segments


# ---------------------------------------------------------------------------
# Audio synthesis
# ---------------------------------------------------------------------------


def generate_audio(
    script: list[dict],
    output_path: str,
    tts_model_name: str = TTS_MODEL,
    silence_between_turns_sec: float = 0.4,
) -> str:
    """Synthesise *script* to a WAV file at *output_path*.

    Each Host1 line is spoken by Ryan; each Host2 line by Aiden.
    Returns *output_path*.
    """
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = _GPU_DTYPE if torch.cuda.is_available() else _CPU_DTYPE

    logger.info("Loading TTS model: %s (device=%s)", tts_model_name, device)
    tts = Qwen3TTSModel.from_pretrained(
        tts_model_name,
        device_map=device,
        dtype=dtype,
    )

    all_chunks: list[np.ndarray] = []
    sample_rate: int | None = None

    for i, segment in enumerate(script):
        speaker = HOST1_SPEAKER if segment["speaker"] == "Host1" else HOST2_SPEAKER
        logger.info(
            "Synthesising line %d/%d  [%s / %s]: %s",
            i + 1,
            len(script),
            segment["speaker"],
            speaker,
            segment["text"][:60],
        )
        wavs, sr = tts.generate_custom_voice(
            text=segment["text"],
            language="English",
            speaker=speaker,
        )
        if sample_rate is None:
            sample_rate = sr
        all_chunks.append(wavs[0].astype(np.float32))
        # Short silence between turns
        all_chunks.append(
            np.zeros(int(sr * silence_between_turns_sec), dtype=np.float32)
        )

    combined = np.concatenate(all_chunks)
    sf.write(output_path, combined, sample_rate)
    logger.info("Audio written to %s (%.1f s)", output_path, len(combined) / sample_rate)
    return output_path


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------


def create_podcast(
    url: str,
    output_path: str | None = None,
    progress_callback=None,
) -> tuple[str, str]:
    """Full pipeline: fetch paper → generate script → synthesise audio.

    Args:
        url: arXiv paper URL (e.g. https://arxiv.org/abs/1706.03762).
        output_path: Where to write the WAV file.  A temp file is created when
            ``None`` is passed.
        progress_callback: Optional callable(step, total, description) for UI
            progress reporting.

    Returns:
        (status_message, audio_output_path)
    """
    if output_path is None:
        fd, output_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)

    def _progress(step, total, desc):
        if progress_callback is not None:
            progress_callback(step / total, desc=desc)

    _progress(0, 3, "Fetching paper from arXiv…")
    paper = fetch_arxiv_paper(url)
    logger.info("Paper: %s", paper["title"])

    _progress(1, 3, "Generating podcast script with Qwen3.5-9B…")
    script = generate_script(paper)

    _progress(2, 3, "Synthesising audio with Qwen3-TTS…")
    generate_audio(script, output_path)

    _progress(3, 3, "Done!")

    authors_str = ", ".join(paper["authors"][:3])
    if len(paper["authors"]) > 3:
        authors_str += f" and {len(paper['authors']) - 3} others"

    status = (
        f"✅ Podcast generated!\n"
        f"📄 {paper['title']}\n"
        f"👥 {authors_str}\n"
        f"📅 {paper['published']}\n"
        f"🎙️ {len(script)} dialogue lines"
    )
    return status, output_path
