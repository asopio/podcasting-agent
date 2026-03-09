import logging

import gradio as gr

from podcast_generator import create_podcast

logging.basicConfig(level=logging.INFO)


def generate_podcast(urls_text, progress=gr.Progress(track_tqdm=True)):
    """Generate a podcast episode from one or more arXiv URLs.

    Args:
        urls_text: Multi-line text containing URLs (only the first arXiv URL
            is used for the current episode).
        progress: Gradio progress tracker.

    Returns:
        Tuple of (status_message, audio_file_path).
    """
    if not urls_text or not urls_text.strip():
        return "⚠️ Please enter at least one URL", None

    urls = [u.strip() for u in urls_text.strip().splitlines() if u.strip()]
    if not urls:
        return "⚠️ Please enter at least one valid URL", None

    # Only the first URL is processed per episode
    url = urls[0]

    try:
        status, audio_path = create_podcast(url, progress_callback=progress)
        return status, audio_path
    except Exception as exc:
        logging.exception("Podcast generation failed")
        return f"❌ Error: {exc}", None


examples = [
    "https://arxiv.org/abs/1706.03762",
    "https://arxiv.org/abs/2005.14165",
    "https://arxiv.org/abs/2303.08774",
]

css = """
#col-container {
    margin: 0 auto;
    max-width: 800px;
}
"""

with gr.Blocks(css=css, theme=gr.themes.Soft()) as demo:
    with gr.Column(elem_id="col-container"):
        gr.Markdown(
            """
            # 🎧 Podcasting Agent
            ### AI-powered podcast generation from arXiv papers

            Enter the URL of an arXiv paper. The agent will:
            1. 📄 Fetch the paper metadata from arXiv
            2. 💬 Generate an engaging two-host dialogue using **Qwen3.5-9B**
            3. 🎙️ Synthesise the audio using **Qwen3-TTS** (two distinct voices)
            """
        )

        with gr.Row():
            urls_input = gr.Textbox(
                label="Article & Paper URLs",
                placeholder="Enter URLs, one per line:\nhttps://arxiv.org/abs/...\nhttps://example.com/article\n...",
                lines=5,
                max_lines=10,
            )

        with gr.Row():
            clear_button = gr.Button("Clear", scale=1)
            generate_button = gr.Button("🎙️ Generate Podcast", scale=2, variant="primary")

        status_output = gr.Textbox(
            label="Status",
            lines=3,
            max_lines=10,
            interactive=False,
        )
        
        audio_output = gr.Audio(
            label="Generated Podcast",
            type="filepath",
        )

        with gr.Accordion("📚 Examples", open=True):
            gr.Examples(
                examples=examples,
                inputs=[urls_input],
                label="Click to use example URLs"
            )
        
        gr.Markdown(
            """
            ---
            **Hosts:** Alex (Ryan voice) · Jordan (Aiden voice) — powered by Qwen3-TTS
            """
        )

    # Event handlers
    generate_button.click(
        fn=generate_podcast,
        inputs=[urls_input],
        outputs=[status_output, audio_output],
    )
    
    def clear_form():
        """Clear all input and output fields."""
        return "", "", None  # urls_input, status_output, audio_output
    
    clear_button.click(
        fn=clear_form,
        inputs=[],
        outputs=[urls_input, status_output, audio_output],
    )

if __name__ == "__main__":
    demo.launch()
