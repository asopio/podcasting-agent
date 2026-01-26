import gradio as gr

# Placeholder function for podcast generation
# This will be replaced with actual LLM and TTS agent implementation
def generate_podcast(urls_text, progress=gr.Progress(track_tqdm=True)):
    """
    Placeholder function for podcast generation.
    
    In the future, this will:
    1. Parse the URLs from the input
    2. Fetch and summarize articles/papers using LLM
    3. Generate dialogue from summaries
    4. Convert dialogue to audio using TTS
    5. Return the audio file
    
    Args:
        urls_text: Multi-line text containing URLs
        progress: Gradio progress tracker
    
    Returns:
        Tuple of (status_message, audio_file_path)
    """
    if not urls_text or not urls_text.strip():
        return "⚠️ Please enter at least one URL", None
    
    # Parse URLs (split by newlines and filter empty lines)
    urls = [url.strip() for url in urls_text.strip().split('\n') if url.strip()]
    
    if not urls:
        return "⚠️ Please enter at least one valid URL", None
    
    # Placeholder response
    status_msg = f"✅ UI Ready! Found {len(urls)} URL(s):\n" + "\n".join(f"  • {url}" for url in urls)
    status_msg += "\n\n⏳ Backend implementation pending (LLM summarization & TTS generation)"
    
    return status_msg, None


examples = [
    "https://arxiv.org/abs/2301.00001\nhttps://example.com/article1",
    "https://arxiv.org/abs/2312.12345",
    "https://huggingface.co/blog/example-post\nhttps://medium.com/@example/article",
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
            ### AI-powered podcast generation from articles and papers
            
            Enter URLs of articles or papers you want to hear discussed. The AI agent will:
            1. 📄 Summarize the content using LLM
            2. 💬 Generate engaging dialogue
            3. 🎙️ Convert to audio using text-to-speech
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
            **Note:** This is the UI interface. Backend implementation for LLM summarization and TTS is pending.
            """
        )

    # Event handlers
    generate_button.click(
        fn=generate_podcast,
        inputs=[urls_input],
        outputs=[status_output, audio_output],
    )
    
    clear_button.click(
        fn=lambda: ("", "", None),
        inputs=[],
        outputs=[urls_input, status_output, audio_output],
    )

if __name__ == "__main__":
    demo.launch()
