#!/usr/bin/env python3
"""
Gemma 4 Unified audio — push-to-talk Gradio front-end.

Record from mic, get streamed text back. Thinking tokens stripped automatically.

Usage:
    python gemma4_audio_gradio.py
    python gemma4_audio_gradio.py --model mlx-community/gemma-4-12B-it-8bit
    python gemma4_audio_gradio.py --port 7861 --share

Press Ctrl+C during recording to stop early and generate immediately.
"""

import argparse

from mlx_vlm.tools.gemma4_audio.constants import (
    AUDIO_SAMPLES_PER_TOKEN,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL,
    DEFAULT_PROMPT,
    DEFAULT_TEMP,
)
from mlx_vlm.tools.gemma4_audio.core import load_model
from mlx_vlm.tools.gemma4_audio.audio import process_gradio_audio
from mlx_vlm.tools.gemma4_audio.prompt import build_prompt
from mlx_vlm.tools.gemma4_audio.inference import run_inference

_model = None
_processor = None


def _ensure_loaded(model_path: str):
    global _model, _processor
    if _model is None:
        _model, _processor = load_model(model_path)


def build_demo(model_path: str, max_tokens: int, temperature: float):
    import gradio as gr

    _ensure_loaded(model_path)

    def infer(audio_tuple, user_prompt, max_tok, temp):
        if audio_tuple is None:
            yield "No audio recorded. Click the mic button and speak."
            return
        audio = process_gradio_audio(audio_tuple)
        n_audio_tokens = len(audio) // AUDIO_SAMPLES_PER_TOKEN
        print(f"[audio] {len(audio)} samples → ~{n_audio_tokens} audio tokens")
        if n_audio_tokens == 0:
            yield "Audio too short — nothing to process."
            return
        prompt_fn = lambda text: build_prompt(_processor, _model.config, text)
        yield from run_inference(_model, _processor, audio, user_prompt,
                                 int(max_tok), float(temp), prompt_fn)

    with gr.Blocks(title="Gemma 4 Audio") as demo:
        gr.Markdown(
            "## Gemma 4 Audio\n"
            "Record from mic, get streamed text back. "
            "Powered by Gemma 4 Unified encoder-free audio on M4 Max."
        )
        with gr.Row():
            with gr.Column(scale=1):
                audio_in = gr.Audio(
                    sources=["microphone"],
                    type="numpy",
                    label="Microphone (click to record)",
                )
                prompt_box = gr.Textbox(
                    value=DEFAULT_PROMPT,
                    label="Prompt",
                    lines=2,
                )
                with gr.Accordion("Settings", open=False):
                    max_tok_slider = gr.Slider(
                        minimum=64, maximum=1024, value=max_tokens, step=64,
                        label="Max tokens",
                    )
                    temp_slider = gr.Slider(
                        minimum=0.0, maximum=1.0, value=temperature, step=0.05,
                        label="Temperature",
                    )
                submit_btn = gr.Button("Generate", variant="primary")
            with gr.Column(scale=1):
                output_box = gr.Textbox(label="Response", lines=16)

        submit_btn.click(
            fn=infer,
            inputs=[audio_in, prompt_box, max_tok_slider, temp_slider],
            outputs=output_box,
        )
        audio_in.stop_recording(
            fn=infer,
            inputs=[audio_in, prompt_box, max_tok_slider, temp_slider],
            outputs=output_box,
        )

    return demo


def main():
    parser = argparse.ArgumentParser(description="Gemma 4 audio Gradio front-end")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--share", action="store_true",
                        help="Create a public Gradio share link")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--temp", type=float, default=DEFAULT_TEMP)
    args = parser.parse_args()

    import gradio as gr
    demo = build_demo(args.model, args.max_tokens, args.temp)
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        theme=gr.themes.Soft(),
    )


if __name__ == "__main__":
    main()
