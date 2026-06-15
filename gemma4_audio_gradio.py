#!/usr/bin/env python3
"""
Gemma 4 Unified audio — push-to-talk Gradio front-end.

Record from mic, get streamed text back. Thinking tokens stripped automatically.

Usage:
    python gemma4_audio_gradio.py
    # then open http://localhost:7860

    python gemma4_audio_gradio.py --model mlx-community/gemma-4-12B-it-8bit
    python gemma4_audio_gradio.py --port 7861 --share
"""

import argparse
import sys

import numpy as np
import scipy.signal

DEFAULT_MODEL = "mlx-community/gemma-4-12B-it-4bit"
SAMPLE_RATE = 16_000
DEFAULT_MAX_TOKENS = 512
DEFAULT_TEMP = 0.3
DEFAULT_PROMPT = "Transcribe what you hear, then respond to it."

# Lazily loaded globals
_model = None
_processor = None


def _load_model(model_path: str):
    global _model, _processor
    if _model is not None:
        return
    print(f"[init] Loading {model_path}...")
    from mlx_vlm import load
    _model, _processor = load(model_path)
    if getattr(_model, "embed_audio", None) is None:
        print("[ERROR] No embed_audio — checkpoint missing audio weights.")
        sys.exit(1)
    print(f"[init] Ready. model_type={_model.model_type}")


def process_gradio_audio(audio_tuple) -> np.ndarray:
    """Convert Gradio mic output (sr, array) to float32 mono 16kHz ndarray."""
    sr, data = audio_tuple
    # Mono: take left channel if stereo
    if data.ndim > 1:
        data = data[:, 0]
    data = data.astype(np.float32)
    # Resample to 16kHz if needed
    if sr != SAMPLE_RATE:
        n_samples = int(len(data) * SAMPLE_RATE / sr)
        data = scipy.signal.resample(data, n_samples)
    # Normalize to [-1, 1]
    peak = np.max(np.abs(data))
    if peak > 1e-6:
        data = data / peak
    return data


def build_prompt(user_text: str) -> str:
    """Build chat-template prompt with audio placeholder."""
    from mlx_vlm.prompt_utils import apply_chat_template
    audio_token = getattr(_processor, "audio_token", "<|audio|>")
    content = f"{audio_token}\n{user_text}"
    return apply_chat_template(_processor, _model.config, content, num_images=0)


def run_inference(audio_tuple, user_prompt: str, max_tokens: int, temperature: float):
    """Generator: yields accumulated response text as tokens arrive."""
    from mlx_vlm.generate import stream_generate
    from mlx_vlm.server.responses_state import ThinkingStreamState

    if audio_tuple is None:
        yield "No audio recorded. Hold the microphone button and speak."
        return

    audio = process_gradio_audio(audio_tuple)
    n_audio_tokens = len(audio) // 640
    print(f"[audio] {len(audio)} samples → ~{n_audio_tokens} audio tokens")

    if n_audio_tokens == 0:
        yield "Audio too short — nothing to process."
        return

    prompt = build_prompt(user_prompt)
    state = ThinkingStreamState()
    acc = ""

    try:
        for token in stream_generate(
            _model,
            _processor,
            prompt=prompt,
            audio=[audio],
            max_tokens=max_tokens,
            temperature=temperature,
            verbose=False,
        ):
            chunk = token.text if hasattr(token, "text") else str(token)
            delta_obj = state.feed(chunk)
            # Only yield visible content, not reasoning tokens
            if delta_obj.content:
                acc += delta_obj.content
                yield acc
    except Exception as e:
        yield f"[error] {e}"


def build_demo(model_path: str, max_tokens: int, temperature: float):
    import gradio as gr

    _load_model(model_path)

    def infer(audio_tuple, user_prompt, max_tok, temp):
        yield from run_inference(audio_tuple, user_prompt, int(max_tok), float(temp))

    with gr.Blocks(title="Gemma 4 Audio", theme=gr.themes.Soft()) as demo:
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
                output_box = gr.Textbox(
                    label="Response",
                    lines=16,
                    show_copy_button=True,
                )

        submit_btn.click(
            fn=infer,
            inputs=[audio_in, prompt_box, max_tok_slider, temp_slider],
            outputs=output_box,
        )
        # Also trigger on audio stop-recording for push-to-talk feel
        audio_in.stop_recording(
            fn=infer,
            inputs=[audio_in, prompt_box, max_tok_slider, temp_slider],
            outputs=output_box,
        )

    return demo


def main():
    parser = argparse.ArgumentParser(description="Gemma 4 audio Gradio front-end")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help="Model repo id or local path")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--share", action="store_true",
                        help="Create a public Gradio share link")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--temp", type=float, default=DEFAULT_TEMP)
    args = parser.parse_args()

    demo = build_demo(args.model, args.max_tokens, args.temp)
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
    )


if __name__ == "__main__":
    main()
