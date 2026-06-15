#!/usr/bin/env python3
"""
Gemma 4 Unified realtime audio — mic → Gemma4 → streamed text.

Model: gemma-4-12B-it (Unified architecture, encoder-free audio)
Audio: raw waveform sliced at 640 samples/token (40ms @ 16kHz), directly projected
       into LM embedding space. No Conformer tower — very low latency.
Speed: ~30 tok/s on M4 Max (4-bit quantized, after Metal compilation warmup)

Usage:
    # Mic input (default 5s):
    python gemma4_realtime_audio.py

    # Mic with custom duration and prompt:
    python gemma4_realtime_audio.py --seconds 8 --prompt "What did I say?"

    # Audio file input:
    python gemma4_realtime_audio.py --audio-file speech.wav

    # Use a different model:
    python gemma4_realtime_audio.py --model mlx-community/gemma-4-12B-it-8bit

Press Ctrl+C during recording to stop early and generate immediately.
"""

import argparse
import sys
import time

import numpy as np

DEFAULT_MODEL = "mlx-community/gemma-4-12B-it-4bit"
SAMPLE_RATE = 16_000
DEFAULT_SECONDS = 5
DEFAULT_MAX_TOKENS = 256
DEFAULT_TEMP = 0.3
DEFAULT_PROMPT = "Transcribe what you hear, then respond to it."


def record_mic(seconds: float, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Record from default mic, return float32 mono waveform."""
    import sounddevice as sd

    print(f"\n[mic] Recording {seconds}s... speak now (Ctrl+C to stop early)")
    frames = []

    def callback(indata, frame_count, time_info, status):
        frames.append(indata[:, 0].copy())

    with sd.InputStream(
        samplerate=sample_rate,
        channels=1,
        dtype="float32",
        callback=callback,
        blocksize=int(sample_rate * 0.1),
    ):
        try:
            sd.sleep(int(seconds * 1000))
        except KeyboardInterrupt:
            print("\n[mic] Stopped early.")

    audio = np.concatenate(frames) if frames else np.zeros(sample_rate, dtype=np.float32)
    print(f"[mic] {len(audio)/sample_rate:.2f}s captured")
    return audio


def load_audio_file(path: str, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    from mlx_vlm.utils import load_audio
    audio = load_audio(path, sr=sample_rate)
    print(f"[file] {len(audio)/sample_rate:.2f}s from {path}")
    return audio


def build_prompt(processor, model_config, user_text: str) -> str:
    """Build chat-template prompt with audio placeholder."""
    from mlx_vlm.prompt_utils import apply_chat_template
    audio_token = getattr(processor, "audio_token", "<|audio|>")
    content = f"{audio_token}\n{user_text}"
    return apply_chat_template(processor, model_config, content, num_images=0)


def main():
    parser = argparse.ArgumentParser(description="Gemma4 realtime audio on M4 Max")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help="Model repo id or local path")
    parser.add_argument("--seconds", type=float, default=DEFAULT_SECONDS,
                        help="Mic recording duration in seconds")
    parser.add_argument("--audio-file", default=None,
                        help="Audio file path instead of mic")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT,
                        help="Text instruction to accompany the audio")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--temp", type=float, default=DEFAULT_TEMP)
    parser.add_argument("--warmup", action="store_true",
                        help="Run a silent warmup pass to pre-compile Metal kernels")
    args = parser.parse_args()

    # ── Load model ──────────────────────────────────────────────────────────
    print(f"[init] Loading {args.model}...")
    t_load = time.time()
    from mlx_vlm import load
    from mlx_vlm.generate import stream_generate

    model, processor = load(args.model)
    print(f"[init] Loaded in {time.time()-t_load:.1f}s")
    print(f"[init] model_type: {model.model_type}")
    print(f"[init] embed_audio: {getattr(model, 'embed_audio', None) is not None}")

    if getattr(model, "embed_audio", None) is None:
        print("[ERROR] No embed_audio — checkpoint missing audio weights.")
        sys.exit(1)

    # ── Optional Metal warmup ────────────────────────────────────────────────
    if args.warmup:
        print("[warmup] Pre-compiling Metal kernels with silent audio...")
        dummy = np.zeros(SAMPLE_RATE, dtype=np.float32)
        warmup_prompt = build_prompt(processor, model.config, ".")
        for _ in stream_generate(
            model, processor, prompt=warmup_prompt, audio=[dummy],
            max_tokens=4, temperature=1.0, verbose=False,
        ):
            pass
        print("[warmup] Done.")

    # ── Get audio ────────────────────────────────────────────────────────────
    if args.audio_file:
        audio = load_audio_file(args.audio_file)
    else:
        audio = record_mic(args.seconds)

    n_audio_tokens = len(audio) // 640  # 640 samples/token @ 16kHz
    print(f"[audio] {len(audio)} samples → ~{n_audio_tokens} audio tokens")

    # ── Build prompt ─────────────────────────────────────────────────────────
    prompt = build_prompt(processor, model.config, args.prompt)
    print(f"\n[prompt] {repr(args.prompt)}")

    # ── Generate ─────────────────────────────────────────────────────────────
    print(f"[generate] max_tokens={args.max_tokens}  temp={args.temp}\n")
    print("=" * 60)

    t_gen = time.time()
    n_tok = 0
    full = ""

    try:
        for token in stream_generate(
            model, processor,
            prompt=prompt,
            audio=[audio],
            max_tokens=args.max_tokens,
            temperature=args.temp,
            verbose=False,
        ):
            chunk = token.text if hasattr(token, "text") else str(token)
            print(chunk, end="", flush=True)
            full += chunk
            n_tok += 1
    except KeyboardInterrupt:
        print("\n[interrupted]")

    elapsed = time.time() - t_gen
    print(f"\n{'='*60}")
    print(f"[done] {n_tok} tokens  {elapsed:.2f}s  {n_tok/max(elapsed,0.001):.1f} tok/s")


if __name__ == "__main__":
    main()
