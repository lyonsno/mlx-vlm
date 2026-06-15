#!/usr/bin/env python3
"""
Gemma 4 Unified realtime audio — mic → Gemma4 → streamed text (→ optional TTS).

Model: gemma-4-12B-it (Unified architecture, encoder-free audio)
Audio: raw waveform sliced at 640 samples/token (40ms @ 16kHz), directly projected
       into LM embedding space. No Conformer tower — very low latency.
Speed: ~30 tok/s on M4 Max (4-bit quantized, after Metal compilation warmup)

Usage:
    python gemma4_realtime_audio.py
    python gemma4_realtime_audio.py --seconds 8 --prompt "What did I say?"
    python gemma4_realtime_audio.py --audio-file speech.wav
    python gemma4_realtime_audio.py --model mlx-community/gemma-4-12B-it-8bit
    python gemma4_realtime_audio.py --tts                    # speak the response
    python gemma4_realtime_audio.py --tts --voice casual_female

Press Ctrl+C during recording to stop early and generate immediately.
"""

import argparse
import time

import numpy as np

from mlx_vlm.tools.gemma4_audio.constants import (
    AUDIO_SAMPLES_PER_TOKEN,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL,
    DEFAULT_PROMPT,
    DEFAULT_SECONDS,
    DEFAULT_TEMP,
    SAMPLE_RATE,
)
from mlx_vlm.tools.gemma4_audio.core import load_model
from mlx_vlm.tools.gemma4_audio.audio import load_audio_file, record_mic
from mlx_vlm.tools.gemma4_audio.prompt import build_prompt
from mlx_vlm.tools.gemma4_audio.inference import run_inference
from mlx_vlm.tools.gemma4_audio.tts import (
    DEFAULT_VOXCPM_MODEL,
    DEFAULT_VOXTRAL_MODEL,
    DEFAULT_VIBEVOICE_MODEL,
    VoxCPMTTSPlayer,
    VoxtralTTSPlayer,
    VibeVoiceTTSPlayer,
)


def main():
    parser = argparse.ArgumentParser(description="Gemma4 realtime audio on M4 Max")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--seconds", type=float, default=DEFAULT_SECONDS,
                        help="Mic recording duration in seconds")
    parser.add_argument("--audio-file", default=None,
                        help="Audio file path instead of mic")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--temp", type=float, default=DEFAULT_TEMP)
    parser.add_argument("--warmup", action="store_true",
                        help="Run a silent warmup pass to pre-compile Metal kernels")
    parser.add_argument("--tts", action="store_true",
                        help="Speak the response via TTS")
    parser.add_argument("--tts-backend", default="voxtral",
                        choices=["voxtral", "voxcpm", "vibevoice"],
                        help="TTS backend: voxtral (streaming, 24kHz), voxcpm (batch, 44.1kHz), or vibevoice (MLX diffusion, 24kHz)")
    parser.add_argument("--tts-model", default=None,
                        help="Override TTS model path/repo (default per backend)")
    parser.add_argument("--voice", default="casual_male",
                        help="Voxtral voice preset (casual_male, neutral_female, …)")
    parser.add_argument("--ref-audio", default=None,
                        help="VoxCPM: reference WAV for voice cloning")
    parser.add_argument("--ref-text", default=None,
                        help="VoxCPM: transcript of --ref-audio")
    parser.add_argument("--voice-path", default=None,
                        help="VibeVoice: path to voice prompt .pt file")
    parser.add_argument("--voice-name", default="en-Davis_man",
                        help="VibeVoice: voice preset name (en-Davis_man, en-Emma_woman, …)")
    parser.add_argument("--diffusion-steps", type=int, default=5,
                        help="VibeVoice: number of diffusion denoising steps (default 5)")
    parser.add_argument("--cfg-scale", type=float, default=1.5,
                        help="VibeVoice: classifier-free guidance scale (default 1.5)")
    args = parser.parse_args()

    model, processor = load_model(args.model)

    if args.warmup:
        print("[warmup] Pre-compiling Metal kernels with silent audio...")
        dummy = np.zeros(SAMPLE_RATE, dtype=np.float32)
        prompt_fn = lambda text: build_prompt(processor, model.config, text)
        for _ in run_inference(model, processor, dummy, ".",
                               max_tokens=4, temperature=1.0,
                               prompt_builder=prompt_fn):
            pass
        print("[warmup] Done.")

    if args.audio_file:
        audio = load_audio_file(args.audio_file)
    else:
        audio = record_mic(args.seconds)

    n_audio_tokens = len(audio) // AUDIO_SAMPLES_PER_TOKEN
    print(f"[audio] {len(audio)} samples → ~{n_audio_tokens} audio tokens")

    prompt_fn = lambda text: build_prompt(processor, model.config, text)
    print(f"\n[prompt] {repr(args.prompt)}")
    print(f"[generate] max_tokens={args.max_tokens}  temp={args.temp}\n")
    print("=" * 60)

    if args.tts:
        if args.tts_backend == "vibevoice":
            tts = VibeVoiceTTSPlayer(
                model_path=args.tts_model or DEFAULT_VIBEVOICE_MODEL,
                voice_path=args.voice_path,
                voice_name=args.voice_name,
                cfg_scale=args.cfg_scale,
                num_diffusion_steps=args.diffusion_steps,
            )
        elif args.tts_backend == "voxcpm":
            tts = VoxCPMTTSPlayer(
                model_path=args.tts_model or DEFAULT_VOXCPM_MODEL,
                ref_audio=args.ref_audio,
                ref_text=args.ref_text,
            )
        else:
            tts = VoxtralTTSPlayer(
                model_path=args.tts_model or DEFAULT_VOXTRAL_MODEL,
                voice=args.voice,
            )
    else:
        tts = None

    t_gen = time.time()
    full = ""

    try:
        for text in run_inference(model, processor, audio, args.prompt,
                                  args.max_tokens, args.temp, prompt_fn):
            delta = text[len(full):]
            print(delta, end="", flush=True)
            full = text
            if tts and delta:
                tts.play_chunk(delta)
    except KeyboardInterrupt:
        print("\n[interrupted]")

    elapsed = time.time() - t_gen
    print(f"\n{'='*60}")
    print(f"[done] {elapsed:.2f}s")

    if tts:
        print("[tts] Flushing remaining speech...")
        tts.flush()
        tts.close()


if __name__ == "__main__":
    main()
