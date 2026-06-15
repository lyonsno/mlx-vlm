# Gemma 4 Realtime Audio

Realtime mic → text (→ optional speech) on Apple Silicon via Gemma 4's built-in audio modality.
No external STT model. No Conformer tower. Raw waveform, directly into the LM.

**~30 tok/s on M4 Max with the 4-bit checkpoint.**

---

## How it works

Gemma 4 Unified (`gemma4_unified`) is encoder-free: raw audio is sliced at 640 samples/token
(40 ms @ 16 kHz) and linearly projected into the LM's hidden dim via `embed_audio`. The model
sees audio tokens and text tokens in the same sequence — no separate audio encoder, no latency
penalty before first token.

The prompt format is one `<|audio|>` placeholder, expanded by the processor at generation time
to match the waveform duration:

```
<|audio|>
Transcribe what you hear, then respond to it.
```

Thinking tokens (`<|channel>thought...</channel|>`) are stripped automatically.

---

## Setup

```sh
# From the mlx-vlm repo root
pip install -e ".[audio]"          # or: pip install mlx-vlm[audio]
pip install sounddevice scipy       # mic recording + resampling
pip install "mistral-common[audio]" # required for Voxtral TTS
```

The 4-bit Gemma 4 checkpoint (~7 GB) downloads automatically on first run from
`mlx-community/gemma-4-12B-it-4bit`.

---

## CLI — mic input

```sh
# Record 5s from mic, generate response
python gemma4_realtime_audio.py

# Custom duration and prompt
python gemma4_realtime_audio.py --seconds 8 --prompt "Summarise what I said."

# Load from audio file instead of mic
python gemma4_realtime_audio.py --audio-file speech.wav

# Different model (8-bit, lower quality but less VRAM)
python gemma4_realtime_audio.py --model mlx-community/gemma-4-12B-it-8bit

# Pre-compile Metal kernels before recording (reduces first-run latency)
python gemma4_realtime_audio.py --warmup

# Speak the response via Voxtral TTS
python gemma4_realtime_audio.py --tts
python gemma4_realtime_audio.py --tts --voice neutral_female
python gemma4_realtime_audio.py --tts --tts-model mlx-community/Voxtral-4B-TTS-2603-mlx-4bit
```

Press **Ctrl+C** during recording to stop early and generate immediately.

### Available TTS voices

`casual_male` (default), `casual_female`, `cheerful_female`, `neutral_male`, `neutral_female`,
`fr_male`, `fr_female`, `es_male`, `es_female`, `de_male`, `de_female`, `it_male`, `it_female`,
`pt_male`, `pt_female`, `nl_male`, `nl_female`, `ar_male`, `hi_male`, `hi_female`

---

## Gradio — push-to-talk web UI

```sh
python gemma4_audio_gradio.py
# → http://localhost:7860

# Different port (if 7860 is in use)
python gemma4_audio_gradio.py --port 7861

# Public share link (Gradio tunnel)
python gemma4_audio_gradio.py --share
```

Click the microphone button, speak, release — the response streams in automatically.
Thinking tokens are stripped before display.

---

## Python API

```python
from mlx_vlm.tools.gemma4_audio.core import load_model
from mlx_vlm.tools.gemma4_audio.audio import load_audio_file, record_mic
from mlx_vlm.tools.gemma4_audio.prompt import build_prompt
from mlx_vlm.tools.gemma4_audio.inference import run_inference

model, processor = load_model("mlx-community/gemma-4-12B-it-4bit")

# From file
audio = load_audio_file("speech.wav")

# Or from mic
audio = record_mic(seconds=5)

prompt_fn = lambda text: build_prompt(processor, model.config, text)

for text in run_inference(model, processor, audio,
                          user_prompt="What did I say?",
                          max_tokens=256, temperature=0.3,
                          prompt_builder=prompt_fn):
    print(text, end="\r")   # accumulated text, each iteration
```

### With TTS

```python
from mlx_vlm.tools.gemma4_audio.tts import VoxtralTTSPlayer

tts = VoxtralTTSPlayer(voice="casual_male")

full = ""
for text in run_inference(...):
    tts.play_chunk(text[len(full):])
    full = text

tts.flush()   # synthesize + play on main thread (MLX thread-local Metal stream)
tts.close()
```

---

## Package layout

```
mlx_vlm/tools/gemma4_audio/
    constants.py   — DEFAULT_MODEL, SAMPLE_RATE, AUDIO_SAMPLES_PER_TOKEN, …
    core.py        — load_model() with embed_audio guard
    prompt.py      — build_prompt()
    audio.py       — record_mic, load_audio_file, process_gradio_audio
    inference.py   — run_inference() generator + ThinkingStreamState stripping
    tts.py         — VoxtralTTSPlayer (Voxtral-4B-TTS, 24 kHz, streaming)

gemma4_realtime_audio.py   — CLI entry point
gemma4_audio_gradio.py     — Gradio entry point
```

---

## Models

| Checkpoint | Size | Notes |
|---|---|---|
| `mlx-community/gemma-4-12B-it-4bit` | ~7 GB | Default. Fast, good quality. |
| `mlx-community/gemma-4-12B-it-8bit` | ~13 GB | Higher quality, more VRAM. |
| `mlx-community/Voxtral-4B-TTS-2603-mlx-bf16` | ~8 GB | TTS default. Best quality. |
| `mlx-community/Voxtral-4B-TTS-2603-mlx-4bit` | ~2 GB | TTS fast path, less VRAM. |

Both Gemma 4 and Voxtral loaded simultaneously: budget ~9–10 GB unified memory for 4-bit + 4-bit.

---

## Known limitations

- **Sequential generation + TTS**: MLX Metal streams are thread-local. Generation completes first, then TTS synthesizes on the main thread. True overlap requires multiprocessing (separate Metal context per process) — not yet implemented.
- **No VAD**: fixed-duration recording window. True realtime would need speech-end detection (e.g. OpenWakeWord + silero-VAD).
- **Channel tokens in CLI**: the raw CLI (`gemma4_realtime_audio.py`) shows raw token output including any `<|channel>thought` spans. The Gradio UI strips them via `ThinkingStreamState`.

---

## Requirements

- Apple Silicon Mac (M1 or later; M4 Max tested at ~30 tok/s)
- macOS 13+, Python 3.10+
- `mlx-vlm`, `mlx-audio`, `sounddevice`, `scipy`
- `mistral-common[audio]` — required for Voxtral TTS tokenizer
