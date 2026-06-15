"""Streaming TTS output interface.

Stub — implementation TBD pending operator TTS candidate selection.
Candidates (2026-06-15 scan):
  - Voxtral TTS (Mistral): 70ms first audio, streaming, Rust ports for macOS
    https://github.com/second-state/voxtral_tts_rs
    https://github.com/mudler/voxtral-tts.c
  - VoxCPM2 (OpenBMB): tier-1 quality 48kHz, RTF 0.3, Rust port for Metal
    https://github.com/OpenBMB/VoxCPM
    https://github.com/mii-nipah/voxcpm-rs
  - Qwen3-TTS (Alibaba): native MLX via mlx-audio, 97ms first audio, fast path
    https://github.com/QwenLM/Qwen3-TTS

When implemented, replace StreamingTTSPlayer with a concrete backend class
that matches this interface. Entry points (CLI, Gradio) import StreamingTTSPlayer
and call play_chunk() as tokens arrive from run_inference().
"""


class StreamingTTSPlayer:
    """Interface for streaming TTS playback. Not yet implemented."""

    def __init__(self, voice: str = "default", sample_rate: int = 24_000):
        self.voice = voice
        self.sample_rate = sample_rate

    def play_chunk(self, text: str) -> None:
        """Synthesize text chunk and queue for playback."""
        raise NotImplementedError("TTS backend not yet selected/integrated.")

    def flush(self) -> None:
        """Block until all queued audio has finished playing."""
        raise NotImplementedError()

    def close(self) -> None:
        """Release audio resources."""
        pass
