"""Streaming TTS output interface + Voxtral backend.

Architecture: StreamingTTSPlayer is the interface; VoxtralTTSPlayer is the
concrete implementation. CLI and Gradio both import StreamingTTSPlayer and
call play_chunk() as text tokens arrive from run_inference(), then flush() at
the end of generation.

Voxtral backend:
  - Model: mlx-community/Voxtral-4B-TTS-2603-mlx-bf16 (4B, 24kHz)
  - Streaming: model.generate(stream=True, streaming_interval=1.0) yields
    GenerationResult chunks with .audio (mx.array) and .sample_rate
  - Playback: mlx_audio.tts.audio_player.AudioPlayer (sounddevice OutputStream)
  - Dep: mistral-common[audio] for tekken tokenizer

Future candidates (add as VoxCPM2TTSPlayer, etc.):
  - VoxCPM2 (OpenBMB, 48kHz, tier-1): mlx_audio.tts.models.voxcpm already present
  - VibeVoice-Realtime (0.5B, 300ms): mlx_audio.tts.models.vibevoice present
"""

import threading
from typing import Optional

import numpy as np


class StreamingTTSPlayer:
    """Interface for streaming TTS playback."""

    def __init__(self, voice: str = "casual_male", sample_rate: int = 24_000):
        self.voice = voice
        self.sample_rate = sample_rate

    def play_chunk(self, text: str) -> None:
        """Synthesize text chunk and queue for playback."""
        raise NotImplementedError

    def flush(self) -> None:
        """Block until all queued audio has finished playing."""
        raise NotImplementedError

    def close(self) -> None:
        """Release audio resources."""
        pass


# ---------------------------------------------------------------------------
# Voxtral backend
# ---------------------------------------------------------------------------

DEFAULT_VOXTRAL_MODEL = "mlx-community/Voxtral-4B-TTS-2603-mlx-bf16"

# Minimum text length before we bother synthesizing a chunk (avoids firing
# TTS on single punctuation tokens as they stream in from the LM).
_MIN_CHUNK_CHARS = 40


class VoxtralTTSPlayer(StreamingTTSPlayer):
    """Streaming TTS via Voxtral-4B-TTS on MLX + sounddevice playback.

    Usage:
        tts = VoxtralTTSPlayer()
        for text_chunk in run_inference(...):
            tts.play_chunk(text_chunk)
        tts.flush()
        tts.close()

    Buffering strategy: accumulate incoming text until a natural sentence
    boundary (., !, ?) or until _MIN_CHUNK_CHARS is reached, then fire TTS
    in a background thread so generation and playback overlap.
    """

    def __init__(
        self,
        model_path: str = DEFAULT_VOXTRAL_MODEL,
        voice: str = "casual_male",
        streaming_interval: float = 1.0,
        temperature: float = 0.7,
    ):
        super().__init__(voice=voice, sample_rate=24_000)
        self._model_path = model_path
        self._streaming_interval = streaming_interval
        self._temperature = temperature

        self._model = None
        self._player = None
        self._text_buf = ""
        self._pending: list[threading.Thread] = []

    # ------------------------------------------------------------------
    # Lazy load
    # ------------------------------------------------------------------

    def _ensure_loaded(self):
        if self._model is not None:
            return
        print(f"[tts] Loading Voxtral from {self._model_path}...")
        import time
        t0 = time.time()
        from mlx_audio.tts.utils import load as load_tts
        from mlx_audio.tts.audio_player import AudioPlayer
        self._model = load_tts(self._model_path)
        self._player = AudioPlayer(sample_rate=self._model.sample_rate)
        print(f"[tts] Voxtral loaded in {time.time()-t0:.1f}s  "
              f"(sample_rate={self._model.sample_rate})")

    # ------------------------------------------------------------------
    # Synthesis helpers
    # ------------------------------------------------------------------

    def _synthesize_and_play(self, text: str):
        """Generate audio for *text* and queue chunks in AudioPlayer."""
        try:
            for result in self._model.generate(
                text=text,
                voice=self.voice,
                temperature=self._temperature,
                stream=True,
                streaming_interval=self._streaming_interval,
                verbose=False,
            ):
                audio_np = np.array(result.audio, dtype=np.float32)
                if audio_np.size > 0:
                    self._player.queue_audio(audio_np)
        except Exception as e:
            print(f"[tts] synthesis error: {e}")

    def _flush_buf(self, text: str):
        """Fire background synthesis thread for *text*."""
        t = threading.Thread(target=self._synthesize_and_play, args=(text,),
                             daemon=True)
        t.start()
        self._pending.append(t)

    @staticmethod
    def _split_at_boundary(text: str):
        """Return (sentence, remainder) split at last sentence-ending punctuation."""
        for i in range(len(text) - 1, -1, -1):
            if text[i] in ".!?":
                return text[: i + 1].strip(), text[i + 1 :]
        return None, text

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def play_chunk(self, text: str) -> None:
        """Accept a text chunk (typically an LM token or small delta).

        Accumulates text and fires synthesis when a sentence boundary is
        reached or the buffer exceeds _MIN_CHUNK_CHARS.
        """
        self._ensure_loaded()
        self._text_buf += text

        sentence, remainder = self._split_at_boundary(self._text_buf)
        if sentence and len(self._text_buf) >= _MIN_CHUNK_CHARS:
            self._flush_buf(sentence)
            self._text_buf = remainder
        elif len(self._text_buf) >= _MIN_CHUNK_CHARS * 3:
            # Hard cap: flush even without punctuation to keep latency low
            self._flush_buf(self._text_buf)
            self._text_buf = ""

    def flush(self) -> None:
        """Synthesize any remaining buffered text and wait for playback to finish."""
        self._ensure_loaded()
        if self._text_buf.strip():
            self._flush_buf(self._text_buf.strip())
            self._text_buf = ""
        # Wait for all synthesis threads to finish queueing audio
        for t in self._pending:
            t.join()
        self._pending.clear()
        # Wait for AudioPlayer to drain
        if self._player and self._player.playing:
            self._player.wait_for_drain()

    def close(self) -> None:
        """Stop playback and release resources."""
        if self._player:
            try:
                self._player.stop_stream()
            except Exception:
                pass
            self._player = None
        self._model = None
