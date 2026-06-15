"""TTS backends for gemma4 audio loop.

All backends share StreamingTTSPlayer interface:
    tts = <Backend>TTSPlayer(...)
    for text in run_inference(...):
        tts.play_chunk(text)   # buffers text, no-op until flush
    tts.flush()                # synthesize + play on calling (main) thread
    tts.close()

MLX Metal streams are thread-local — synthesis must run on the same thread
that holds the GPU context (the main thread). Do not call flush() from a
background thread.

Backends
--------
VoxtralTTSPlayer  — Voxtral-4B-TTS, 24kHz, streaming chunks (low perceived latency)
VoxCPMTTSPlayer   — VoxCPM1.5, 44.1kHz, batch decode (higher quality, no streaming)
"""

import numpy as np


class StreamingTTSPlayer:
    """Interface for TTS playback after LM generation."""

    def __init__(self, voice: str = "casual_male", sample_rate: int = 24_000):
        self.voice = voice
        self.sample_rate = sample_rate

    def play_chunk(self, text: str) -> None:
        """Accept a text delta from the LM (buffered until flush)."""
        raise NotImplementedError

    def flush(self) -> None:
        """Synthesize all buffered text and block until playback finishes."""
        raise NotImplementedError

    def close(self) -> None:
        """Release audio resources."""
        pass


# ---------------------------------------------------------------------------
# Voxtral backend
# ---------------------------------------------------------------------------

DEFAULT_VOXTRAL_MODEL = "mlx-community/Voxtral-4B-TTS-2603-mlx-bf16"


class VoxtralTTSPlayer(StreamingTTSPlayer):
    """TTS via Voxtral-4B-TTS on MLX.

    play_chunk() accumulates text deltas. flush() runs synthesis on the
    calling (main) thread — required because MLX Metal streams are
    thread-local and cannot be used from background threads.

    Voxtral streams audio chunks during synthesis via stream=True; each chunk
    is fed to AudioPlayer.queue_audio() so playback begins before synthesis
    completes.
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

    def play_chunk(self, text: str) -> None:
        """Buffer text delta — synthesis happens in flush() on the main thread."""
        self._text_buf += text

    def flush(self) -> None:
        """Synthesize buffered text and wait for playback to drain."""
        text = self._text_buf.strip()
        self._text_buf = ""
        if not text:
            return
        self._ensure_loaded()
        print(f"[tts] Synthesizing {len(text)} chars...")
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
            return
        # Wait for AudioPlayer to drain
        if self._player.playing:
            self._player.wait_for_drain()

    def close(self) -> None:
        if self._player:
            try:
                self._player.stop_stream()
            except Exception:
                pass
            self._player = None
        self._model = None


# ---------------------------------------------------------------------------
# VoxCPM backend
# ---------------------------------------------------------------------------

DEFAULT_VOXCPM_MODEL = "mlx-community/VoxCPM1.5"


class VoxCPMTTSPlayer(StreamingTTSPlayer):
    """TTS via VoxCPM1.5 on MLX.

    VoxCPM generates the entire utterance in one batch decode (no streaming),
    then plays it via sounddevice. Quality is tier-1 at 44.1kHz; latency is
    higher than Voxtral because there are no intermediate audio chunks.

    Supports zero-shot synthesis (text only) and voice cloning
    (ref_audio + ref_text). Voice cloning requires a reference WAV file.
    """

    def __init__(
        self,
        model_path: str = DEFAULT_VOXCPM_MODEL,
        ref_audio: str | None = None,
        ref_text: str | None = None,
        inference_timesteps: int = 10,
        cfg_value: float = 2.0,
    ):
        super().__init__(voice="default", sample_rate=44_100)
        self._model_path = model_path
        self._ref_audio = ref_audio
        self._ref_text = ref_text
        self._inference_timesteps = inference_timesteps
        self._cfg_value = cfg_value
        self._model = None
        self._text_buf = ""

    def _ensure_loaded(self):
        if self._model is not None:
            return
        print(f"[tts] Loading VoxCPM from {self._model_path}...")
        import time
        t0 = time.time()
        from mlx_audio.tts.utils import load as load_tts
        self._model = load_tts(self._model_path)
        self.sample_rate = self._model.sample_rate
        print(f"[tts] VoxCPM loaded in {time.time()-t0:.1f}s  "
              f"(sample_rate={self._model.sample_rate})")

    def play_chunk(self, text: str) -> None:
        """Buffer text delta — synthesis happens in flush() on the main thread."""
        self._text_buf += text

    def flush(self) -> None:
        """Synthesize buffered text and play via sounddevice."""
        text = self._text_buf.strip()
        self._text_buf = ""
        if not text:
            return
        self._ensure_loaded()
        print(f"[tts] Synthesizing {len(text)} chars (VoxCPM)...")
        try:
            import sounddevice as sd
            for result in self._model.generate(
                text=text,
                ref_audio=self._ref_audio,
                ref_text=self._ref_text,
                inference_timesteps=self._inference_timesteps,
                cfg_value=self._cfg_value,
            ):
                audio_np = np.array(result.audio, dtype=np.float32)
                if audio_np.size > 0:
                    sd.play(audio_np, samplerate=result.sample_rate)
                    sd.wait()
        except Exception as e:
            print(f"[tts] synthesis error: {e}")

    def close(self) -> None:
        self._model = None
