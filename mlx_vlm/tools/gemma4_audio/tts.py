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
VoxtralTTSPlayer   — Voxtral-4B-TTS, 24kHz, streaming chunks (low perceived latency)
VoxCPMTTSPlayer    — VoxCPM1.5, 44.1kHz, batch decode (higher quality, no streaming)
VibeVoiceTTSPlayer — VibeVoice-Realtime-0.5B, 24kHz, MLX native diffusion TTS
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


# ---------------------------------------------------------------------------
# VibeVoice backend
# ---------------------------------------------------------------------------

DEFAULT_VIBEVOICE_MODEL = "microsoft/VibeVoice-Realtime-0.5B"
DEFAULT_VIBEVOICE_VOICE = "en-Davis_man"


class VibeVoiceTTSPlayer(StreamingTTSPlayer):
    """TTS via VibeVoice-Realtime-0.5B on MLX — true streaming.

    Incrementally tokenizes incoming text. When enough tokens accumulate
    to fill a text window (5 tokens), immediately runs diffusion for that
    window's speech tokens (6 latents) and plays the audio. Audio starts
    playing while Gemma 4 is still generating text.

    play_chunk() tokenizes text deltas and fires generation windows.
    flush() handles the final partial window + drain.
    """

    def __init__(
        self,
        model_path: str = DEFAULT_VIBEVOICE_MODEL,
        voice_path: str = None,
        voice_name: str = DEFAULT_VIBEVOICE_VOICE,
        cfg_scale: float = 1.5,
        num_diffusion_steps: int = 5,
    ):
        super().__init__(voice=voice_name, sample_rate=24_000)
        self._model_path = model_path
        self._voice_path = voice_path
        self._voice_name = voice_name
        self._cfg_scale = cfg_scale
        self._num_diffusion_steps = num_diffusion_steps
        self._model = None
        self._voice_prompt = None
        self._tokenizer = None
        self._sd_stream = None
        # Streaming state
        self._token_buf = []       # accumulated token ids not yet windowed
        self._text_buf = ""        # raw text for re-tokenization
        self._gen_state = None     # persistent generation state across windows
        self._initialized = False

    def _ensure_loaded(self):
        if self._model is not None:
            return
        import time
        print(f"[tts] Loading VibeVoice from {self._model_path}...")
        t0 = time.time()
        from .vibevoice_mlx import load_vibevoice, convert_voice_prompt
        self._model, self._config = load_vibevoice(self._model_path)

        voice_path = self._voice_path
        if voice_path is None:
            import os
            candidates = [
                f"/private/tmp/vibevoice-ref/demo/voices/streaming_model/{self._voice_name}.pt",
                os.path.expanduser(f"~/.cache/vibevoice/voices/{self._voice_name}.pt"),
            ]
            for c in candidates:
                if os.path.exists(c):
                    voice_path = c
                    break
            if voice_path is None:
                try:
                    from huggingface_hub import hf_hub_download
                    voice_path = hf_hub_download(
                        "microsoft/VibeVoice-Realtime-0.5B",
                        f"demo/voices/streaming_model/{self._voice_name}.pt",
                    )
                except Exception:
                    raise FileNotFoundError(
                        f"Voice file not found for '{self._voice_name}'. "
                        f"Provide --voice-path or place .pt file in "
                        f"/private/tmp/vibevoice-ref/demo/voices/streaming_model/"
                    )

        print(f"[tts] Loading voice prompt: {voice_path}")
        self._voice_prompt = convert_voice_prompt(voice_path)

        from transformers import AutoTokenizer
        self._tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
        print(f"[tts] VibeVoice loaded in {time.time()-t0:.1f}s")

    def _init_gen_state(self):
        """Initialize persistent generation state from voice prompt."""
        if self._initialized:
            return
        import copy
        from .vibevoice_mlx import KVCache, StreamingCache
        import mlx.core as mx

        def _restore_cache(kv_list):
            caches = []
            for k, v in kv_list:
                c = KVCache()
                c.keys = k
                c.values = v
                c.offset = k.shape[2]
                caches.append(c)
            return caches

        vp = copy.deepcopy(self._voice_prompt)
        self._gen_state = {
            "lm_cache": _restore_cache(vp["lm"]["kv_cache"]),
            "tts_lm_cache": _restore_cache(vp["tts_lm"]["kv_cache"]),
            "neg_lm_cache": _restore_cache(vp["neg_lm"]["kv_cache"]),
            "neg_tts_lm_cache": _restore_cache(vp["neg_tts_lm"]["kv_cache"]),
            "lm_hidden": vp["lm"]["last_hidden_state"],
            "tts_lm_hidden": vp["tts_lm"]["last_hidden_state"],
            "neg_tts_lm_hidden": vp["neg_tts_lm"]["last_hidden_state"],
            "acoustic_cache": StreamingCache(),
        }

        import sounddevice as sd
        self._sd_stream = sd.OutputStream(samplerate=self.sample_rate, channels=1, dtype='float32')
        self._sd_stream.start()
        self._initialized = True

    def _run_window(self, token_ids):
        """Run one text window through the pipeline. Plays audio immediately."""
        import mlx.core as mx
        from .vibevoice_mlx import TTS_SPEECH_WINDOW_SIZE

        m = self._model
        s = self._gen_state
        text_ids = mx.array([token_ids])

        # Base LM forward
        cur_embeds = m.language_model.embed_tokens(text_ids)
        s["lm_hidden"] = m.language_model(inputs_embeds=cur_embeds, cache=s["lm_cache"])
        mx.eval(s["lm_hidden"])

        # TTS LM forward with spliced LM hidden states
        tts_embeds = m.tts_language_model.embed_tokens(text_ids)
        splice_start = tts_embeds.shape[1] - s["lm_hidden"].shape[1]
        if splice_start > 0:
            tts_embeds = mx.concatenate([tts_embeds[:, :splice_start], s["lm_hidden"]], axis=1)
        else:
            tts_embeds = s["lm_hidden"]
        type_embed = m.tts_input_types(mx.ones(tts_embeds.shape[:2], dtype=mx.int32))
        tts_embeds = tts_embeds + type_embed
        s["tts_lm_hidden"] = m.tts_language_model(inputs_embeds=tts_embeds, cache=s["tts_lm_cache"])
        mx.eval(s["tts_lm_hidden"])

        # Generate speech tokens for this window
        for _ in range(TTS_SPEECH_WINDOW_SIZE):
            pos_cond = s["tts_lm_hidden"][:, -1:, :].reshape(1, -1)
            neg_cond = s["neg_tts_lm_hidden"][:, -1:, :].reshape(1, -1)

            speech_latent = m.sample_speech_tokens(
                pos_cond, neg_cond,
                cfg_scale=self._cfg_scale,
                num_steps=self._num_diffusion_steps,
            )

            # Decode to audio
            scaled = speech_latent.reshape(1, 1, -1) / m.speech_scaling_factor - m.speech_bias_factor
            scaled_for_decode = mx.transpose(scaled, (0, 2, 1))
            audio_chunk = m.acoustic_decoder(scaled_for_decode, cache=s["acoustic_cache"])
            mx.eval(audio_chunk)

            # Play immediately
            audio_np = np.array(audio_chunk.reshape(-1), dtype=np.float32)
            if audio_np.size > 0 and self._sd_stream is not None:
                self._sd_stream.write(audio_np.reshape(-1, 1))

            # Feed speech back into TTS LM
            acoustic_embed = m.acoustic_connector(speech_latent.reshape(1, 1, -1))
            type_embed_speech = m.tts_input_types(mx.zeros((1, 1), dtype=mx.int32))
            tts_input = acoustic_embed + type_embed_speech

            s["tts_lm_hidden"] = m.tts_language_model(inputs_embeds=tts_input, cache=s["tts_lm_cache"])
            s["neg_tts_lm_hidden"] = m.tts_language_model(inputs_embeds=tts_input, cache=s["neg_tts_lm_cache"])
            mx.eval(s["tts_lm_hidden"], s["neg_tts_lm_hidden"])

            # Check EOS
            eos_logit = m.eos_classifier(s["tts_lm_hidden"][:, -1, :])
            import mlx.core as mx
            if mx.sigmoid(eos_logit).item() > 0.5:
                return True  # finished

        return False  # not finished

    def play_chunk(self, text: str) -> None:
        """Tokenize incoming text and fire generation windows as they fill."""
        self._ensure_loaded()
        self._init_gen_state()

        self._text_buf += text
        # Re-tokenize the full buffer to handle token boundary issues
        tokens = self._tokenizer.encode(self._text_buf, add_special_tokens=False)

        # Keep a margin — don't consume the last few chars in case the
        # next chunk changes tokenization at the boundary
        from .vibevoice_mlx import TTS_TEXT_WINDOW_SIZE
        while len(tokens) >= TTS_TEXT_WINDOW_SIZE:
            window = tokens[:TTS_TEXT_WINDOW_SIZE]
            tokens = tokens[TTS_TEXT_WINDOW_SIZE:]
            finished = self._run_window(window)
            if finished:
                self._text_buf = ""
                return

        # Decode remaining tokens back to text to preserve the unconsumed tail
        if tokens:
            self._text_buf = self._tokenizer.decode(tokens)
        else:
            self._text_buf = ""

    def flush(self) -> None:
        """Handle the final partial window of text."""
        if not self._text_buf.strip():
            return
        self._ensure_loaded()
        self._init_gen_state()

        # Tokenize remaining text (add newline like VibeVoice expects)
        remaining = self._text_buf.strip() + "\n"
        tokens = self._tokenizer.encode(remaining, add_special_tokens=False)
        self._text_buf = ""

        if not tokens:
            return

        from .vibevoice_mlx import TTS_TEXT_WINDOW_SIZE
        # Process full windows
        while len(tokens) >= TTS_TEXT_WINDOW_SIZE:
            window = tokens[:TTS_TEXT_WINDOW_SIZE]
            tokens = tokens[TTS_TEXT_WINDOW_SIZE:]
            if self._run_window(window):
                break

        # Process final partial window
        if tokens:
            self._run_window(tokens)

        # Let the audio stream drain
        if self._sd_stream is not None:
            import time
            time.sleep(0.5)  # brief pause to let buffer drain

    def close(self) -> None:
        if self._sd_stream is not None:
            try:
                self._sd_stream.stop()
                self._sd_stream.close()
            except Exception:
                pass
            self._sd_stream = None
        self._model = None
        self._voice_prompt = None
        self._gen_state = None
        self._initialized = False
