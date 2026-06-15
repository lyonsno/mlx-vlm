"""Tests for gemma4_realtime_audio.py — F4 from finding review.

Fast, deterministic, no model loads, no GPU. Exercises:
  1. build_prompt roundtrip (audio token injection)
  2. load_audio_file shape/dtype contract
  3. embed_audio guard (sys.exit on missing attribute)
"""

import importlib
import os
import sys
import types
from unittest import mock

import numpy as np
import pytest

# The script lives at the repo root, not inside a package.
# Import it as a module from the filesystem.
_SCRIPT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "gemma4_realtime_audio.py",
)
_spec = importlib.util.spec_from_file_location("gemma4_realtime_audio", _SCRIPT_PATH)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

build_prompt = _mod.build_prompt
load_audio_file = _mod.load_audio_file
main = _mod.main
SAMPLE_RATE = _mod.SAMPLE_RATE


# ---------------------------------------------------------------------------
# Test 1: build_prompt roundtrip
# ---------------------------------------------------------------------------
class TestBuildPrompt:
    """Verify build_prompt injects the audio placeholder and user text."""

    def test_roundtrip_contains_audio_token_and_user_text(self):
        """Mock apply_chat_template to return its content arg unchanged,
        then verify the audio placeholder and user text are both present."""

        # Processor stub with audio_token attribute (mimics Gemma4Processor)
        processor = types.SimpleNamespace(audio_token="<|audio|>")
        config = types.SimpleNamespace()

        # Stub apply_chat_template: just returns the content string it receives
        with mock.patch(
            "mlx_vlm.prompt_utils.apply_chat_template",
            side_effect=lambda proc, cfg, content, **kw: content,
        ):
            result = build_prompt(processor, config, "hello")

        assert "<|audio|>" in result, "audio placeholder token missing from prompt"
        assert "hello" in result, "user text missing from prompt"

    def test_fallback_audio_token_when_attribute_missing(self):
        """When processor lacks audio_token attr, fallback <|audio|> is used."""
        processor = types.SimpleNamespace()  # no audio_token attribute
        config = types.SimpleNamespace()

        with mock.patch(
            "mlx_vlm.prompt_utils.apply_chat_template",
            side_effect=lambda proc, cfg, content, **kw: content,
        ):
            result = build_prompt(processor, config, "test")

        assert "<|audio|>" in result, "fallback audio token not injected"


# ---------------------------------------------------------------------------
# Test 2: load_audio_file shape/dtype
# ---------------------------------------------------------------------------
class TestLoadAudioFile:
    """Verify load_audio_file returns correct shape and dtype for a 1s WAV."""

    def test_shape_and_dtype_from_wav(self, tmp_path):
        """Create a 1-second 16kHz sine WAV, load via load_audio_file,
        and assert shape=(16000,) dtype=float32."""
        import scipy.io.wavfile

        # Generate 1s of 440Hz sine at 16kHz
        sr = 16_000
        t = np.linspace(0, 1, sr, endpoint=False, dtype=np.float32)
        tone = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

        wav_path = str(tmp_path / "test_tone.wav")
        scipy.io.wavfile.write(wav_path, sr, tone)

        # Mock load_audio to do a simple scipy read + float32 conversion,
        # avoiding mlx_audio dependency.
        def _mock_load_audio(path, sr=16_000):
            rate, data = scipy.io.wavfile.read(path)
            data = data.astype(np.float32)
            # Normalize int16 if needed
            if data.dtype == np.int16 or np.max(np.abs(data)) > 1.0:
                data = data / 32768.0
            return data

        with mock.patch("mlx_vlm.utils.load_audio", side_effect=_mock_load_audio):
            audio = load_audio_file(wav_path, sample_rate=sr)

        assert audio.shape == (sr,), f"expected shape ({sr},), got {audio.shape}"
        assert audio.dtype == np.float32, f"expected float32, got {audio.dtype}"


# ---------------------------------------------------------------------------
# Test 3: embed_audio guard
# ---------------------------------------------------------------------------
class TestEmbedAudioGuard:
    """Verify main() exits with code 1 and prints [ERROR] when the loaded
    model lacks embed_audio."""

    def test_exits_on_missing_embed_audio(self, capsys):
        """Patch mlx_vlm.load to return a model without embed_audio;
        assert sys.exit(1) and [ERROR] in output."""

        # Model stub: has model_type but no embed_audio
        fake_model = types.SimpleNamespace(model_type="gemma4_unified")
        # Processor stub
        fake_processor = types.SimpleNamespace(
            audio_token="<|audio|>",
            tokenizer=types.SimpleNamespace(),
        )

        with (
            mock.patch(
                "sys.argv",
                ["gemma4_realtime_audio.py", "--audio-file", "dummy.wav"],
            ),
            mock.patch(
                "mlx_vlm.load",
                return_value=(fake_model, fake_processor),
            ),
            mock.patch(
                "mlx_vlm.generate.stream_generate",
                return_value=iter([]),
            ),
            pytest.raises(SystemExit) as exc_info,
        ):
            main()

        assert exc_info.value.code == 1, f"expected exit(1), got exit({exc_info.value.code})"
        captured = capsys.readouterr()
        assert "[ERROR]" in captured.out, "expected [ERROR] message on missing embed_audio"
