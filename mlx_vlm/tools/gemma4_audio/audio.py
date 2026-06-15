import numpy as np

from .constants import SAMPLE_RATE


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
            sd.stop()

    if not frames:
        print("[mic] WARNING: no audio captured, check mic permissions")
    audio = np.concatenate(frames) if frames else np.zeros(sample_rate, dtype=np.float32)
    print(f"[mic] {len(audio)/sample_rate:.2f}s captured")
    return audio


def load_audio_file(path: str, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Load audio file, resample to target rate, return float32 mono waveform."""
    from mlx_vlm.utils import load_audio
    audio = load_audio(path, sr=sample_rate)
    print(f"[file] {len(audio)/sample_rate:.2f}s from {path}")
    return audio


def process_gradio_audio(audio_tuple) -> np.ndarray:
    """Convert Gradio mic output (sr, ndarray) to float32 mono 16kHz ndarray."""
    import scipy.signal

    sr, data = audio_tuple
    if data.ndim > 1:
        data = data[:, 0]
    data = data.astype(np.float32)
    if sr != SAMPLE_RATE:
        n_samples = int(len(data) * SAMPLE_RATE / sr)
        data = scipy.signal.resample(data, n_samples)
    peak = np.max(np.abs(data))
    if peak > 1e-6:
        data = data / peak
    return data
