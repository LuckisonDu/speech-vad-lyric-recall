"""Audio I/O helpers with two decoding backends.

Used by exp_vad_bench.py and exp_batched.py (and by other scripts in this folder).
No inputs/outputs of its own. Default backend is librosa; the "av" backend uses the
PyAV decoder bundled with faster-whisper (for environments where librosa is unavailable).
Note: the two backends use different resamplers, so the same file does not give
bit-identical samples; record which backend produced a result file.
"""
import os
import numpy as np

# ===== Edit here =====
BACKEND = os.environ.get("AUDIO_BACKEND", "librosa")   # "librosa" (default) or "av" (faster-whisper's PyAV decoder); scripts may also set audio_io.BACKEND directly
# =====================


def load_mono16k(path, duration=None):
    if BACKEND == "av":
        from faster_whisper.audio import decode_audio
        a = decode_audio(str(path), sampling_rate=16000)
        if duration:
            a = a[: int(duration * 16000)]
        return np.asarray(a, dtype=np.float32)
    import librosa
    a, _ = librosa.load(str(path), sr=16000, mono=True, duration=duration)
    return np.asarray(a, dtype=np.float32)


def load_stereo(path, sr, duration=None):
    """Return a (2, n) float32 array."""
    if BACKEND == "av":
        from faster_whisper.audio import decode_audio
        l, r = decode_audio(str(path), sampling_rate=sr, split_stereo=True)
        a = np.stack([l, r]).astype(np.float32)
        return a[:, : int(duration * sr)] if duration else a
    import librosa
    a, _ = librosa.load(str(path), sr=sr, mono=False, duration=duration)
    a = np.atleast_2d(a)
    if a.shape[0] == 1:
        a = np.repeat(a, 2, axis=0)
    return np.ascontiguousarray(a, dtype=np.float32)


def resample(x, orig_sr, target_sr):
    if BACKEND == "av":
        import torch, torchaudio
        return torchaudio.functional.resample(torch.from_numpy(np.asarray(x, dtype=np.float32)), orig_sr, target_sr).numpy()
    import librosa
    return librosa.resample(np.asarray(x, dtype=np.float32), orig_sr=orig_sr, target_sr=target_sr).astype(np.float32)


def write_wav16(path, x, sr):
    """Mono 16-bit wav (standard library only, no soundfile dependency)."""
    import wave
    y = (np.clip(np.asarray(x, dtype=np.float32), -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr); w.writeframes(y.tobytes())
