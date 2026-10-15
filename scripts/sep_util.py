"""Chunked HDemucs (torchaudio HDEMUCS_HIGH_MUSDB_PLUS) vocal separation: the single implementation shared by all scripts.

Chunk handling (corrected version): a chunk starts at `pos`; we separate w[:, max(0, pos-OV) : pos+CH]
(OV seconds of left context), drop the first OV seconds of the output when pos > 0, so every chunk contributes
exactly CH samples (the last chunk takes the remainder). The output has the same length as the input and is
sample-aligned with it (asserted).

An earlier version of the chunking (separating w[:, pos : pos+CH+OV] and then dropping the first OV samples)
placed the output of chunk k at the wrong offset: after the first chunk everything was 2 s late and 2 s were missing
at each 20 s boundary (a 120 s song came out as 118 s). Results in this repository were recomputed with this
corrected implementation.

Other steps: mono mean/std normalisation, mean of the two output channels, multiply back by std, resample to 16 kHz
(librosa or PyAV through audio_io.py), 16-bit wav cache.
"""
import numpy as np

CH_SEC = 20.0    # seconds per chunk
OV_SEC = 2.0     # seconds of left context carried by each chunk


def load_hdemucs(device="cuda"):
    import torchaudio
    b = torchaudio.pipelines.HDEMUCS_HIGH_MUSDB_PLUS
    m = b.get_model().eval().to(device)
    return m, b.sample_rate, m.sources.index("vocals")


def chunked_vocals(sep, w, sr_m, vi, device="cuda", ch_sec=CH_SEC, ov_sec=OV_SEC):
    """w: (2, n) float32 torch tensor, already normalised. Returns (2, n) vocals, sample-aligned with w."""
    import torch
    n = w.shape[1]
    CH, OV = int(ch_sec * sr_m), int(ov_sec * sr_m)
    outs, pos = [], 0
    while pos < n:
        a = max(0, pos - OV); e = min(n, pos + CH)
        with torch.no_grad():
            v = sep(w[:, a:e].unsqueeze(0).to(device))[0][vi].cpu()
        assert v.shape[1] == e - a, (v.shape, e - a)
        v = v[:, pos - a:]                       # for pos > 0 drop the OV context samples
        assert v.shape[1] == e - pos
        outs.append(v); pos = e
    V = torch.cat(outs, 1)
    assert V.shape[1] == n, (V.shape, n)
    return V


def separate_mono(path, sep, sr_m, vi, device="cuda", duration=None):
    """Read the first `duration` seconds of `path` (None = whole file); return mono vocals (numpy) at sr_m, same length as the input."""
    import torch, audio_io
    w = torch.from_numpy(audio_io.load_stereo(str(path), sr_m, duration))
    ref = w.mean(0); w = (w - ref.mean()) / (ref.std() + 1e-8)
    V = chunked_vocals(sep, w, sr_m, vi, device)
    out = (V.mean(0) * (ref.std() + 1e-8)).numpy()
    assert len(out) == w.shape[1]
    return out


def separate_16k(path, sep, sr_m, vi, device="cuda", duration=None):
    """As separate_mono, resampled to 16 kHz float32."""
    import audio_io
    return audio_io.resample(separate_mono(path, sep, sr_m, vi, device, duration), sr_m, 16000).astype(np.float32)


def separate_cached(path, vp, sep, sr_m, vi, device="cuda", duration=None):
    """Read `vp` if it exists; otherwise separate, save as 16 kHz 16-bit mono wav, and read it back
    (so a fresh run and a resumed run feed identical audio downstream)."""
    import audio_io
    from pathlib import Path
    vp = Path(vp)
    if vp.exists():
        return audio_io.load_mono16k(vp)
    v16 = separate_16k(path, sep, sr_m, vi, device, duration)
    vp.parent.mkdir(parents=True, exist_ok=True)
    tmp = vp.with_suffix(".tmp.wav")
    audio_io.write_wav16(tmp, v16, 16000)
    tmp.replace(vp)                              # atomic: an interrupted run leaves no half-written file
    return audio_io.load_mono16k(vp)
