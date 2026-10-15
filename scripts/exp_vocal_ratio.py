#!/usr/bin/env python3
"""Vocal-to-accompaniment level ratio (CPU only): quantifies "the accompaniment masks the voice".

For the 384 ACE-Step generated songs: mixture (16 kHz mono, first 120 s) minus separated vocals
(the 16 kHz copy saved by the vocal-separation experiment) ~= accompaniment.
vr_db = 10*log10(vocal energy / accompaniment energy) over the whole song; vr_active_db is the
same ratio on vocal-active frames only (0.5 s frames whose vocal energy exceeds the song's median).
Inputs:  GEN_ROOT/runs/<exp>/<arm>/<song>@s<seed>.wav, GEN_ROOT/sep_vocals/<exp>/<arm>/<song>@s<seed>.vocals.wav
Outputs: RESULTS_DIR/vocal_ratio48.csv (overwritten).
Supports: the vocal/accompaniment ratio comparison between ACE-Step 2B and XL (mechanism section).
"""
import csv
from pathlib import Path
import numpy as np, librosa

# ===== Edit here =====
REPO = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO / "results"                   # output CSV goes here
GEN_ROOT = REPO / "data" / "audio" / "generated"           # runs/<exp>/<arm>/<song>@s<seed>.wav and sep_vocals/<exp>/<arm>/<song>@s<seed>.vocals.wav
ARMS = ["vad48_2b/B", "vad48_2b/C", "vad48_xl/B", "vad48_xl/C"]   # ACE-Step 2B / XL arms used in the paper
OUT_NAME = "vocal_ratio48.csv"
# =====================


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / OUT_NAME, "w", newline="") as f:
        wr = csv.writer(f); wr.writerow(["exp", "arm", "song", "seed", "vr_db", "vr_active_db", "voc_rms", "acc_rms"])
        for ea in ARMS:
            exp, arm = ea.split("/")
            for w in sorted((GEN_ROOT / "runs" / exp / arm).glob("*.wav")):
                song, seed = w.stem.split("@s")
                vp = GEN_ROOT / "sep_vocals" / exp / arm / f"{w.stem}.vocals.wav"
                if not vp.exists():
                    continue
                m, _ = librosa.load(str(w), sr=16000, mono=True, duration=120.0)
                v, _ = librosa.load(str(vp), sr=16000, mono=True)
                assert abs(len(m) - len(v)) <= 160, (w, len(m), len(v))   # vocals must be aligned and the same length as the mix
                n = min(len(m), len(v)); m, v = m[:n], v[:n]; a = m - v
                eps = 1e-10
                vr = 10 * np.log10((np.mean(v ** 2) + eps) / (np.mean(a ** 2) + eps))
                fr = 8000; k = n // fr
                ve = (v[:k * fr].reshape(k, fr) ** 2).mean(1); ae = (a[:k * fr].reshape(k, fr) ** 2).mean(1)
                act = ve > np.median(ve)
                vra = 10 * np.log10((ve[act].mean() + eps) / (ae[act].mean() + eps))
                wr.writerow([exp, arm, song, seed, round(float(vr), 2), round(float(vra), 2),
                             round(float(np.sqrt(np.mean(v ** 2))), 5), round(float(np.sqrt(np.mean(a ** 2))), 5)])
    print(f"wrote {RESULTS_DIR / OUT_NAME}")


if __name__ == "__main__":
    main()
