#!/usr/bin/env python3
"""Controlled remix (causal test, CPU only): keep the vocals fixed, change only the accompaniment level,
and measure how many seconds Silero VAD keeps.

vocals = separated vocals (16 kHz copy saved by the vocal-separation experiment); accompaniment = mixture - vocals.
Gain g in {0 (vocals only), 0.25, 0.5, 1 (~ original mixture), 2}; remix = vocals + g * accompaniment (clipped to [-1, 1]).
Note: because accompaniment = mixture - vocals, g = 1 equals the original mixture up to float rounding and clipping
(376 of 384 songs keep exactly the same seconds; the rest differ by at most 2.0 s, songs at the threshold edge).
Inputs:  GEN_ROOT/runs/<exp>/<arm>/<song>@s<seed>.wav, GEN_ROOT/sep_vocals/<exp>/<arm>/<song>@s<seed>.vocals.wav
Outputs: RESULTS_DIR/remix48.csv (overwritten).
Supports: the accompaniment-gain remix experiment (mechanism section).
"""
import csv
from pathlib import Path
import numpy as np, librosa
from faster_whisper.vad import VadOptions, get_speech_timestamps

# ===== Edit here =====
REPO = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO / "results"                   # output CSV goes here
GEN_ROOT = REPO / "data" / "audio" / "generated"           # runs/<exp>/<arm>/<song>@s<seed>.wav and sep_vocals/<exp>/<arm>/<song>@s<seed>.vocals.wav
ARMS = ["vad48_2b/B", "vad48_2b/C", "vad48_xl/B", "vad48_xl/C"]   # ACE-Step 2B / XL arms used in the paper
GAINS = [0.0, 0.25, 0.5, 1.0, 2.0]               # accompaniment gains
OUT_NAME = "remix48.csv"
# =====================


def kept(a):
    return round(sum(t["end"] - t["start"] for t in get_speech_timestamps(a, VadOptions())) / 16000.0, 2)


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / OUT_NAME, "w", newline="") as f:
        wr = csv.writer(f); wr.writerow(["exp", "arm", "song", "seed"] + [f"kept_g{g}" for g in GAINS])
        for ea in ARMS:
            exp, arm = ea.split("/")
            for w in sorted((GEN_ROOT / "runs" / exp / arm).glob("*.wav")):
                song, seed = w.stem.split("@s")
                vp = GEN_ROOT / "sep_vocals" / exp / arm / f"{w.stem}.vocals.wav"
                if not vp.exists():
                    continue
                m, _ = librosa.load(str(w), sr=16000, mono=True, duration=120.0)
                v, _ = librosa.load(str(vp), sr=16000, mono=True)
                n = min(len(m), len(v)); m, v = m[:n].astype(np.float32), v[:n].astype(np.float32); a = m - v
                wr.writerow([exp, arm, song, seed] + [kept(np.clip(v + g * a, -1, 1).astype(np.float32)) for g in GAINS])
                f.flush()
    print(f"wrote {RESULTS_DIR / OUT_NAME}")


if __name__ == "__main__":
    main()
