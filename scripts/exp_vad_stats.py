#!/usr/bin/env python3
"""Mechanism analysis: how much audio the speech VAD keeps per song (CPU only, no ASR).

Uses faster-whisper's bundled Silero VAD (the same model and default parameters that
transcribe(vad_filter=True) uses). Per song (first 120 s):
  kept_sec   total seconds kept as speech (including the default 400 ms padding on each side)
  n_chunks   number of kept chunks
  p_mean     mean per-window (32 ms) speech probability; p_gt50 = fraction of windows with p > 0.5
For songs whose separated vocals exist (GEN_ROOT/sep_vocals/<exp>/<arm>/<song>@s<seed>.vocals.wav,
written by the vocal-separation experiment), the same stats on the vocals are written with prefix voc_.
Inputs:  GEN_ROOT/runs/<exp>/<arm>/<song>@s<seed>.wav, GEN_ROOT/sep_vocals/..., ORIG_DIR/<song>.flac.
Outputs: RESULTS_DIR/vad_stats48.csv, RESULTS_DIR/vad_stats48_heart_levo.csv (resumable).
Supports: the "VAD discards most of the song" mechanism numbers (kept seconds per family).
"""
import csv
from pathlib import Path
import numpy as np, librosa
from faster_whisper.vad import VadOptions, get_speech_timestamps, get_vad_model

# ===== Edit here =====
REPO = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO / "results"                   # output CSVs go here
GEN_ROOT = REPO / "data" / "audio" / "generated"           # runs/<exp>/<arm>/<song>@s<seed>.wav (120 s crops you generate) and sep_vocals/<exp>/<arm>/<song>@s<seed>.vocals.wav
ORIG_DIR = Path("/path/to/original_recordings")  # <song>.flac commercial originals (not redistributable)
# (output csv, list of "<exp>/<arm>", include originals?) - configurations used for the paper
RUNS = [
    ("vad_stats48.csv", ["vad48_2b/B", "vad48_2b/C", "vad48_xl/B", "vad48_xl/C"], True),   # ACE-Step + 48 originals
    ("vad_stats48_heart_levo.csv", ["vad48_heart/H", "vad48_levo/L"], False),             # HeartMuLa, LeVo (same originals, not repeated)
]
# =====================

FIELDS = ["src", "exp", "arm", "song", "seed", "dur", "kept_sec", "n_chunks", "p_mean", "p_gt50",
          "voc_kept_sec", "voc_n_chunks", "voc_p_mean", "voc_p_gt50"]


def stats(a, model):
    ts = get_speech_timestamps(a, VadOptions())
    kept = sum(t["end"] - t["start"] for t in ts) / 16000.0
    pad = (-len(a)) % 512
    p = np.asarray(model(np.pad(a, (0, pad)))).reshape(-1)
    return round(kept, 2), len(ts), round(float(p.mean()), 4), round(float((p > 0.5).mean()), 4)


def run_one(model, out_name, arms, with_orig):
    todo, songs = [], []
    for ea in arms:
        exp, arm = ea.split("/")
        for w in sorted((GEN_ROOT / "runs" / exp / arm).glob("*.wav")):
            song, seed = w.stem.split("@s")
            todo.append(("gen", exp, arm, song, seed, w))
            if song not in songs:
                songs.append(song)
    todo += [("orig", "orig", "orig", s, "0", ORIG_DIR / f"{s}.flac") for s in songs] if with_orig else []
    out_csv = RESULTS_DIR / out_name
    done = set()
    if out_csv.exists():
        done = {(r["exp"], r["arm"], r["song"], r["seed"]) for r in csv.DictReader(open(out_csv))}
    new = not out_csv.exists()
    with open(out_csv, "a", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            wr.writeheader()
        for k, (src, exp, arm, s, seed, path) in enumerate(todo, 1):
            if (exp, arm, s, seed) in done:
                continue
            a, _ = librosa.load(str(path), sr=16000, mono=True, duration=120.0)
            a = np.asarray(a, dtype=np.float32)
            row = dict(zip(["kept_sec", "n_chunks", "p_mean", "p_gt50"], stats(a, model)))
            vp = GEN_ROOT / "sep_vocals" / exp / arm / f"{s}@s{seed}.vocals.wav"
            if vp.exists():
                v, _ = librosa.load(str(vp), sr=16000, mono=True)
                row.update(zip(["voc_kept_sec", "voc_n_chunks", "voc_p_mean", "voc_p_gt50"],
                               stats(np.asarray(v, dtype=np.float32), model)))
            wr.writerow({"src": src, "exp": exp, "arm": arm, "song": s, "seed": seed,
                         "dur": round(len(a) / 16000.0, 1), **row})
            f.flush()
            if k % 20 == 0 or k == len(todo):
                print(f"[{k}/{len(todo)}]", flush=True)
    print(f"wrote {out_csv}")


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    model = get_vad_model()
    for out_name, arms, with_orig in RUNS:
        run_one(model, out_name, arms, with_orig)


if __name__ == "__main__":
    main()
