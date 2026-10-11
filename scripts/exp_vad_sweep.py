#!/usr/bin/env python3
"""Mechanism analysis (CPU only): VAD threshold sweep and VAD on separated vocals.

(1) Silero VAD threshold swept over 0.1..0.5: seconds kept at each threshold;
(2) the same sweep on the separated vocals (voc_ columns; empty for originals, which have no separated vocals).
Question: can a lower threshold recover the singing, and is the voice itself non-speech-like
or masked by the accompaniment? VAD = faster-whisper's bundled Silero, other parameters default.
Inputs:  GEN_ROOT/runs/<exp>/<arm>/<song>@s<seed>.wav, GEN_ROOT/sep_vocals/..., ORIG_DIR/<song>.flac.
Outputs: RESULTS_DIR/vad_sweep48.csv, RESULTS_DIR/vad_sweep48_heart_levo.csv (resumable).
Supports: the threshold-sweep and separated-vocals mechanism numbers.
"""
import csv
from pathlib import Path
import numpy as np, librosa
from faster_whisper.vad import VadOptions, get_speech_timestamps

# ===== Edit here =====
REPO = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO / "results"                   # output CSVs go here
GEN_ROOT = REPO / "data" / "audio" / "generated"           # runs/<exp>/<arm>/<song>@s<seed>.wav and sep_vocals/<exp>/<arm>/<song>@s<seed>.vocals.wav
ORIG_DIR = Path("/path/to/original_recordings")  # <song>.flac commercial originals (not redistributable); missing files are skipped
THR = [0.1, 0.2, 0.3, 0.4, 0.5]                  # VAD thresholds swept
# (output csv, list of "<exp>/<arm>", include originals?) - configurations used for the paper
RUNS = [
    ("vad_sweep48.csv", ["vad48_2b/B", "vad48_2b/C", "vad48_xl/B", "vad48_xl/C"], True),
    ("vad_sweep48_heart_levo.csv", ["vad48_heart/H", "vad48_levo/L"], False),
]
# =====================

FIELDS = ["src", "exp", "arm", "song", "seed"] + [f"kept_{t}" for t in THR] + [f"voc_kept_{t}" for t in THR]


def kept(a, t):
    return round(sum(x["end"] - x["start"] for x in get_speech_timestamps(a, VadOptions(threshold=t))) / 16000.0, 2)


def run_one(out_name, arms, with_orig):
    todo, songs = [], []
    for ea in arms:
        exp, arm = ea.split("/")
        for w in sorted((GEN_ROOT / "runs" / exp / arm).glob("*.wav")):
            song, seed = w.stem.split("@s")
            todo.append(("gen", exp, arm, song, seed, w))
            if song not in songs:
                songs.append(song)
    todo += [("orig", "orig", "orig", s, "0", ORIG_DIR / f"{s}.flac") for s in songs
             if (ORIG_DIR / f"{s}.flac").exists()] if with_orig else []
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
            row = {"src": src, "exp": exp, "arm": arm, "song": s, "seed": seed, **{f"kept_{t}": kept(a, t) for t in THR}}
            vp = GEN_ROOT / "sep_vocals" / exp / arm / f"{s}@s{seed}.vocals.wav"
            if vp.exists():
                v, _ = librosa.load(str(vp), sr=16000, mono=True)
                v = np.asarray(v, dtype=np.float32)
                row.update({f"voc_kept_{t}": kept(v, t) for t in THR})
            wr.writerow(row); f.flush()
            if k % 40 == 0 or k == len(todo):
                print(f"[{k}/{len(todo)}]", flush=True)
    print(f"wrote {out_csv}")


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for out_name, arms, with_orig in RUNS:
        run_one(out_name, arms, with_orig)


if __name__ == "__main__":
    main()
