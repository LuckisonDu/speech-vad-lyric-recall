#!/usr/bin/env python3
"""Third speech VAD: how much of each mixture WebRTC VAD (mode 2) keeps, over the 7 systems x 672 songs.
Only measures kept duration; no transcription.

Reads  : generated audio under AUDIO_ROOT (first 120 s of each file is used); the system list ARMS
         comes from exp_energy_gate.py, the VAD segmentation from exp_webrtc_gate.webrtc_segments.
Writes : ../results/webrtc_kept48.csv (exp, arm, file, kept_sec, dur) and prints per-system
         median / minimum kept fraction.
Supports the paper's WebRTC kept-fraction numbers (paper_numbers.py reads webrtc_kept48.csv).
"""
import csv, glob, numpy as np, audio_io
from collections import defaultdict
from pathlib import Path
from exp_webrtc_gate import webrtc_segments
from exp_energy_gate import ARMS

# ===== Edit here =====
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"   # output folder
AUDIO_ROOT = Path("data/audio/generated")   # contains <root>/<exp>/<arm>/<song>@s<seed>.wav (generated songs you produce yourself; <root> as in ARMS)
DURATION_SEC = 120.0                        # analyse the first 120 s of each song
SR = 16000                                  # sample rate of the loaded audio
# =====================

rows = []
for root, ea in ARMS:
    exp, arm = ea.split("/")
    for w in sorted(glob.glob(str(AUDIO_ROOT / root / exp / arm / "*.wav"))):
        a = np.asarray(audio_io.load_mono16k(w, duration=DURATION_SEC), dtype=np.float32)
        s = webrtc_segments(a)
        rows.append({"exp": exp, "arm": arm, "file": w.split("/")[-1], "kept_sec": round(sum(e - b for b, e in s), 2), "dur": round(len(a) / SR, 2)})
with open(RESULTS_DIR / "webrtc_kept48.csv", "w", newline="") as f:
    wr = csv.DictWriter(f, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
g = defaultdict(list)
for r in rows: g[r["exp"] + "/" + r["arm"]].append(r["kept_sec"] / r["dur"])
for k, v in g.items(): print(k, len(v), round(float(np.median(v)), 3), round(float(np.min(v)), 3))
