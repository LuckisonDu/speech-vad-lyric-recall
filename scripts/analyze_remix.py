#!/usr/bin/env python3
"""Summarize the controlled remix experiment (separated vocals + g x residual accompaniment).

Reads:  RESULTS_DIR/remix48.csv (from exp_remix.py; Silero VAD retained seconds per song at each gain g).
Writes: RESULTS_DIR/remix48_summary.json
  median_kept[system][g]      median retained seconds per ACE-Step configuration
  xlB_minus_2bB_mean[g]       mean paired difference XL - 2B (no planner), paired by song and seed
Supports: the remix paragraph of the mechanism section (macros \\NRemix...).
"""
import csv, json, statistics as st
from pathlib import Path

# ===== Edit here =====
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"   # folder with remix48.csv
GAINS = ["0.0", "0.25", "0.5", "1.0", "2.0"]                        # residual gains used in exp_remix.py
SYSTEMS = {("vad48_2b", "B"): "2B·B", ("vad48_2b", "C"): "2B·C",   # (exp, arm) -> system label
           ("vad48_xl", "B"): "XL·B", ("vad48_xl", "C"): "XL·C"}
# =====================

rows = list(csv.DictReader(open(RESULTS_DIR / "remix48.csv")))
kept = {s: {g: [] for g in GAINS} for s in SYSTEMS.values()}
by = {}
for r in rows:
    s = SYSTEMS[(r["exp"], r["arm"])]
    for g in GAINS:
        kept[s][g].append(float(r[f"kept_g{g}"]))
    by[(s, r["song"], r["seed"])] = r
out = {"median_kept": {s: {g: round(st.median(v), 1) for g, v in d.items()} for s, d in kept.items()},
       "xlB_minus_2bB_mean": {}}
for g in GAINS:
    diffs = [float(by[("XL·B", so, se)][f"kept_g{g}"]) - float(r[f"kept_g{g}"])
             for (s, so, se), r in by.items() if s == "2B·B" and ("XL·B", so, se) in by]
    out["xlB_minus_2bB_mean"][g] = round(st.mean(diffs), 1)
(RESULTS_DIR / "remix48_summary.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
print(json.dumps(out, ensure_ascii=False))
