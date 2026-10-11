#!/usr/bin/env python3
"""Evaluation "multiverse": the same 384 ACE-Step songs, four systems compared under 20 reasonable
evaluation settings, to see how stable the ranking is.

Settings = 10 recognition pipelines x 2 metrics (recall, higher is better; WER, lower is better).
Nothing is re-transcribed; everything is read from existing results:
  no VAD (6):   faster-whisper VAD off (P1); P1 + re-decode on separated vocals when < 0.5 words/s (P2);
                separate vocals for all songs then decode (VAD off); openai-whisper default;
                transformers pipeline (30 s chunks); Parakeet-TDT (non-Whisper)
  with VAD (4): faster-whisper VAD on (Silero threshold 0.5 / 0.3 / 0.1); WhisperX default (pyannote)
(PER is only available for the faster-whisper on/off pipelines, so it is left out to keep settings balanced.)
Per setting x per system pair (6 pairs): pair by song and seed, average the two seeds per song,
song-level bootstrap 95% CI (N=20000, seed=101) -> "A significantly better / B significantly better / n.s.".
Also reports the full four-system ranking under each setting.
Exploratory summary of results reported elsewhere in the paper, not a preregistered test.

Reads  : ../results/vad_gen48.csv, sep_redecode48.csv, tool_openai.csv, tool_hf.csv, tool_parakeet.csv,
         tool_fw_thr_0.3.csv, tool_fw_thr_0.1.csv, tool_whisperx.csv
Writes : ../results/multiverse_summary.json (includes per-cell results, used by make_figs.py fig5)
"""
import csv, json, random
from collections import defaultdict
from itertools import combinations
from pathlib import Path
import numpy as np

# ===== Edit here =====
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"   # folder holding the result CSV/JSON files
N_BOOT = 20000      # bootstrap resamples
BOOT_SEED = 101     # bootstrap RNG seed
P2_RATE = 0.5       # P2 rule: words per second below which the vocals re-decode is used
CROP_SEC = 120      # length of the analysed crop in seconds
# =====================

R = RESULTS_DIR
SYS = {("vad48_2b", "B"): "2B", ("vad48_2b", "C"): "2B+pl", ("vad48_xl", "B"): "XL", ("vad48_xl", "C"): "XL+pl"}
NAMES = ["2B", "2B+pl", "XL", "XL+pl"]
data = defaultdict(dict)        # pipeline -> (system, song, seed) -> {recall, wer}


def put(pipe, r, rec, w):
    data[pipe][(SYS[(r["exp"], r["arm"])], r["song"], r["seed"])] = {"recall": float(rec), "wer": float(w)}


off = {}
for r in csv.DictReader(open(R / "vad_gen48.csv")):
    if r["src"] != "gen":
        continue
    put("fw VAD off" if r["vad"] == "False" else "fw VAD 0.5", r, r["recall"], r["wer"])
    if r["vad"] == "False":
        off[(r["exp"], r["arm"], r["song"], r["seed"])] = r
for r in csv.DictReader(open(R / "sep_redecode48.csv")):
    put("all vocals", r, r["recall"], r["wer"])
    o = off[(r["exp"], r["arm"], r["song"], r["seed"])]
    use = int(o["n_hyp"]) / CROP_SEC < P2_RATE and int(r["n_hyp"]) / CROP_SEC >= P2_RATE      # P2 rule
    put("fw off + check", r, (r if use else o)["recall"], (r if use else o)["wer"])
for fn, pipe in (("tool_openai.csv", "openai-whisper"), ("tool_hf.csv", "transformers"), ("tool_parakeet.csv", "Parakeet"),
                 ("tool_fw_thr_0.3.csv", "fw VAD 0.3"), ("tool_fw_thr_0.1.csv", "fw VAD 0.1"), ("tool_whisperx.csv", "WhisperX")):
    for r in csv.DictReader(open(R / fn)):
        put(pipe, r, r["recall"], r["wer"])

NOVAD = ["fw VAD off", "fw off + check", "all vocals", "openai-whisper", "transformers", "Parakeet"]
VAD = ["fw VAD 0.5", "fw VAD 0.3", "fw VAD 0.1", "WhisperX"]
PIPES = NOVAD + VAD
for p in PIPES:
    assert len(data[p]) == 384, (p, len(data[p]))
songs = sorted({k[1] for k in data["fw VAD off"]})
rng = np.random.default_rng(BOOT_SEED)
IDX = rng.integers(0, len(songs), size=(N_BOOT, len(songs)))   # all cells share one set of resamples (paired, comparable)


def score(pipe, metric, sysn, song):
    v = [data[pipe][(sysn, song, sd)][metric] for sd in ("1", "2")]
    x = sum(v) / 2
    return x if metric == "recall" else -x                        # negate WER so that larger is better


cells, rankings = [], {}
for metric in ("recall", "wer"):
    for pipe in PIPES:
        means = {s: float(np.mean([score(pipe, metric, s, g) for g in songs])) for s in NAMES}
        rankings[f"{pipe} | {metric}"] = sorted(NAMES, key=lambda s: -means[s])
        for a, b in combinations(NAMES, 2):
            d = np.array([score(pipe, metric, a, g) - score(pipe, metric, b, g) for g in songs])
            bs = np.sort(d[IDX].mean(axis=1))
            lo, hi = float(bs[500]), float(bs[19499])
            verdict = a if lo > 0 else (b if hi < 0 else "n.s.")
            cells.append({"pipe": pipe, "vad": pipe in VAD, "metric": metric, "pair": f"{a} vs {b}", "a": a, "b": b,
                          "diff": round(float(d.mean()), 4), "ci95": [round(lo, 4), round(hi, 4)], "verdict": verdict})

pairs = {}
for a, b in combinations(NAMES, 2):
    cs = [c for c in cells if c["a"] == a and c["b"] == b]
    def cnt(sub):
        return {a: sum(c["verdict"] == a for c in sub), b: sum(c["verdict"] == b for c in sub), "n.s.": sum(c["verdict"] == "n.s." for c in sub), "n": len(sub)}
    pairs[f"{a} vs {b}"] = {"all": cnt(cs), "no_vad": cnt([c for c in cs if not c["vad"]]), "vad": cnt([c for c in cs if c["vad"]]),
                           "both_directions_significant": any(c["verdict"] == a for c in cs) and any(c["verdict"] == b for c in cs),
                           "both_directions_significant_no_vad": any(c["verdict"] == a for c in cs if not c["vad"]) and any(c["verdict"] == b for c in cs if not c["vad"])}
uniq = defaultdict(list)
for k, v in rankings.items():
    uniq[" > ".join(v)].append(k)
top = defaultdict(int)
for v in rankings.values():
    top[v[0]] += 1
out = {"n_universes": len(rankings), "pipelines_no_vad": NOVAD, "pipelines_vad": VAD, "metrics": ["recall", "wer"],
       "pairs": pairs, "distinct_rankings": len(uniq), "rankings": {k: v for k, v in uniq.items()},
       "distinct_rankings_no_vad": len({" > ".join(v) for k, v in rankings.items() if k.split(" | ")[0] in NOVAD}),
       "distinct_rankings_vad": len({" > ".join(v) for k, v in rankings.items() if k.split(" | ")[0] in VAD}),
       "top_system_counts": dict(top),
       "pairs_reversed_significantly": sum(v["both_directions_significant"] for v in pairs.values()),
       "pairs_reversed_significantly_no_vad": sum(v["both_directions_significant_no_vad"] for v in pairs.values()),
       "cells": cells}
(R / "multiverse_summary.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
print(json.dumps({k: v for k, v in out.items() if k not in ("cells",)}, indent=1, ensure_ascii=False))
