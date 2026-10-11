#!/usr/bin/env python3
"""Mechanism analysis: relate the speech time kept by the VAD (exp_vad_stats.py) to the recognition
results (exp_vad_gen.py). Question: are words lost with VAD on because the VAD filters out whole sung
segments, and is XL filtered more than 2B?

Reads:  ../results/vad_stats48.csv, ../results/vad_gen48.csv
Writes: ../results/vad_stats48_summary.json
Supports: mechanism section (kept seconds per system, Spearman correlations between kept time and
recall / word keep ratio, XL vs 2B kept-seconds difference).
"""
import csv, json, statistics as st
from collections import defaultdict
from pathlib import Path

# ===== Edit here =====
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"   # folder holding the result CSV/JSON files
# =====================

R = RESULTS_DIR
SYS = {("vad48_2b", "B"): "2B·B", ("vad48_2b", "C"): "2B·C", ("vad48_xl", "B"): "XL·B", ("vad48_xl", "C"): "XL·C", ("orig", "orig"): "orig"}
S = {(r["exp"], r["arm"], r["song"], r["seed"]): r for r in csv.DictReader(open(R / "vad_stats48.csv"))}
G = defaultdict(dict)
for r in csv.DictReader(open(R / "vad_gen48.csv")):
    G[(r["exp"], r["arm"], r["song"], r["seed"])][r["vad"]] = r


def rank(v):
    o = sorted(range(len(v)), key=lambda i: v[i]); r = [0.0] * len(v); i = 0
    while i < len(o):
        j = i
        while j + 1 < len(o) and v[o[j + 1]] == v[o[i]]:
            j += 1
        for k in range(i, j + 1):
            r[o[k]] = (i + j) / 2.0
        i = j + 1
    return r


def spearman(a, b):
    ra, rb = rank(a), rank(b); ma, mb = st.mean(ra), st.mean(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = (sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb)) ** 0.5
    return round(num / den, 4) if den else float("nan")


out = {"per_system": {}}
allk = [k for k in S if k in G and len(G[k]) == 2]
for name in ("2B·B", "2B·C", "XL·B", "XL·C", "orig"):
    ks = [k for k in allk if SYS[k[:2]] == name]
    kept = [float(S[k]["kept_sec"]) for k in ks]
    out["per_system"][name] = {
        "n": len(ks), "kept_sec_median": round(st.median(kept), 2), "kept_sec_mean": round(st.mean(kept), 2),
        "kept_lt_10s": sum(x < 10 for x in kept), "kept_0s": sum(x == 0 for x in kept),
        "p_gt50_median": round(st.median(float(S[k]["p_gt50"]) for k in ks), 4),
        "recall_off_median": round(st.median(float(G[k]["False"]["recall"]) for k in ks), 4)}
gen = [k for k in allk if k[0] != "orig"]
keep_words = [int(G[k]["True"]["n_hyp"]) / max(1, int(G[k]["False"]["n_hyp"])) for k in gen]
kept_frac = [float(S[k]["kept_sec"]) / 120.0 for k in gen]
out["gen_spearman_keptsec_vs_wordkeep"] = spearman(kept_frac, keep_words)
out["gen_spearman_keptsec_vs_recall_on"] = spearman(kept_frac, [float(G[k]["True"]["recall"]) for k in gen])
out["gen_spearman_keptsec_vs_recall_off"] = spearman(kept_frac, [float(G[k]["False"]["recall"]) for k in gen])
# same song and seed: kept seconds of XL·B minus 2B·B
pair = [(float(S[("vad48_xl", "B", k[2], k[3])]["kept_sec"]) - float(S[k]["kept_sec"])) for k in gen
        if k[:2] == ("vad48_2b", "B") and ("vad48_xl", "B", k[2], k[3]) in S]
out["xlB_minus_2bB_kept_sec"] = {"pairs": len(pair), "mean": round(st.mean(pair), 2), "median": round(st.median(pair), 2),
                                 "xl_less": sum(x < 0 for x in pair), "xl_more": sum(x > 0 for x in pair)}
(R / "vad_stats48_summary.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
print("system    n  kept sec median/mean (of 120 s)  <10 s  =0 s  median share of windows with speech prob >0.5  median recall VAD off")
for n, v in out["per_system"].items():
    print(f"{n:5s} {v['n']:3d}  {v['kept_sec_median']:6.1f} / {v['kept_sec_mean']:6.1f}   {v['kept_lt_10s']:3d}  {v['kept_0s']:3d}   {v['p_gt50_median']:.3f}   {v['recall_off_median']:.3f}")
print("generated songs (384 items) Spearman: kept-time share vs word keep ratio", out["gen_spearman_keptsec_vs_wordkeep"],
      "| vs recall VAD on", out["gen_spearman_keptsec_vs_recall_on"], "| vs recall VAD off", out["gen_spearman_keptsec_vs_recall_off"])
print("XL·B - 2B·B kept seconds (same song and seed)", out["xlB_minus_2bB_kept_sec"])
