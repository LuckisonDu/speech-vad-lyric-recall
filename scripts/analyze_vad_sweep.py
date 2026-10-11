#!/usr/bin/env python3
"""VAD threshold sweep and VAD on separated vocals (exp_vad_sweep.py).
Answers: (1) how much singing is kept when the threshold is lowered from the default 0.5;
(2) with the accompaniment removed, does the VAD recognise the vocals as speech (does the XL vs 2B gap remain).

Reads:  ../results/vad_sweep48.csv
Writes: ../results/vad_sweep48_summary.json
Supports: mechanism section (median kept seconds at thresholds 0.1/0.3/0.5, mix vs separated vocals).
"""
import csv, json, statistics as st
from pathlib import Path

# ===== Edit here =====
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"   # folder holding the result CSV/JSON files
THR = ["0.1", "0.2", "0.3", "0.4", "0.5"]                          # VAD thresholds swept (column suffixes in the CSV)
# =====================

R = RESULTS_DIR
SYS = {("vad48_2b", "B"): "2B·B", ("vad48_2b", "C"): "2B·C", ("vad48_xl", "B"): "XL·B", ("vad48_xl", "C"): "XL·C", ("orig", "orig"): "orig"}
rows = list(csv.DictReader(open(R / "vad_sweep48.csv")))
out = {}
print("median kept seconds (of 120 s)   mix: thr 0.1 / 0.2 / 0.3 / 0.4 / 0.5   |  separated vocals: 0.1 / 0.2 / 0.3 / 0.4 / 0.5   | items still <30 s on mix @0.1")
for name in ("2B·B", "2B·C", "XL·B", "XL·C", "orig"):
    rs = [r for r in rows if SYS[(r["exp"], r["arm"])] == name]
    mix = {t: round(st.median(float(r[f"kept_{t}"]) for r in rs), 1) for t in THR}
    voc = {t: (round(st.median(float(r[f"voc_kept_{t}"]) for r in rs if r[f"voc_kept_{t}"]), 1) if any(r[f"voc_kept_{t}"] for r in rs) else None) for t in THR}
    lt30 = sum(float(r["kept_0.1"]) < 30 for r in rs)
    out[name] = {"n": len(rs), "mix_kept_median": mix, "vocals_kept_median": voc, "mix_thr0.1_lt30s": lt30}
    print(f"{name:5s} n={len(rs):3d}   " + " / ".join(f"{mix[t]:5.1f}" for t in THR) + "   |  " +
          " / ".join("  -  " if voc[t] is None else f"{voc[t]:5.1f}" for t in THR) + f"   | {lt30}")
(R / "vad_sweep48_summary.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
