#!/usr/bin/env python3
"""Effect of the three pipelines P0 / P1 / P2 (pre-registered definitions).
P0 = VAD on; P1 = VAD off; P2 = VAD off, but if fewer than 0.5 recognised words per second, use the
VAD-off recognition on separated vocals instead (only if that one has >= 0.5 words/s).
Also reports, for items that were not flagged, whether recognising on vocals would be worse
(i.e. "could one simply always recognise on separated vocals").

Reads:  ../results/vad_gen48.csv, ../results/sep_redecode48.csv
Writes: ../results/p2_summary.json
Supports: corrected-pipeline section (P0/P1/P2 means, items below 0.3, flagged/rescued counts,
all-vocals mean, unflagged vocals-minus-mix difference).
"""
import csv, json, statistics as st
from collections import defaultdict
from pathlib import Path

# ===== Edit here =====
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"   # folder holding the result CSV/JSON files
WPS, WIN = 0.5, 120.0   # flag threshold in words per second; window length in seconds
ACE_SEEDS = ("1", "2")  # vad_gen48.csv has ACE-Step seeds 1-6, but sep_redecode48.csv (re-recognition on separated vocals, needed for P2) only has seeds 1 and 2;
                        # P0 / P1 / P2 must be compared on the same songs, so only seeds 1 and 2 are used
# =====================

R = RESULTS_DIR
SYS = {("vad48_2b", "B"): "2B·B", ("vad48_2b", "C"): "2B·C", ("vad48_xl", "B"): "XL·B", ("vad48_xl", "C"): "XL·C"}
G = defaultdict(dict)
for r in csv.DictReader(open(R / "vad_gen48.csv")):
    if r["src"] == "gen" and r["seed"] in ACE_SEEDS:
        G[(r["exp"], r["arm"], r["song"], r["seed"])][r["vad"]] = r
V = {(r["exp"], r["arm"], r["song"], r["seed"]): r for r in csv.DictReader(open(R / "sep_redecode48.csv"))}


def wps(r):
    return int(r["n_hyp"]) / WIN


rows = []          # per item: system, P0, P1, P2, flagged, rescued by vocals, recall on vocals
for k, g in G.items():
    off, on, v = g["False"], g["True"], V[k]
    flagged = wps(off) < WPS
    rescued = flagged and wps(v) >= WPS
    p2 = float(v["recall"]) if rescued else float(off["recall"])
    rows.append({"key": k, "sys": SYS[k[:2]], "P0": float(on["recall"]), "P1": float(off["recall"]), "P2": p2,
                 "flagged": flagged, "rescued": rescued, "voc": float(v["recall"]), "voc_wps": wps(v)})
out = {"per_system": {}}
for s in ["2B·B", "2B·C", "XL·B", "XL·C", "all"]:
    rs = [x for x in rows if s == "all" or x["sys"] == s]
    fl = [x for x in rs if x["flagged"]]
    nf = [x for x in rs if not x["flagged"]]
    out["per_system"][s] = {
        "n": len(rs), "P0_mean": round(st.mean(x["P0"] for x in rs), 4), "P1_mean": round(st.mean(x["P1"] for x in rs), 4),
        "P2_mean": round(st.mean(x["P2"] for x in rs), 4),
        "below_0.3": {p: sum(x[p] < 0.3 for x in rs) for p in ("P0", "P1", "P2")},
        "flagged": len(fl), "rescued": sum(x["rescued"] for x in fl),
        "flagged_recall_P1_mean": round(st.mean(x["P1"] for x in fl), 4) if fl else None,
        "flagged_recall_P2_mean": round(st.mean(x["P2"] for x in fl), 4) if fl else None,
        "flagged_P2_ge_0.6": sum(x["P2"] >= 0.6 for x in fl),
        "still_flagged_after_P2": sum(not x["rescued"] for x in fl),
        "unflagged_voc_minus_mix_mean": round(st.mean(x["voc"] - x["P1"] for x in nf), 4),
        "unflagged_voc_worse": sum(x["voc"] < x["P1"] for x in nf), "unflagged_voc_better": sum(x["voc"] > x["P1"] for x in nf),
        "all_vocals_mean": round(st.mean(x["voc"] for x in rs), 4)}
for p in ("P0", "P1", "P2"):
    out[f"ranking_{p}"] = sorted(((s, out["per_system"][s][f"{p}_mean"]) for s in ("2B·B", "2B·C", "XL·B", "XL·C")), key=lambda x: -x[1])
(R / "p2_summary.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
print("system  n   P0     P1     P2   | recall<0.3 P0/P1/P2 | flagged rescued rescued>=0.6 still-flagged | flagged mean P1->P2 | unflagged: vocals-mix mean (worse/better) | all-vocals mean")
for s, v in out["per_system"].items():
    b = v["below_0.3"]
    print(f"{s:5s} {v['n']:3d} {v['P0_mean']:.3f} {v['P1_mean']:.3f} {v['P2_mean']:.3f} | {b['P0']:3d}/{b['P1']:3d}/{b['P2']:3d} | "
          f"{v['flagged']:2d} {v['rescued']:2d} {v['flagged_P2_ge_0.6']:2d} {v['still_flagged_after_P2']:2d} | "
          f"{v['flagged_recall_P1_mean']}→{v['flagged_recall_P2_mean']} | {v['unflagged_voc_minus_mix_mean']:+.3f} ({v['unflagged_voc_worse']}/{v['unflagged_voc_better']}) | {v['all_vocals_mean']:.3f}")
for p in ("P0", "P1", "P2"):
    print("ranking", p, out[f"ranking_{p}"])
print("items still flagged (also < 0.5 words/s on vocals):")
for x in rows:
    if x["flagged"] and not x["rescued"]:
        # the "text" column is only present in the full per-item CSVs from the experiment scripts
        k = x["key"]; print("  ", x["sys"], k[2], "s" + k[3], "P1", x["P1"], "vocals recall", x["voc"], "|", V[k].get("text", "")[:50].replace("\n", " "))
