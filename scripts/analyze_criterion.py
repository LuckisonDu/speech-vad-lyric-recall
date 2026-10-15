#!/usr/bin/env python3
"""A simple criterion for when a filter-then-score metric (e.g. lyric recall with VAD on)
reverses the ranking of two systems.

Approximation: score with VAD on ~= kept fraction c x score with VAD off a (system means).
So the VAD-on winner of A vs B is sign(c_A*a_A - c_B*a_B); a reversal happens iff
a_A > a_B while c_A / c_B < a_B / a_A (coverage ratio below the inverse quality ratio).
Test: predict VAD-on winner and reversal from only two means per system (c = mean kept
fraction from the stand-alone VAD run, a = mean VAD-off recall) and compare with observation.
  (1) default threshold 0.5, 6 systems (three families), 15 pairs;
  (2) the 4 ACE-Step systems at thresholds 0.3 and 0.1, 6 pairs each
      (kept seconds from vad_sweep48.csv, VAD-on recall from tool_fw_thr_*.csv).
Post-hoc analysis on data already seen (stated as such in the paper).

Reads  : ../results/vad_gen48.csv, vad_gen48_heart.csv, vad_gen48_levo.csv, tool_fw_thr_0.3.csv,
         tool_fw_thr_0.1.csv, vad_stats48.csv, vad_stats48_heart_levo.csv, vad_sweep48.csv
Writes : ../results/criterion_summary.json
"""
import csv, json, statistics as st
from collections import defaultdict
from itertools import combinations
from pathlib import Path

# ===== Edit here =====
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"   # folder holding the result CSV/JSON files
CROP_SEC = 120      # length of the analysed crop; kept fraction = kept_sec / CROP_SEC, capped at 1
ACE_SEEDS = ("1", "2")   # vad_gen48.csv / vad_stats48.csv contain ACE-Step seeds 1-6, but the inputs for thresholds 0.3 and 0.1
                         # (vad_sweep48.csv, tool_fw_thr_*.csv) only have seeds 1 and 2. The criterion uses system means, and
                         # a, c and the VAD-on score must come from the same songs, so ACE-Step uses seeds 1 and 2 only
                         # (HeartMuLa / LeVo use all their seeds as before)
# =====================

R = RESULTS_DIR
ACE = {"2B", "2B+pl", "XL", "XL+pl"}
use = lambda k: k[0] not in ACE or k[2] in ACE_SEEDS
SYS = {("vad48_2b", "B"): "2B", ("vad48_2b", "C"): "2B+pl", ("vad48_xl", "B"): "XL", ("vad48_xl", "C"): "XL+pl",
       ("vad48_heart", "H"): "HeartMuLa", ("vad48_levo", "L"): "LeVo"}
off, on = {}, defaultdict(dict)
for fn in ("vad_gen48.csv", "vad_gen48_heart.csv", "vad_gen48_levo.csv"):
    for r in csv.DictReader(open(R / fn)):
        if r["src"] == "gen":
            k = (SYS[(r["exp"], r["arm"])], r["song"], r["seed"])
            if not use(k):
                continue
            if r["vad"] == "False":
                off[k] = float(r["recall"])
            else:
                on["0.5"][k] = float(r["recall"])
for t in ("0.3", "0.1"):
    for r in csv.DictReader(open(R / f"tool_fw_thr_{t}.csv")):
        on[t][(SYS[(r["exp"], r["arm"])], r["song"], r["seed"])] = float(r["recall"])
kept = defaultdict(dict)
for fn in ("vad_stats48.csv", "vad_stats48_heart_levo.csv"):
    for r in csv.DictReader(open(R / fn)):
        if r["src"] == "gen":
            if use((SYS[(r["exp"], r["arm"])], r["song"], r["seed"])):
                kept["0.5"][(SYS[(r["exp"], r["arm"])], r["song"], r["seed"])] = min(1.0, float(r["kept_sec"]) / CROP_SEC)
for r in csv.DictReader(open(R / "vad_sweep48.csv")):
    if r["src"] == "gen":
        for t in ("0.3", "0.1"):
            kept[t][(SYS[(r["exp"], r["arm"])], r["song"], r["seed"])] = min(1.0, float(r[f"kept_{t}"]) / CROP_SEC)


def run(t, names):
    mean = lambda d, n: st.mean(v for k, v in d.items() if k[0] == n)
    m = {n: {"a": mean(off, n), "on": mean(on[t], n), "c": mean(kept[t], n)} for n in names}
    rows = []
    for x, y in combinations(names, 2):
        pred = m[x]["c"] * m[x]["a"] - m[y]["c"] * m[y]["a"]
        obs = m[x]["on"] - m[y]["on"]; offd = m[x]["a"] - m[y]["a"]
        rows.append({"threshold": t, "a": x, "b": y, "off_diff": round(offd, 4), "on_diff_observed": round(obs, 4),
                     "on_diff_predicted": round(pred, 4), "sign_correct": (pred > 0) == (obs > 0),
                     "reversal_predicted": (offd > 0) != (pred > 0), "reversal_observed": (offd > 0) != (obs > 0),
                     "coverage_ratio": round(m[x]["c"] / m[y]["c"], 3), "inverse_quality_ratio": round(m[y]["a"] / m[x]["a"], 3)})
    return rows, {n: {k: round(v, 4) for k, v in d.items()} for n, d in m.items()}


rows5, means5 = run("0.5", list(SYS.values()))
ace = ["2B", "2B+pl", "XL", "XL+pl"]
rows3, _ = run("0.3", ace); rows1, _ = run("0.1", ace)
allr = rows5 + rows3 + rows1
summ = lambda rs: {"n": len(rs), "sign_correct": sum(r["sign_correct"] for r in rs),
                   "reversal_classified_correctly": sum(r["reversal_predicted"] == r["reversal_observed"] for r in rs),
                   "reversals_observed": sum(r["reversal_observed"] for r in rs), "reversals_predicted": sum(r["reversal_predicted"] for r in rs)}
out = {"default_threshold_six_systems": summ(rows5), "ace_threshold_0.3": summ(rows3), "ace_threshold_0.1": summ(rows1),
       "all": summ(allr), "misses": [r for r in allr if not r["sign_correct"] or r["reversal_predicted"] != r["reversal_observed"]],
       "system_means_threshold_0.5": means5, "rows": allr}
(R / "criterion_summary.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
print(json.dumps({k: v for k, v in out.items() if k not in ("rows", "system_means_threshold_0.5")}, indent=1, ensure_ascii=False))
