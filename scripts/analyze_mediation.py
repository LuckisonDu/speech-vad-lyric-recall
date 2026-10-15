#!/usr/bin/env python3
"""Mediation analysis: how much of a VAD-on score gap between systems comes from the VAD keeping
different amounts of audio (the measuring tool), and how much from the song itself being intelligible
(the VAD-off score).

Model (no system labels; fitted on all 1536 songs (key = system x song x seed) from three families):
    recall_on = b0 + b1*off + b2*k + b3*off*k,   off = VAD-off recall, k = VAD kept seconds / 120 (clipped to [0,1])
For a system pair A - B (paired by song and seed), split the VAD-on gap with the fitted f:
    coverage part  = mean f(off_A, k_A) - f(off_A, k_B)   (only A's kept seconds swapped for B's)
    intrinsic part = mean f(off_A, k_B) - f(off_B, k_B)   (same kept seconds, only the VAD-off score differs)
    residual       = observed gap - (coverage + intrinsic)
CIs: lyric-level (48 lyrics) cluster bootstrap, refitting each time (N=2000, seed=101). Also reports
the model R^2 and the coefficients of system dummies added to the model (residual system effect).
Post-hoc analysis on results already seen (stated as such in the paper).

Reads  : ../results/vad_gen48.csv, vad_gen48_heart.csv, vad_gen48_levo.csv, vad_stats48.csv, vad_stats48_heart_levo.csv
Writes : ../results/mediation_summary.json (also used for the supplementary decomposition table)
"""
import csv, json
from collections import defaultdict
from pathlib import Path
import numpy as np

# ===== Edit here =====
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"   # folder holding the result CSV/JSON files
CROP_SEC = 120.0    # length of the analysed crop; k = kept_sec / CROP_SEC, capped at 1
N_BOOT = 2000       # cluster-bootstrap resamples (each one refits the model)
BOOT_SEED = 101     # bootstrap RNG seed
SEEDS = ("1", "2", "3", "4", "5", "6")   # generation seeds; ACE-Step has 1-6, HeartMuLa / LeVo have 1-4, missing combinations are skipped
N_EXPECT = 4 * 48 * 6 + 2 * 48 * 4       # = 1536: ACE-Step 4 configurations x 288 + HeartMuLa / LeVo 192 each
# =====================

R = RESULTS_DIR
SYS = {("vad48_2b", "B"): "2B", ("vad48_2b", "C"): "2B+pl", ("vad48_xl", "B"): "XL", ("vad48_xl", "C"): "XL+pl",
       ("vad48_heart", "H"): "HeartMuLa", ("vad48_levo", "L"): "LeVo"}
rec = defaultdict(dict)
for fn in ("vad_gen48.csv", "vad_gen48_heart.csv", "vad_gen48_levo.csv"):
    for r in csv.DictReader(open(R / fn)):
        if r["src"] == "gen":
            rec[(SYS[(r["exp"], r["arm"])], r["song"], r["seed"])][r["vad"]] = float(r["recall"])
kept = {}
for fn in ("vad_stats48.csv", "vad_stats48_heart_levo.csv"):
    for r in csv.DictReader(open(R / fn)):
        if r["src"] == "gen":
            kept[(SYS[(r["exp"], r["arm"])], r["song"], r["seed"])] = min(1.0, float(r["kept_sec"]) / CROP_SEC)
keys = sorted(k for k in rec if k in kept)
assert len(keys) == N_EXPECT, len(keys)   # 4 ACE configs x 288 + HeartMuLa / LeVo 192 each
songs = sorted({k[1] for k in keys})
names = list(SYS.values())


def X(off, k):
    off, k = np.asarray(off), np.asarray(k)
    return np.stack([np.ones_like(off), off, k, off * k], axis=1)


def fit(ks):
    off = [rec[k]["False"] for k in ks]; kk = [kept[k] for k in ks]; y = np.array([rec[k]["True"] for k in ks])
    beta, *_ = np.linalg.lstsq(X(off, kk), y, rcond=None)
    pred = X(off, kk) @ beta
    r2 = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    return beta, r2


f = lambda beta, off, k: float(X([off], [k])[0] @ beta)


def decompose(beta, a, b, ss):
    tot = cov = intr = n = 0
    for s in ss:
        for sd in SEEDS:
            ka, kb = (a, s, sd), (b, s, sd)
            if ka not in rec or kb not in rec:
                continue
            oa, ob = rec[ka]["False"], rec[kb]["False"]
            tot += rec[ka]["True"] - rec[kb]["True"]
            cov += f(beta, oa, kept[ka]) - f(beta, oa, kept[kb])
            intr += f(beta, oa, kept[kb]) - f(beta, ob, kept[kb])
            n += 1
    return tot / n, cov / n, intr / n, (tot - cov - intr) / n


beta, r2 = fit(keys)
# residual system effect after adding system dummies (reference = 2B)
off = [rec[k]["False"] for k in keys]; kk = [kept[k] for k in keys]; y = np.array([rec[k]["True"] for k in keys])
D = np.stack([[1.0 if k[0] == nm else 0.0 for k in keys] for nm in names if nm != "2B"], axis=1)
b2, *_ = np.linalg.lstsq(np.hstack([X(off, kk), D]), y, rcond=None)
pred2 = np.hstack([X(off, kk), D]) @ b2
r2_sys = 1 - ((y - pred2) ** 2).sum() / ((y - y.mean()) ** 2).sum()
# reference R^2 with system dummies + VAD-off score only (no kept seconds)
b3, *_ = np.linalg.lstsq(np.hstack([np.stack([np.ones(len(keys)), np.array(off)], 1), D]), y, rcond=None)
pred3 = np.hstack([np.stack([np.ones(len(keys)), np.array(off)], 1), D]) @ b3
r2_nokept = 1 - ((y - pred3) ** 2).sum() / ((y - y.mean()) ** 2).sum()

PAIRS = [("XL", "2B"), ("XL+pl", "2B"), ("XL", "2B+pl"), ("XL+pl", "2B+pl"), ("LeVo", "HeartMuLa"), ("LeVo", "2B+pl")]
rng = np.random.default_rng(BOOT_SEED)
out = {"model": "recall_on ~ 1 + off + k + off*k (no system terms), fit on " + str(len(keys)) + " songs",
       "beta": [round(float(x), 4) for x in beta], "r2": round(float(r2), 3),
       "r2_with_system_dummies": round(float(r2_sys), 3), "r2_system_and_off_without_kept": round(float(r2_nokept), 3),
       "residual_system_coef_vs_2B": {nm: round(float(c), 4) for nm, c in zip([n for n in names if n != "2B"], b2[4:])},
       "pairs": {}}
boots = {p: [] for p in PAIRS}
for _ in range(N_BOOT):
    smp = [songs[i] for i in rng.integers(0, len(songs), len(songs))]
    ks = [(nm, s, sd) for s in smp for nm in names for sd in SEEDS if (nm, s, sd) in rec]
    bb, _ = fit(ks)
    for p in PAIRS:
        boots[p].append(decompose(bb, p[0], p[1], smp))
for p in PAIRS:
    t, c, i, r = decompose(beta, p[0], p[1], songs)
    arr = np.array(boots[p])
    ci = lambda j: [round(float(np.percentile(arr[:, j], 2.5)), 3), round(float(np.percentile(arr[:, j], 97.5)), 3)]
    out["pairs"][f"{p[0]} − {p[1]}"] = {"observed_on_gap": round(t, 3), "coverage_part": round(c, 3), "coverage_ci95": ci(1),
                                        "intrinsic_part": round(i, 3), "intrinsic_ci95": ci(2), "residual": round(r, 3), "residual_ci95": ci(3),
                                        "coverage_share_of_gap": round(c / t, 2) if abs(t) > 1e-9 else None}
(R / "mediation_summary.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
print(json.dumps(out, indent=1, ensure_ascii=False))
