#!/usr/bin/env python3
"""Separate-then-VAD control: does the ranking still reverse when Silero VAD is run on
separated vocals (HDemucs) instead of the mixture?

Reads  : ../results/sep_vad48.csv, sep_vad48_fam.csv, sep_vad48_dr.csv (vocals + VAD, from exp_sep_vad.py)
         ../results/vad_gen48*.csv (same songs on the mixture, VAD off / on)
Writes : ../results/sep_vad_summary.json
Stats  : lyric-clustered bootstrap (20000 resamples, seed 101); each lyric is first averaged over seeds.
Supports the paper's "separate, then VAD" control (paper_numbers.py reads sep_vad_summary.json).
Also exports SYS, PAIRS, load, per_lyric, boot for analyze_energy_gate.py and analyze_sep2.py.
"""
import csv, json
from collections import defaultdict
from pathlib import Path
import numpy as np

# ===== Edit here =====
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"   # folder holding the result CSV/JSON files
N_BOOT = 20000      # bootstrap resamples
BOOT_SEED = 101     # bootstrap RNG seed
# =====================

R = RESULTS_DIR
SYS = {("vad48_2b", "B"): "2B", ("vad48_2b", "C"): "2B+P", ("vad48_xl", "B"): "XL", ("vad48_xl", "C"): "XL+P",
       ("vad48_heart", "H"): "HeartMuLa", ("vad48_levo", "L"): "LeVo", ("vad48_diffrhythm", "D"): "DiffRhythm"}
PAIRS = [("XL", "2B"), ("XL+P", "2B"), ("XL", "2B+P"), ("XL+P", "2B+P"), ("2B+P", "2B"), ("XL+P", "XL"), ("LeVo", "HeartMuLa"),
         ("DiffRhythm", "HeartMuLa"), ("DiffRhythm", "LeVo"), ("DiffRhythm", "2B"), ("DiffRhythm", "XL"), ("DiffRhythm", "2B+P")]


def load():
    d = defaultdict(dict)   # cond -> {(sys, song, seed): (recall, kept)}
    for f in [x for x in ["sep_vad48.csv", "sep_vad48_fam.csv", "sep_vad48_dr.csv"] if (R / x).exists()]:
        for r in csv.DictReader(open(R / f)):
            d["vocals_vad"][(SYS[(r["exp"], r["arm"])], r["song"], r["seed"])] = (float(r["recall"]), float(r["dur_after_vad"]))
    for f in [x for x in ["vad_gen48.csv", "vad_gen48_heart.csv", "vad_gen48_levo.csv", "vad_gen48_dr.csv"] if (R / x).exists()]:
        for r in csv.DictReader(open(R / f)):
            if r["src"] != "gen" or (r["exp"], r["arm"]) not in SYS:
                continue
            c = "mix_off" if r["vad"] == "False" else "mix_on"
            d[c][(SYS[(r["exp"], r["arm"])], r["song"], r["seed"])] = (float(r["recall"]), None)
    return d


def per_lyric(tab, s):
    m = defaultdict(list)
    for (sy, song, sd), (rec, _) in tab.items():
        if sy == s:
            m[song].append(rec)
    return {k: np.mean(v) for k, v in m.items()}


def boot(a, b, n=N_BOOT, seed=BOOT_SEED):
    ks = sorted(set(a) & set(b)); x = np.array([a[k] - b[k] for k in ks])
    rng = np.random.default_rng(seed); idx = rng.integers(0, len(ks), (n, len(ks)))
    bs = x[idx].mean(1)
    return float(x.mean()), [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], len(ks)


def main():
    d = load(); out = {"n": {c: len(t) for c, t in d.items()}, "systems": {}, "pairs": {}}
    for s in sorted(set(SYS.values())):
        t = {k: v for k, v in d["vocals_vad"].items() if k[0] == s}
        if not t:
            continue
        out["systems"][s] = {"n": len(t), "kept_median": float(np.median([v[1] for v in t.values()])),
                             "recall_vocals_vad": float(np.mean([v[0] for v in t.values()])),
                             **{f"recall_{c}": float(np.mean([v[0] for k, v in d[c].items() if k[0] == s])) for c in ("mix_off", "mix_on")}}
    for a, b in PAIRS:
        out["pairs"][f"{a} - {b}"] = {}
        for c in ("mix_off", "mix_on", "vocals_vad"):
            pa, pb = per_lyric(d[c], a), per_lyric(d[c], b)
            if pa and pb:
                m, ci, n = boot(pa, pb)
                out["pairs"][f"{a} - {b}"][c] = {"d": round(m, 3), "ci": [round(x, 3) for x in ci], "n_lyrics": n,
                                                 "sig": "+" if ci[0] > 0 else ("-" if ci[1] < 0 else "0")}
    json.dump(out, open(R / "sep_vad_summary.json", "w"), indent=1, ensure_ascii=False)
    print(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
