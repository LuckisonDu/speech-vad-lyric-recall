#!/usr/bin/env python3
"""Second-separator robustness check (htdemucs_ft), summarising the outputs of exp_sep2.py.

A: median seconds kept per system (Silero / energy gate; on the mixture, HDemucs vocals,
   htdemucs_ft vocals, and a g=0.5 controlled remix).
B: recall and paired differences (lyric-clustered bootstrap, 20000 resamples, seed 101,
   reusing analyze_sep_vad.per_lyric / boot).
   Conditions: mix_off / mix_on (vad_gen48.csv), hd_vocals_vad (sep_vad48.csv), hd_egate (energy_gate48.csv),
               ht_vocals_vad / ht_egate / ht_remix05_vad (sep2_asr.csv).
Reads  : ../results/sep2_kept.csv, vad_gen48.csv, sep_vad48.csv, energy_gate48.csv, sep2_asr.csv
Writes : ../results/sep2_summary.json
Supports the paper's second-separator robustness numbers.
"""
import csv, json
from collections import defaultdict
from pathlib import Path
import numpy as np
from analyze_sep_vad import per_lyric, boot

# ===== Edit here =====
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"   # folder holding the result CSV/JSON files
# =====================

R = RESULTS_DIR
SYS = {("vad48_2b", "B"): "2B", ("vad48_2b", "C"): "2B+P", ("vad48_xl", "B"): "XL", ("vad48_xl", "C"): "XL+P"}
PAIRS = [("XL", "2B"), ("XL+P", "2B+P"), ("XL+P", "2B"), ("XL", "2B+P"), ("2B+P", "2B"), ("XL+P", "XL")]
KCOLS = ["silero_mix", "silero_hdemucs", "silero_htft", "energy_hdemucs", "energy_htft", "silero_remix05_htft"]


def main():
    out = {"separator": "demucs 4.0.1 htdemucs_ft (shifts=1, overlap=0.25, first 120 s, 44.1 kHz stereo -> 16 kHz mono)",
           "kept_median": {}, "recall_mean": {}, "pairs": {}}
    K = defaultdict(lambda: defaultdict(list))
    for r in csv.DictReader(open(R / "sep2_kept.csv")):
        s = SYS[(r["exp"], r["arm"])]
        for c in KCOLS:
            K[s][c].append(float(r[c]))
    for s in SYS.values():
        out["kept_median"][s] = {c: round(float(np.median(K[s][c])), 2) for c in KCOLS} | {"n": len(K[s]["silero_mix"])}
    rows = list(csv.DictReader(open(R / "sep2_kept.csv")))
    agree = {}
    for g, a_, b_ in [("silero", "silero_hdemucs", "silero_htft"), ("energy", "energy_hdemucs", "energy_htft")]:
        x = np.array([float(r[a_]) for r in rows]); y = np.array([float(r[b_]) for r in rows])
        agree[g] = {"pearson_r": round(float(np.corrcoef(x, y)[0, 1]), 3), "median_abs_diff_sec": round(float(np.median(np.abs(x - y))), 2), "n": len(rows)}
    out["separator_agreement_per_song"] = agree
    d = defaultdict(dict)
    for r in csv.DictReader(open(R / "vad_gen48.csv")):
        if r["src"] == "gen" and (r["exp"], r["arm"]) in SYS:
            d["mix_off" if r["vad"] == "False" else "mix_on"][(SYS[(r["exp"], r["arm"])], r["song"], r["seed"])] = (float(r["recall"]), None)
    for f, c in [("sep_vad48.csv", "hd_vocals_vad"), ("energy_gate48.csv", "hd_egate")]:
        for r in csv.DictReader(open(R / f)):
            if (r["exp"], r["arm"]) in SYS:
                d[c][(SYS[(r["exp"], r["arm"])], r["song"], r["seed"])] = (float(r["recall"]), None)
    if (R / "sep2_asr.csv").exists():
        for r in csv.DictReader(open(R / "sep2_asr.csv")):
            d["ht_" + r["mode"]][(SYS[(r["exp"], r["arm"])], r["song"], r["seed"])] = (float(r["recall"]), float(r["kept_sec"]))
    out["n"] = {c: len(t) for c, t in d.items()}
    for s in SYS.values():
        out["recall_mean"][s] = {c: round(float(np.mean([v[0] for k, v in t.items() if k[0] == s])), 4) for c, t in d.items() if any(k[0] == s for k in t)}
    for a, b in PAIRS:
        out["pairs"][f"{a} - {b}"] = {}
        for c in d:
            pa, pb = per_lyric(d[c], a), per_lyric(d[c], b)
            if pa and pb:
                m, ci, n = boot(pa, pb)
                out["pairs"][f"{a} - {b}"][c] = {"d": round(m, 3), "ci": [round(x, 3) for x in ci], "n_lyrics": n,
                                                 "sig": "+" if ci[0] > 0 else ("-" if ci[1] < 0 else "0")}
    json.dump(out, open(R / "sep2_summary.json", "w"), indent=1, ensure_ascii=False)
    print(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
