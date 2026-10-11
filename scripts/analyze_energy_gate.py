#!/usr/bin/env python3
"""Summary of the energy-gate control (exp_energy_gate.py): an RMS gate on separated vocals,
reported next to mixture VAD off / on and separated vocals + Silero.

Reads  : ../results/energy_gate48.csv, energy_gate_truth.csv, plus the files read by analyze_sep_vad.load()
Writes : ../results/energy_gate_summary.json
Stats  : lyric-clustered bootstrap (20000 resamples, seed 101), same as the paper.
Supports the paper's singing-specific gate control and the DiffRhythm (fourth family) numbers.
"""
import csv, json
from pathlib import Path
import numpy as np
from analyze_sep_vad import SYS, PAIRS, load, per_lyric, boot

# ===== Edit here =====
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"   # folder holding the result CSV/JSON files
LOW_RECALL = 0.3    # songs with energy-gated recall below this are counted as "empty_or_low"
# =====================

R = RESULTS_DIR


def main():
    d = load()
    for r in csv.DictReader(open(R / "energy_gate48.csv")):
        d["energy"][(SYS[(r["exp"], r["arm"])], r["song"], r["seed"])] = (float(r["recall"]), float(r["kept_sec"]))
    out = {"n": {c: len(t) for c, t in d.items()}, "systems": {}, "pairs": {}}
    for s in sorted(set(SYS.values())):
        t = {k: v for k, v in d["energy"].items() if k[0] == s}
        out["systems"][s] = {"n": len(t), "kept_median": float(np.median([v[1] for v in t.values()])),
                             "recall_energy": float(np.mean([v[0] for v in t.values()])),
                             "empty_or_low": int(sum(v[0] < LOW_RECALL for v in t.values())),
                             **{f"recall_{c}": float(np.mean([v[0] for k, v in d[c].items() if k[0] == s])) for c in ("mix_off", "mix_on", "vocals_vad")}}
    for a, b in PAIRS:
        out["pairs"][f"{a} - {b}"] = {}
        for c in ("mix_off", "mix_on", "vocals_vad", "energy"):
            pa, pb = per_lyric(d[c], a), per_lyric(d[c], b)
            if pa and pb:
                m, ci, n = boot(pa, pb)
                out["pairs"][f"{a} - {b}"][c] = {"d": round(m, 3), "ci": [round(x, 3) for x in ci], "n_lyrics": n,
                                                 "sig": "+" if ci[0] > 0 else ("-" if ci[1] < 0 else "0")}
    out["contradictions_vs_off"] = sum(1 for v in out["pairs"].values()
                                       if v["energy"]["sig"] != "0" and v["mix_off"]["sig"] != "0" and v["energy"]["sig"] != v["mix_off"]["sig"])
    tr = list(csv.DictReader(open(R / "energy_gate_truth.csv")))
    tw = sum(int(r["n_words"]) for r in tr); kw = sum(int(r["words_kept"]) for r in tr)
    out["truth"] = {"n_songs": len(tr), "n_words": tw, "words_dropped_pooled": round(1 - kw / tw, 4),
                    "median_song_frac_dropped": float(np.median([float(r["frac_words_dropped"]) for r in tr]))}
    json.dump(out, open(R / "energy_gate_summary.json", "w"), indent=1, ensure_ascii=False)
    print(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
