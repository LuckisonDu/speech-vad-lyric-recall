#!/usr/bin/env python3
"""How many songs and seeds are needed to resolve the comparison: draw n of the 48 songs without
replacement (seed 1 only, or the mean of seeds 1 and 2) and record how often the mean difference
XL·B - 2B·B is positive (on the full data it is positive with VAD off and negative with VAD on).
5000 draws per setting, RNG seed 101. Also reports, per VAD setting, the number of songs whose two
seeds disagree on the winner.

Reads:  ../results/vad_gen48.csv
Writes: ../results/sampling_summary.json
Supports: sample-size analysis (share of positive outcomes for n = 5/10/20/30/48 songs; seed disagreement).
"""
import csv, json, random
from collections import defaultdict
from pathlib import Path

# ===== Edit here =====
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"   # folder holding the result CSV/JSON files
N_SONGS = (5, 10, 20, 30, 48)   # subsample sizes
N_DRAWS = 5000                  # draws per setting
RNG_SEED = 101                  # RNG seed of the subsampling
# =====================

R = RESULTS_DIR
rec = defaultdict(dict)
for r in csv.DictReader(open(R / "vad_gen48.csv")):
    if r["src"] == "gen":
        rec[(r["exp"], r["arm"], r["song"], r["seed"], r["vad"])] = float(r["recall"])
songs = sorted({k[2] for k in rec})
def d(song, seed, vad):
    return rec[("vad48_xl", "B", song, seed, vad)] - rec[("vad48_2b", "B", song, seed, vad)]
out = {"n_songs_total": len(songs), "p_positive": {}}
rnd = random.Random(RNG_SEED)
for vad, tag in (("False", "vad_off"), ("True", "vad_on")):
    for seeds, stag in ((("1",), "one_seed"), (("1", "2"), "two_seeds")):
        for n in N_SONGS:
            pos = 0
            for _ in range(N_DRAWS):
                ss = rnd.sample(songs, n)
                m = sum(d(s, sd, vad) for s in ss for sd in seeds) / (n * len(seeds))
                pos += m > 0
            out["p_positive"][f"{tag}_{stag}_n{n}"] = round(pos / N_DRAWS, 3)
    disagree = sum((d(s, "1", vad) > 0) != (d(s, "2", vad) > 0) for s in songs if d(s, "1", vad) != 0 and d(s, "2", vad) != 0)
    out[f"{tag}_seed_disagree"] = disagree
(R / "sampling_summary.json").write_text(json.dumps(out, indent=1))
for k, v in out["p_positive"].items():
    print(k, v)
print({k: v for k, v in out.items() if k.endswith("disagree")})
