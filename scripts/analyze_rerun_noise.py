#!/usr/bin/env python3
"""Re-run noise of faster-whisper on GPU compared with the main effect.
Data: rows duplicated by two processes that were appending to the same table while seeds 3-6 were evaluated:
the same audio, the same model, the same settings, transcribed twice (results/vad_gen48_with_dups.csv = the table before de-duplication).
Per-song re-run sd = sqrt(mean(d^2) / 2); re-run noise of a mean over N_PAIRS pairs = sd * sqrt(2) / sqrt(N_PAIRS) (the two systems are re-run independently).
Reads : ../results/vad_gen48_with_dups.csv, ../results/vad_gen48_summary.json
Writes: ../results/rerun_noise_summary.json
"""
import collections, csv, json, math
from pathlib import Path

# ===== Edit here =====
R = Path(__file__).resolve().parent.parent / "results"
DUPS_CSV = R / "vad_gen48_with_dups.csv"          # table before de-duplication (contains duplicate keys)
MAIN_SUMMARY = R / "vad_gen48_summary.json"       # source of the main effect
OUT = R / "rerun_noise_summary.json"
# =====================
g = collections.defaultdict(list)
for r in csv.DictReader(open(DUPS_CSV)):
    g[(r["exp"], r["arm"], r["song"], r["seed"], r["vad"])].append(float(r["recall"]))
main = json.load(open(MAIN_SUMMARY))["compare"]["XL·B − 2B·B"]
n_pairs = main["pairs"]
out = {"source": "duplicate transcriptions of identical audio (2B systems, seeds 3-6)", "n_pairs_main": n_pairs,
       "systems_with_duplicates": sorted({f"{k[0]}/{k[1]}" for k, v in g.items() if len(v) > 1})}
for vad, tag in (("False", "off"), ("True", "on")):
    d = [v[0] - v[1] for k, v in g.items() if len(v) > 1 and k[4] == vad]
    sd = math.sqrt(sum(x * x for x in d) / len(d) / 2)
    noise = sd * math.sqrt(2) / math.sqrt(n_pairs)
    eff = main[f"vad_{tag}"]["diff_mean"]
    out[tag] = {"n_dup": len(d), "n_differ": sum(x != 0 for x in d), "max_abs_diff": round(max(abs(x) for x in d), 4),
                "rerun_sd_per_song": round(sd, 4), "noise_sd_pair_mean": round(noise, 4),
                "main_effect": eff, "effect_over_noise": round(abs(eff) / noise, 1)}
json.dump(out, open(OUT, "w"), ensure_ascii=False, indent=1)
print(json.dumps(out, ensure_ascii=False, indent=1))
