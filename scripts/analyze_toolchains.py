#!/usr/bin/env python3
"""Summary of alternative recognition toolchains (exp_toolchains.py) and of the vocal-to-accompaniment
level ratio (exp_vocal_ratio.py). Toolchains: the same 384 generated songs, each tool with its default
settings, ranked side by side with faster-whisper VAD off / on (from vad_gen48.csv).

Reads:  ../results/vad_gen48.csv, ../results/tool_{fw_thr_0.3,fw_thr_0.1,openai,whisperx,hf,parakeet}.csv,
        ../results/vocal_ratio48.csv, ../results/vad_stats48.csv
Writes: ../results/toolchains_summary.json
Supports: toolchain table (per-system mean recall, ranking and XL·B-2B·B difference per toolchain),
vocal ratio medians and the Spearman correlation between vocal ratio and VAD kept seconds.
"""
import csv, json, random, statistics as st
from collections import defaultdict
from pathlib import Path

# ===== Edit here =====
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"   # folder holding the result CSV/JSON files
# (tool file tag, condition label written as JSON key); files are RESULTS_DIR / f"tool_{tag}.csv"
TOOLS = (("fw_thr_0.3", "faster-whisper VAD on (threshold 0.3)"), ("fw_thr_0.1", "faster-whisper VAD on (threshold 0.1)"),
         ("openai", "openai-whisper default"), ("whisperx", "WhisperX default (VAD always on)"),
         ("hf", "transformers pipeline"), ("parakeet", "Parakeet-TDT (non-Whisper)"))
# =====================

R = RESULTS_DIR
SYS = {("vad48_2b", "B"): "2B·B", ("vad48_2b", "C"): "2B·C", ("vad48_xl", "B"): "XL·B", ("vad48_xl", "C"): "XL·C"}
NAMES = ["2B·B", "2B·C", "XL·B", "XL·C"]
score = defaultdict(dict)       # condition -> {key: recall}
nhyp = defaultdict(dict)
for r in csv.DictReader(open(R / "vad_gen48.csv")):
    if r["src"] == "gen":
        c = "faster-whisper VAD off" if r["vad"] == "False" else "faster-whisper VAD on (threshold 0.5)"
        k = (SYS[(r["exp"], r["arm"])], r["song"], r["seed"]); score[c][k] = float(r["recall"]); nhyp[c][k] = int(r["n_hyp"])
for tool, label in TOOLS:
    for r in csv.DictReader(open(R / f"tool_{tool}.csv")):
        k = (SYS[(r["exp"], r["arm"])], r["song"], r["seed"]); score[label][k] = float(r["recall"]); nhyp[label][k] = int(r["n_hyp"])


def boot(by, n=20000, seed=101):
    songs = sorted(by); rnd = random.Random(seed); out = []
    for _ in range(n):
        tot = cnt = 0
        for _ in songs:
            x = by[songs[rnd.randrange(len(songs))]]; tot += sum(x); cnt += len(x)
        out.append(tot / cnt)
    out.sort(); return round(out[int(.025 * n)], 4), round(out[int(.975 * n)], 4)


summ = {}
for c, d in score.items():
    m = {n: round(st.mean(v for k, v in d.items() if k[0] == n), 4) for n in NAMES}
    by = defaultdict(list)
    for k, v in d.items():
        if k[0] == "2B·B" and ("XL·B", k[1], k[2]) in d:
            by[k[1]].append(d[("XL·B", k[1], k[2])] - v)
    diff = [x for xs in by.values() for x in xs]
    summ[c] = {"n": len(d), "mean": m, "ranking": sorted(NAMES, key=lambda n: -m[n]),
               "below_0.3": {n: sum(v < 0.3 for k, v in d.items() if k[0] == n) for n in NAMES},
               "zero_words": sum(nhyp[c][k] == 0 for k in d),
               "XL·B−2B·B": {"mean": round(st.mean(diff), 4), "ci95": boot(by)}}
vr = defaultdict(list)
for r in csv.DictReader(open(R / "vocal_ratio48.csv")):
    vr[SYS[(r["exp"], r["arm"])]].append((float(r["vr_db"]), float(r["vr_active_db"])))
summ["vocal_ratio_db_median"] = {n: {"whole": round(st.median(x[0] for x in vr[n]), 2), "active": round(st.median(x[1] for x in vr[n]), 2)} for n in NAMES}
# vocal ratio vs VAD kept time
kept = {(SYS[(r["exp"], r["arm"])], r["song"], r["seed"]): float(r["kept_sec"]) for r in csv.DictReader(open(R / "vad_stats48.csv")) if r["src"] == "gen"}
vrk = {(SYS[(r["exp"], r["arm"])], r["song"], r["seed"]): float(r["vr_db"]) for r in csv.DictReader(open(R / "vocal_ratio48.csv"))}
ks = [k for k in vrk if k in kept]


def rank(v):
    o = sorted(range(len(v)), key=lambda i: v[i]); r = [0.0] * len(v)
    for pos, i in enumerate(o):
        r[i] = pos
    return r


a, b = rank([vrk[k] for k in ks]), rank([kept[k] for k in ks]); ma, mb = st.mean(a), st.mean(b)
summ["spearman_vocalratio_vs_keptsec"] = round(sum((x - ma) * (y - mb) for x, y in zip(a, b)) / (sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b)) ** 0.5, 4)
(R / "toolchains_summary.json").write_text(json.dumps(summ, indent=1, ensure_ascii=False))
print("condition                                 mean recall 2B·B / 2B·C / XL·B / XL·C   ranking (best->worst)     XL·B-2B·B [95%]          zero words")
for c, v in summ.items():
    if c.startswith(("vocal", "spearman")):
        continue
    m = v["mean"]
    print(f"{c:40s} {m['2B·B']:.3f} / {m['2B·C']:.3f} / {m['XL·B']:.3f} / {m['XL·C']:.3f}   {'>'.join(v['ranking']):24s} {v['XL·B−2B·B']['mean']:+.3f} {v['XL·B−2B·B']['ci95']}  {v['zero_words']}")
print("vocal / accompaniment level ratio (dB, median, whole song / vocal-active frames):", summ["vocal_ratio_db_median"])
print("vocal ratio vs VAD kept seconds, Spearman:", summ["spearman_vocalratio_vs_keptsec"])
