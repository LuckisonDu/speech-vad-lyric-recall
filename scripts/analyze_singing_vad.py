#!/usr/bin/env python3
"""Singing-voice VAD control: analysis (see exp_singing_vad.py).
Reads : ../results/vad_gen48.csv (VAD off, Silero on), ../results/energy_gate48.csv (alternative: vocal-stem energy gate),
        ../results/vad_stats48.csv (seconds kept by Silero), ../results/singing_vad/asr_*.csv (transcription after D1 gating),
        ../results/singing_vad/seg_stats.csv (interval overlap and singing ground truth).
Writes: ../results/singing_vad_summary.json
Criterion: paired recall difference XL.B minus 2B.B per lyric (bootstrap over lyrics, N=20000, seed 101), for each gating condition,
and the interaction (difference with gating minus difference with VAD off).
"""
import csv, glob, json, random, statistics as st
from collections import defaultdict
from pathlib import Path

# ===== Edit here =====
R = Path(__file__).resolve().parent.parent / "results"
SV = R / "singing_vad"
SEEDS = [1, 2]                       # main-analysis seeds (extra seeds that have ASR rows are listed separately as "ext")
N_BOOT, BOOT_SEED = 20000, 101
OUT_JSON = R / "singing_vad_summary.json"
# =====================
EXPA, EXPB = "vad48_2b", "vad48_xl"          # A = 2B.B, B = XL.B; difference = XL - 2B


def load(path, filt=lambda r: True):
    return [r for r in csv.DictReader(open(path)) if filt(r)]


def tab(rows, cond_of, val="recall"):
    d = defaultdict(dict)
    for r in rows:
        c = cond_of(r)
        if c is None: continue
        d[c][(r["exp"], r["song"], int(r["seed"]))] = float(r[val])
    return d


gen = load(R / "vad_gen48.csv", lambda r: r["src"] == "gen" and r["arm"] == "B")
conds = {}
for val in ("recall", "wer"):
    conds[val] = tab(gen, lambda r: "off" if r["vad"] == "False" else "silero", val)
    eg = load(R / "energy_gate48.csv", lambda r: r["arm"] == "B")
    for k, v in tab(eg, lambda r: "energy_gate_stem", val).items(): conds[val][k] = v
    for f in glob.glob(str(SV / "asr_*.csv")):
        for r in load(f):
            if r["arm"] != "B": continue
            conds[val].setdefault(r["cond"], {})[(r["exp"], r["song"], int(r["seed"]))] = float(r[val])
kept = defaultdict(dict)
for r in load(R / "vad_stats48.csv", lambda r: r["arm"] == "B" and r.get("kept_sec") not in (None, "")):
    kept["silero"][(r["exp"], r["song"], int(r["seed"]))] = float(r["kept_sec"])
for r in load(R / "energy_gate48.csv", lambda r: r["arm"] == "B"):
    kept["energy_gate_stem"][(r["exp"], r["song"], int(r["seed"]))] = float(r["kept_sec"])
for f in glob.glob(str(SV / "asr_*.csv")):
    for r in load(f):
        if r["arm"] == "B": kept[r["cond"]][(r["exp"], r["song"], int(r["seed"]))] = float(r["kept_sec"])


def songs_with(cond, seeds):
    d = conds["recall"].get(cond, {})
    return sorted({s for (e, s, sd) in d if e == EXPA and all((e, s, k) in d and (EXPB, s, k) in d for k in seeds)})


def paired(cond, seeds, val="recall", lyrics=None):
    d = conds[val][cond]; L = lyrics or songs_with(cond, seeds)
    return {s: st.mean(d[(EXPB, s, k)] - d[(EXPA, s, k)] for k in seeds) for s in L}


def boot(vals, n=N_BOOT):
    rnd = random.Random(BOOT_SEED); x = list(vals.values()); m = len(x)
    bs = sorted(st.mean(rnd.choice(x) for _ in range(m)) for _ in range(n))
    return st.mean(x), [bs[int(0.025 * n)], bs[int(0.975 * n)]]


def boot_diff(a, b, n=N_BOOT):
    """Bootstrap of b - a (paired by lyric): interaction = difference with gating minus difference with VAD off."""
    rnd = random.Random(BOOT_SEED + 1); L = sorted(set(a) & set(b)); m = len(L)
    vals = [b[s] - a[s] for s in L]
    bs = sorted(st.mean(rnd.choice(vals) for _ in range(m)) for _ in range(n))
    return st.mean(vals), [bs[int(0.025 * n)], bs[int(0.975 * n)]]


res = {"seeds_main": SEEDS, "conds": {}}
order = ["off", "silero", "d1_cal", "d1_calx0.5", "d1_calx2", "energy_gate_stem"]
for seedset, tag in ((SEEDS, "main"), (None, "ext")):
    if tag == "ext":
        allseeds = sorted({sd for (e, s, sd) in conds["recall"].get("d1_cal", {})})
        if allseeds == sorted(SEEDS) or not allseeds: continue
        seedset = allseeds
    block = {}
    common = None
    for c in order:
        if c not in conds["recall"]: continue
        L = set(songs_with(c, seedset)); common = L if common is None else common & L
    L = sorted(common or [])
    for c in order:
        if c not in conds["recall"] or not L: continue
        mean_a = st.mean(conds["recall"][c][(EXPA, s, k)] for s in L for k in seedset)
        mean_b = st.mean(conds["recall"][c][(EXPB, s, k)] for s in L for k in seedset)
        dd, ci = boot(paired(c, seedset, "recall", L))
        dw, ciw = boot(paired(c, seedset, "wer", L)) if c in conds["wer"] else (None, None)
        kb = {e: st.mean(kept[c][(e, s, k)] for s in L for k in seedset if (e, s, k) in kept[c]) if c in kept else None for e in (EXPA, EXPB)}
        block[c] = {"n_lyrics": len(L), "seeds": seedset, "recall_2B": mean_a, "recall_XL": mean_b, "xl_minus_2b_recall": dd, "ci95": ci,
                    "xl_minus_2b_wer": dw, "wer_ci95": ciw, "kept_sec_2B": kb[EXPA], "kept_sec_XL": kb[EXPB]}
    if "off" in block:
        for c in order:
            if c in block and c != "off":
                dif, cii = boot_diff(paired("off", seedset, "recall", L), paired(c, seedset, "recall", L))
                block[c]["interaction_vs_off"] = dif; block[c]["interaction_ci95"] = cii
    res["conds"][tag] = block

# Interval overlap and singing ground truth (seg_stats.csv; arm B, seeds 1-2)
seg = load(SV / "seg_stats.csv") if (SV / "seg_stats.csv").exists() else []
segres = {}
for det in ("d1_cal", "d1_calx0.5", "d1_calx2"):
    rows = [r for r in seg if r["det"] == det and r["arm"] == "B" and int(r["seed"]) in SEEDS]
    if not rows: continue
    def col(k): return [float(r[k]) for r in rows if r.get(k) not in (None, "")]
    segres[det] = {"n_songs": len(rows), "kept_sec_median": st.median(col("kept_sec")), "silero_kept_sec_median": st.median(col("silero_kept_sec")),
                   "iou_discard_median": st.median(col("iou_discard")), "inter_discard_sec_total": sum(col("inter_discard_sec")),
                   "union_discard_sec_total": sum(col("union_discard_sec")),
                   "iou_discard_pooled": sum(col("inter_discard_sec")) / sum(col("union_discard_sec")) if sum(col("union_discard_sec")) else None,
                   "disc_sing_frac_median": st.median(col("disc_sing_frac")) if col("disc_sing_frac") else None,
                   "silero_disc_sing_frac_median": st.median(col("silero_disc_sing_frac")) if col("silero_disc_sing_frac") else None,
                   "sing_kept_recall_median": st.median(col("sing_kept_recall")) if col("sing_kept_recall") else None,
                   "silero_sing_kept_recall_median": st.median(col("silero_sing_kept_recall")) if col("silero_sing_kept_recall") else None}
res["segments"] = segres
json.dump(res, open(OUT_JSON, "w"), ensure_ascii=False, indent=1)

print(json.dumps(res, ensure_ascii=False, indent=1))
