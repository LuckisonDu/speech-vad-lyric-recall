#!/usr/bin/env python3
"""Cross-family analysis: ACE-Step 1.5 (four systems), HeartMuLa and LeVo side by side under VAD off/on.

Reads (no re-recognition):
  ../results/vad_gen48.csv, vad_gen48_heart.csv, vad_gen48_levo.csv     recall with VAD on / off
  ../results/sep_redecode48_heart.csv, sep_redecode48_levo.csv          re-recognition on separated vocals (for P2)
  ../results/vad_stats48.csv, vad_stats48_heart_levo.csv                VAD kept seconds
  WER / PER paired differences are recomputed here with the same algorithm as analyze_wer_per.py (g2p_en).
  NOTE: WER/PER need the transcript ("text") column of vad_gen48_heart.csv / vad_gen48_levo.csv, which the
  public CSVs omit; use the full per-item CSVs produced by the experiment scripts, plus one
  "<song>.meta.json" per song with "ref_words".
Writes: ../results/cross_family_summary.json
Supports: cross-family section (per-family recall, kept seconds, P0/P1/P2, LeVo vs HeartMuLa and LeVo vs
ACE-Step comparisons, pooled and per-system Spearman correlations in the supplement).

Statistics as in the main experiment: paired on song and seed, cluster bootstrap over songs (48),
N=20000, seed=101; Wilcoxon signed-rank on per-song means over seeds (only if scipy is available).
P2 rule (pre-registered): if VAD-off output has < 0.5 words/s (fewer than 60 words in 120 s) and
re-recognition on separated vocals gives >= 0.5 words/s, use the separated-vocals result.
Absolute scores across families were recognised on different machines and are indicative only;
HeartMuLa and LeVo were recognised on the same machine.

Requirements: g2p_en and scipy (set NLTK_DATA if the g2p_en NLTK data lives elsewhere).
"""
import csv, json, random, re, statistics as st
from collections import defaultdict
from pathlib import Path
from verify_metrics import norm_words, wer

try:
    from scipy.stats import wilcoxon
except Exception:
    wilcoxon = None

# ===== Edit here =====
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"   # folder holding the result CSV/JSON files
META = Path("data/lyrics_meta")   # folder with "<song>.meta.json" files, each containing "ref_words" (reference lyric words of the 120 s window)
SEEDS = ("1", "2", "3", "4", "5", "6")   # ACE-Step has seeds 1-6, HeartMuLa / LeVo have 1-4. A paired comparison uses the seeds present on both
                                         # sides (ACE-Step vs HeartMuLa / LeVo = seeds 1-4); per-system statistics use all seeds of that system
# =====================

R = RESULTS_DIR
SYS = {("vad48_2b", "B"): "2B·B", ("vad48_2b", "C"): "2B·C", ("vad48_xl", "B"): "XL·B", ("vad48_xl", "C"): "XL·C",
       ("vad48_heart", "H"): "HeartMuLa", ("vad48_levo", "L"): "LeVo"}

rec = defaultdict(dict)   # (system, song, seed) -> {vad: row}
for fn in ("vad_gen48.csv", "vad_gen48_heart.csv", "vad_gen48_levo.csv"):
    for r in csv.DictReader(open(R / fn)):
        if r["src"] == "gen":
            rec[(SYS[(r["exp"], r["arm"])], r["song"], r["seed"])][r["vad"]] = r
kept = {}
for fn in ("vad_stats48.csv", "vad_stats48_heart_levo.csv"):
    for r in csv.DictReader(open(R / fn)):
        if r["src"] == "gen":
            kept[(SYS[(r["exp"], r["arm"])], r["song"], r["seed"])] = float(r["kept_sec"])
sep = {}
for fn in ("sep_redecode48_heart.csv", "sep_redecode48_levo.csv"):
    for r in csv.DictReader(open(R / fn)):
        sep[(SYS[(r["exp"], r["arm"])], r["song"], r["seed"])] = r

# WER / PER: same algorithm as analyze_wer_per.py (reference = ref_words; PER = wer on phoneme sequences, g2p_en without stress digits)
from g2p_en import G2p
g2p = G2p(); _c = {}


def ph(words):
    k = " ".join(words)
    if k not in _c:
        _c[k] = [re.sub(r"\d", "", p) for p in g2p(k) if p.strip() and p[0].isalpha()] if words else []
    return _c[k]


WP = {}
for k in rec:
    if k[0] in ("HeartMuLa", "LeVo"):
        ref = json.loads((META / f"{k[1]}.meta.json").read_text())["ref_words"]
        for vad in ("False", "True"):
            hyp = norm_words(rec[k][vad]["text"])
            WP[(k, vad)] = {"wer": wer(ref, hyp), "per": wer(ph(ref), ph(hyp))}


def metric(k, vad, m):
    return float(rec[k][vad]["recall"]) if m == "recall" else WP[(k, vad)][m]


songs = sorted({k[1] for k in rec})


def paired(a, b, m, vad):
    """a - b, cluster bootstrap over songs; returns mean, interval, Wilcoxon p and number of songs."""
    per_song = {}
    for s in songs:
        # only seeds present on both sides are used (ACE-Step vs HeartMuLa / LeVo = seeds 1-4)
        ds = [metric((a, s, sd), vad, m) - metric((b, s, sd), vad, m) for sd in SEEDS
              if (a, s, sd) in rec and (b, s, sd) in rec]
        if ds:
            per_song[s] = st.mean(ds)
    ks = sorted(per_song)
    mean = st.mean(per_song.values())
    rnd = random.Random(101)
    bs = sorted(st.mean(per_song[ks[rnd.randrange(len(ks))]] for _ in ks) for _ in range(20000))
    p = wilcoxon(list(per_song.values())).pvalue if wilcoxon else None
    return {"diff_mean": round(mean, 4), "ci95": [round(bs[500], 4), round(bs[19499], 4)], "wilcoxon_p_by_song": p, "n_songs": len(ks)}


def rank(x):
    o = sorted(range(len(x)), key=lambda j: x[j]); r = [0.0] * len(x); j = 0
    while j < len(o):
        k = j
        while k + 1 < len(o) and x[o[k + 1]] == x[o[j]]:
            k += 1
        for m in range(j, k + 1):
            r[o[m]] = (j + k) / 2
        j = k + 1
    return r


def spear(a, b):
    ra, rb = rank(a), rank(b); ma, mb = st.mean(ra), st.mean(rb)
    return sum((x - ma) * (y - mb) for x, y in zip(ra, rb)) / (sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb)) ** .5


out = {"per_system": {}, "compare": {}}
for name in SYS.values():
    ks = [k for k in rec if k[0] == name]
    off = [float(rec[k]["False"]["recall"]) for k in ks]; on = [float(rec[k]["True"]["recall"]) for k in ks]
    kp = [kept[k] for k in ks]
    v = {"n": len(ks), "recall_off_mean": round(st.mean(off), 4), "recall_on_mean": round(st.mean(on), 4),
         "recall_off_median": round(st.median(off), 4), "recall_on_median": round(st.median(on), 4),
         "broken_on": sum(a >= .6 and b < .3 for a, b in zip(off, on)), "worse_on": sum(b < a for a, b in zip(off, on)),
         "zero_on": sum(int(rec[k]["True"]["n_hyp"]) == 0 for k in ks), "below03_on": sum(b < .3 for b in on),
         "kept_sec_median": round(st.median(kp), 1), "kept_sec_mean": round(st.mean(kp), 1)}
    if name in ("HeartMuLa", "LeVo"):
        p2 = []; flagged = rescued = 0; allsep = []
        for k in ks:
            o = rec[k]["False"]; x = float(o["recall"])
            if int(o["n_hyp"]) / 120 < 0.5:
                flagged += 1
                if int(sep[k]["n_hyp"]) / 120 >= 0.5:
                    x = float(sep[k]["recall"]); rescued += 1
            p2.append(x); allsep.append(float(sep[k]["recall"]))
        v.update({"P0_mean": v["recall_on_mean"], "P1_mean": v["recall_off_mean"], "P2_mean": round(st.mean(p2), 4),
                  "all_vocals_mean": round(st.mean(allsep), 4), "flagged": flagged, "rescued": rescued,
                  "below03": {"P0": sum(b < .3 for b in on), "P1": sum(a < .3 for a in off), "P2": sum(x < .3 for x in p2),
                              "all_vocals": sum(x < .3 for x in allsep)},
                  **{f"{m}_{t}_mean": round(st.mean(WP[(k, vad)][m] for k in ks), 4)
                     for m in ("wer", "per") for t, vad in (("off", "False"), ("on", "True"))}})
    out["per_system"][name] = v

for a, b in (("LeVo", "HeartMuLa"), ("LeVo", "2B·B"), ("LeVo", "2B·C"), ("LeVo", "XL·B"), ("LeVo", "XL·C"),
             ("HeartMuLa", "2B·C"), ("HeartMuLa", "XL·B")):
    out["compare"][f"{a} − {b}"] = {f"recall_{t}": paired(a, b, "recall", vad) for t, vad in (("off", "False"), ("on", "True"))}
for t, vad in (("off", "False"), ("on", "True")):
    for m in ("wer", "per"):
        out["compare"]["LeVo − HeartMuLa"][f"{m}_{t}"] = paired("LeVo", "HeartMuLa", m, vad)

ks = [k for k in rec if k in kept]
out["spearman_kept_vs_recall_on_all"] = round(spear([kept[k] for k in ks], [float(rec[k]["True"]["recall"]) for k in ks]), 3)
out["spearman_kept_vs_recall_off_all"] = round(spear([kept[k] for k in ks], [float(rec[k]["False"]["recall"]) for k in ks]), 3)
out["n_items_all"] = len(ks)
# per-system correlations (within each system: kept seconds vs recall with VAD on / off), reported in the supplement
out["spearman_kept_by_system"] = {}
for n in SYS.values():
    kk = [k for k in ks if k[0] == n]
    out["spearman_kept_by_system"][n] = {"on": round(spear([kept[k] for k in kk], [float(rec[k]["True"]["recall"]) for k in kk]), 3),
                                         "off": round(spear([kept[k] for k in kk], [float(rec[k]["False"]["recall"]) for k in kk]), 3),
                                         "n": len(kk)}
names = list(SYS.values())
out["rank_by_recall_on"] = sorted(names, key=lambda n: -out["per_system"][n]["recall_on_mean"])
out["rank_by_recall_off"] = sorted(names, key=lambda n: -out["per_system"][n]["recall_off_mean"])
out["rank_by_kept_mean"] = sorted(names, key=lambda n: -out["per_system"][n]["kept_sec_mean"])
d = [kept[("LeVo", s, sd)] - kept[("HeartMuLa", s, sd)] for (n, s, sd) in rec if n == "LeVo"]
out["levo_minus_heart_kept"] = {"mean": round(st.mean(d), 1), "levo_more": sum(x > 0 for x in d), "n": len(d)}
(R / "cross_family_summary.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
print(json.dumps({k: v for k, v in out.items() if k != "per_system"}, indent=1, ensure_ascii=False))
