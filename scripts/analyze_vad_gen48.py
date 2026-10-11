#!/usr/bin/env python3
"""Main experiment analysis: how switching the speech VAD on/off changes recall scores and system
rankings on generated songs (48 songs x {2B, XL} x {B, C} x seeds 1, 2, plus the original recordings).

Reads:  ../results/vad_gen48.csv (output of exp_vad_gen.py)
Writes: ../results/vad_gen48_summary.json
Supports: main results table (per-system recall with VAD off/on, broken items, zero-word items,
word-keep ratio) and the pairwise system comparisons / ranking reversal in the main text.

Definitions:
  keep ratio = recognised words with VAD on / recognised words with VAD off;
  "broken"   = recall < 0.3 on one side while >= 0.6 on the other side.
System comparisons are paired on the same song and seed. CI = paired bootstrap resampling songs
(N=20000, seed=101, 95%). Test = Wilcoxon signed-rank on per-song means over the two seeds
(only if scipy is available; otherwise only the interval is reported).
"""
import csv, json, random, statistics as st
from collections import defaultdict
from pathlib import Path

# ===== Edit here =====
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"   # folder holding the result CSV/JSON files
NAME = "vad_gen48"   # base name of the input CSV (without .csv); the summary is written as <NAME>_summary.json
# =====================

R = RESULTS_DIR
rows = list(csv.DictReader(open(R / f"{NAME}.csv")))
SYS_NAME = {("vad48_2b", "B"): "2B·B", ("vad48_2b", "C"): "2B·C", ("vad48_xl", "B"): "XL·B",
            ("vad48_xl", "C"): "XL·C", ("orig", "orig"): "orig"}

rec = defaultdict(dict)        # rec[system][(song, seed)] = {"False": row, "True": row}
for r in rows:
    s = SYS_NAME.get((r["exp"], r["arm"]), f"{r['exp']}/{r['arm']}")
    rec[s].setdefault((r["song"], r["seed"]), {})[r["vad"]] = r
for s in rec:                  # keep only items that have both VAD settings
    rec[s] = {k: v for k, v in rec[s].items() if "False" in v and "True" in v}

try:
    from scipy.stats import wilcoxon
except Exception:
    wilcoxon = None


def per_system(d):
    off = [float(v["False"]["recall"]) for v in d.values()]
    on = [float(v["True"]["recall"]) for v in d.values()]
    keep = [int(v["True"]["n_hyp"]) / max(1, int(v["False"]["n_hyp"])) for v in d.values()]
    wps_on = [int(v["True"]["n_hyp"]) / 120.0 for v in d.values()]
    return {"n": len(d),
            "recall_off_mean": round(st.mean(off), 4), "recall_on_mean": round(st.mean(on), 4),
            "recall_off_median": round(st.median(off), 4), "recall_on_median": round(st.median(on), 4),
            "keep_median": round(st.median(keep), 4),
            "broken_by_vad_on": sum(a >= 0.6 and b < 0.3 for a, b in zip(off, on)),
            "broken_by_vad_off": sum(b >= 0.6 and a < 0.3 for a, b in zip(off, on)),
            "zero_words_on": sum(int(v["True"]["n_hyp"]) == 0 for v in d.values()),
            "worse_with_vad": sum(b < a for a, b in zip(off, on)),
            "cr_gt_2.4_off": sum(float(v["False"]["cr_max"]) > 2.4 for v in d.values()),
            "wps_lt_0.5_on": sum(w < 0.5 for w in wps_on),
            "wps_lt_0.5_off": sum(int(v["False"]["n_hyp"]) / 120.0 < 0.5 for v in d.values())}


def boot_ci(by_song, n=20000, seed=101):
    """by_song: {song: [paired differences, ...]}; resample songs, statistic = mean of all paired differences."""
    songs = sorted(by_song)
    rnd = random.Random(seed)
    stat = []
    for _ in range(n):
        tot = cnt = 0
        for _ in songs:
            x = by_song[songs[rnd.randrange(len(songs))]]
            tot += sum(x); cnt += len(x)
        stat.append(tot / cnt)
    stat.sort()
    return round(stat[int(0.025 * n)], 4), round(stat[int(0.975 * n)], 4)


def compare(a, b):
    """b - a, paired on song and seed; computed with VAD off and on, plus the interaction (on - off)."""
    keys = sorted(set(rec[a]) & set(rec[b]))
    out = {"pairs": len(keys), "songs": len({k[0] for k in keys})}
    diffs = {}
    for vad, tag in (("False", "vad_off"), ("True", "vad_on")):
        d = {k: float(rec[b][k][vad]["recall"]) - float(rec[a][k][vad]["recall"]) for k in keys}
        diffs[tag] = d
        by_song = defaultdict(list)
        for k, v in d.items():
            by_song[k[0]].append(v)
        res = {"mean_a": round(st.mean(float(rec[a][k][vad]["recall"]) for k in keys), 4),
               "mean_b": round(st.mean(float(rec[b][k][vad]["recall"]) for k in keys), 4),
               "diff_mean": round(st.mean(d.values()), 4), "ci95": boot_ci(by_song),
               "b_wins": sum(v > 0 for v in d.values()), "a_wins": sum(v < 0 for v in d.values())}
        if wilcoxon:
            sm = [st.mean(v) for v in by_song.values()]
            res["wilcoxon_p_by_song"] = float(f"{wilcoxon(sm).pvalue:.3g}") if any(sm) else 1.0
        out[tag] = res
    dd = {k: diffs["vad_on"][k] - diffs["vad_off"][k] for k in keys}
    by_song = defaultdict(list)
    for k, v in dd.items():
        by_song[k[0]].append(v)
    out["interaction_on_minus_off"] = {"diff_mean": round(st.mean(dd.values()), 4), "ci95": boot_ci(by_song)}
    # per item: does the winner with VAD off change when VAD is turned on
    flip = sum((diffs["vad_off"][k] > 0) != (diffs["vad_on"][k] > 0) for k in keys
               if diffs["vad_off"][k] != 0 and diffs["vad_on"][k] != 0)
    out["winner_flips"] = flip
    # failure counts: items broken by turning VAD on (paired)
    def broken(s, k):
        return float(rec[s][k]["False"]["recall"]) >= 0.6 and float(rec[s][k]["True"]["recall"]) < 0.3
    out["broken_a"] = sum(broken(a, k) for k in keys)
    out["broken_b"] = sum(broken(b, k) for k in keys)
    out["broken_only_a"] = sum(broken(a, k) and not broken(b, k) for k in keys)
    out["broken_only_b"] = sum(broken(b, k) and not broken(a, k) for k in keys)
    return out


summary = {"per_system": {s: per_system(d) for s, d in rec.items()}, "compare": {}}
gen = [s for s in ("2B·B", "2B·C", "XL·B", "XL·C") if s in rec]
for i, a in enumerate(gen):
    for b in gen[i + 1:]:
        summary["compare"][f"{b} − {a}"] = compare(a, b)
for vad, tag in (("False", "vad_off"), ("True", "vad_on")):
    summary[f"ranking_{tag}"] = sorted(((s, round(st.mean(float(v[vad]["recall"]) for v in rec[s].values()), 4))
                                        for s in gen), key=lambda x: -x[1])
(R / f"{NAME}_summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))

print("== per system (recall mean off->on | median off->on | keep median | broken-on / broken-off / zero words / worse with VAD | repetition flag (off) | <0.5 words/s off/on)")
for s, v in summary["per_system"].items():
    print(f"{s:6s} n={v['n']:3d}  {v['recall_off_mean']:.3f}→{v['recall_on_mean']:.3f} | "
          f"{v['recall_off_median']:.3f}→{v['recall_on_median']:.3f} | {v['keep_median']:.3f} | "
          f"{v['broken_by_vad_on']}/{v['broken_by_vad_off']}/{v['zero_words_on']}/{v['worse_with_vad']} | "
          f"{v['cr_gt_2.4_off']} | {v['wps_lt_0.5_off']}/{v['wps_lt_0.5_on']}")
print("== ranking  off:", summary["ranking_vad_off"], " on:", summary["ranking_vad_on"])
print("== system comparisons (later - earlier, same song and seed)")
for name, c in summary["compare"].items():
    o, n, x = c["vad_off"], c["vad_on"], c["interaction_on_minus_off"]
    print(f"{name}: pairs {c['pairs']}  off {o['diff_mean']:+.3f} {o['ci95']} wins {o['b_wins']}/losses {o['a_wins']} p={o.get('wilcoxon_p_by_song')} | "
          f"on {n['diff_mean']:+.3f} {n['ci95']} wins {n['b_wins']}/losses {n['a_wins']} p={n.get('wilcoxon_p_by_song')} | "
          f"interaction {x['diff_mean']:+.3f} {x['ci95']} | winner flips {c['winner_flips']} | "
          f"broken-on first {c['broken_a']} second {c['broken_b']} (only first {c['broken_only_a']} only second {c['broken_only_b']})")
