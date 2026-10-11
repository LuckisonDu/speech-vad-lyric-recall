#!/usr/bin/env python3
"""Summary of exp_demo_vad.py: on the English demo songs published by the model authors, recall / WER
and rankings of each system under three recognition conditions (Silero VAD off, VAD on, energy gate),
compared with the PER rankings published in the papers; near-tied system pairs get a per-song paired
comparison (win/loss counts + song-level bootstrap).

Reads  : ../results/demo_vad.csv
Writes : ../results/demo_vad_summary.json (and prints a table)
Supports the paper's author-demo-song check.
"""
import csv, itertools, json, random, statistics as st
from pathlib import Path

# ===== Edit here =====
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"   # folder holding the result CSV/JSON files
N_BOOT = 10000      # bootstrap resamples
BOOT_SEED = 0       # bootstrap RNG seed
# =====================

R = RESULTS_DIR
PAPER_PER = {  # lower is better; sources listed in the header of exp_demo_vad.py
    ("levo", "text"): {"suno": 21.6, "mureka": 7.2, "yue": 36.4, "diffrhythm": 12.3, "acestep": 37.1, "overview": 7.2},
    ("dr2", "text"): {"diffrhythmplus": 0.15, "acestep": 0.23, "levo": 0.19, "diffrhythm2": 0.13},
    ("dr2", "audio"): {"diffrhythmplus": 0.15, "levo": 0.19, "diffrhythm2": 0.13},   # published PER is for text prompts; reference only
}
NAMES = {"overview": "LeVo", "levo": "LeVo", "suno": "Suno", "mureka": "Mureka", "yue": "YuE", "diffrhythm": "DiffRhythm",
         "acestep": "ACE-Step", "diffrhythmplus": "DiffRhythm+", "diffrhythm2": "DiffRhythm2"}
KEY_PAIRS = {("levo", "text"): [("overview", "mureka"), ("yue", "acestep"), ("diffrhythm", "suno")],
             ("dr2", "text"): [("diffrhythm2", "diffrhythmplus"), ("diffrhythmplus", "levo"), ("levo", "acestep")],
             ("dr2", "audio"): [("diffrhythm2", "diffrhythmplus"), ("diffrhythmplus", "levo")]}
CONDS = ["off", "on", "energy"]


def kendall(a, b):
    """Kendall tau-b between two rankings (list of systems best->worst vs. rank dict; ties share a rank)."""
    ra = {s: i for i, s in enumerate(a)}
    xs = list(a)
    c = d = 0
    for x, y in itertools.combinations(xs, 2):
        s1 = ra[x] - ra[y]; s2 = b[x] - b[y]
        if s1 * s2 > 0: c += 1
        elif s1 * s2 < 0: d += 1
    n = len(xs) * (len(xs) - 1) / 2
    ties_b = sum(1 for x, y in itertools.combinations(xs, 2) if b[x] == b[y])
    return round((c - d) / ((n * (n - ties_b)) ** 0.5), 3) if n - ties_b > 0 else None


def boot(diffs, n=N_BOOT, seed=BOOT_SEED):
    rnd = random.Random(seed); k = len(diffs)
    ms = sorted(sum(rnd.choice(diffs) for _ in range(k)) / k for _ in range(n))
    return [round(ms[int(0.025 * n)], 4), round(ms[int(0.975 * n) - 1], 4)], round(sum(m > 0 for m in ms) / n, 3)


def main():
    rows = list(csv.DictReader(open(R / "demo_vad.csv")))
    V = {}
    for r in rows:
        V[(r["demo"], r["subset"], r["system"], r["id"], r["cond"])] = r
    out = {"n_rows": len(rows), "groups": {}}
    for (demo, subset), per in PAPER_PER.items():
        systems = sorted({k[2] for k in V if k[:2] == (demo, subset)})
        songs = sorted({k[3] for k in V if k[:2] == (demo, subset)})
        if not systems:
            continue
        g = {"n_songs": len(songs), "songs": songs, "systems": {}, "rank": {}, "pairs": {}}
        for s in systems:
            e = {"name": NAMES[s], "paper_per": per.get(s)}
            for c in CONDS:
                rr = [V[(demo, subset, s, u, c)] for u in songs if (demo, subset, s, u, c) in V]
                if not rr:
                    continue
                e[f"n_{c}"] = len(rr)
                e[f"recall_{c}"] = round(st.mean(float(x["recall"]) for x in rr), 4)
                e[f"wer_{c}"] = round(st.mean(float(x["wer"]) for x in rr), 4)
                e[f"median_kept_{c}"] = round(st.median(float(x["kept_sec"]) for x in rr), 1)
            rr = [V[(demo, subset, s, u, "off")] for u in songs if (demo, subset, s, u, "off") in V]
            e["median_dur"] = round(st.median(float(x["dur"]) for x in rr), 1) if rr else None
            ron = [V[(demo, subset, s, u, "on")] for u in songs if (demo, subset, s, u, "on") in V]
            if ron:
                e["median_silero_kept_frac"] = round(st.median(float(x["kept_sec"]) / float(x["dur"]) for x in ron), 3)
            g["systems"][s] = e
        paper_rank = {s: sorted(set(per.values())).index(per[s]) for s in systems if s in per}
        g["rank"]["paper_PER"] = [NAMES[s] for s in sorted(systems, key=lambda s: per[s])]
        for c in CONDS:
            for m in ("recall", "wer"):
                k = f"{m}_{c}"
                if not all(k in g["systems"][s] for s in systems):
                    continue
                order = sorted(systems, key=lambda s: (-g["systems"][s][k] if m == "recall" else g["systems"][s][k]))
                g["rank"][k] = [NAMES[s] for s in order]
                g["rank"][f"kendall_vs_paper_{k}"] = kendall(order, paper_rank)
        for a, b in KEY_PAIRS[(demo, subset)]:
            if a not in systems or b not in systems:
                continue
            p = {}
            for c in CONDS:
                for m in ("recall", "wer"):
                    d = [float(V[(demo, subset, a, u, c)][m]) - float(V[(demo, subset, b, u, c)][m])
                         for u in songs if (demo, subset, a, u, c) in V and (demo, subset, b, u, c) in V]
                    if not d:
                        continue
                    ci, pg = boot(d)
                    p[f"{m}_{c}"] = {"n": len(d), "mean_diff": round(st.mean(d), 4), "a_better": sum((x > 0) if m == "recall" else (x < 0) for x in d),
                                     "b_better": sum((x < 0) if m == "recall" else (x > 0) for x in d), "boot95": ci, "boot_P_diff_gt0": pg}
            # how much the VAD changes this pair's gap (per-song on-minus-off difference of differences)
            for m in ("recall", "wer"):
                dd = []
                for u in songs:
                    try:
                        dd.append((float(V[(demo, subset, a, u, "on")][m]) - float(V[(demo, subset, b, u, "on")][m]))
                                  - (float(V[(demo, subset, a, u, "off")][m]) - float(V[(demo, subset, b, u, "off")][m])))
                    except KeyError:
                        pass
                if dd:
                    ci, pg = boot(dd)
                    p[f"{m}_on_minus_off_gap"] = {"n": len(dd), "mean": round(st.mean(dd), 4), "boot95": ci}
            g["pairs"][f"{NAMES[a]} − {NAMES[b]}"] = p
        out["groups"][f"{demo}/{subset}"] = g
    (R / "demo_vad_summary.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    for gk, g in out["groups"].items():
        print(f"\n== {gk}  ({g['n_songs']} songs)")
        print(f"{'system':<12}{'PER':>6} | " + " | ".join(f"rec_{c:<6} wer_{c:<6}" for c in CONDS) + " | dur silero_kept energy_kept")
        for s, e in g["systems"].items():
            print(f"{e['name']:<12}{str(e['paper_per']):>6} | " + " | ".join(f"{e.get('recall_'+c,float('nan')):.3f}      {e.get('wer_'+c,float('nan')):.3f}     " for c in CONDS)
                  + f" | {e['median_dur']} {e.get('median_kept_on')} {e.get('median_kept_energy')}")
        for k, v in g["rank"].items():
            print(f"  {k}: {v}")
        for pk, p in g["pairs"].items():
            print(f"  {pk}: " + "; ".join(f"{k} d={v.get('mean_diff', v.get('mean'))} " + (f"win{v['a_better']} loss{v['b_better']} " if 'a_better' in v else "") + f"CI{v['boot95']}" for k, v in p.items() if k.startswith("recall")))


if __name__ == "__main__":
    main()
