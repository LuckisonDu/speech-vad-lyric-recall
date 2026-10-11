#!/usr/bin/env python3
"""Alternative metrics: WER and PER (in the surveyed literature 17/27 papers report PER and 7 report WER).
Checks whether the VAD-induced ranking reversal also holds under these two metrics.
No re-recognition: the metrics are recomputed from the transcripts stored in the per-item CSVs.

Reads:  ../results/vad_gen48.csv, ../results/vad_gen48_heart.csv, ../results/vad_gen48_levo.csv
        (these must be the FULL per-item CSVs with the "text" transcript column, as produced by the
        experiment scripts; the public CSVs omit that column, so rerun the experiments first),
        plus one "<song>.meta.json" per song (key "ref_words": normalised reference lyric words of the window).
Writes: ../results/wer_per_summary.json
Supports: WER/PER table and the XL·B-2B·B WER/PER differences with VAD off/on.

WER = word-level edit distance / number of reference words (verify_metrics.wer);
PER = phoneme-level edit distance / number of reference phonemes (g2p_en, stress digits removed).
Both are lower-is-better.
"""
import csv, json, random, re, statistics as st
from collections import defaultdict
from pathlib import Path
from verify_metrics import norm_words, wer
from g2p_en import G2p

# ===== Edit here =====
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"   # folder holding the result CSV/JSON files
META = Path("data/lyrics_meta")   # folder with "<song>.meta.json" files, each containing "ref_words" (reference lyric words of the 120 s window)
# =====================

R = RESULTS_DIR
SYS = {("vad48_2b", "B"): "2B·B", ("vad48_2b", "C"): "2B·C", ("vad48_xl", "B"): "XL·B", ("vad48_xl", "C"): "XL·C",
       ("vad48_heart", "H"): "HeartMuLa", ("vad48_levo", "L"): "LeVo"}
g2p = G2p(); _c = {}


def ph(words):
    k = " ".join(words)
    if k not in _c:
        _c[k] = [re.sub(r"\d", "", p) for p in g2p(k) if p.strip() and p[0].isalpha()] if words else []
    return _c[k]


rows = defaultdict(dict)
for fn in ("vad_gen48.csv", "vad_gen48_heart.csv", "vad_gen48_levo.csv"):
    if not (R / fn).exists():
        continue
    for r in csv.DictReader(open(R / fn)):
        if r["src"] != "gen":
            continue
        ref = json.loads((META / f"{r['song']}.meta.json").read_text())["ref_words"]
        hyp = norm_words(r["text"])
        rp, hp = ph(ref), ph(hyp)
        rows[(SYS[(r["exp"], r["arm"])], r["song"], r["seed"])][r["vad"]] = {
            "recall": float(r["recall"]), "wer": wer(ref, hyp), "per": wer(rp, hp)}


def boot(by, n=20000, seed=101):
    songs = sorted(by); rnd = random.Random(seed); out = []
    for _ in range(n):
        tot = cnt = 0
        for _ in songs:
            x = by[songs[rnd.randrange(len(songs))]]; tot += sum(x); cnt += len(x)
        out.append(tot / cnt)
    out.sort(); return round(out[int(.025 * n)], 4), round(out[int(.975 * n)], 4)


names = [n for n in ("2B·B", "2B·C", "XL·B", "XL·C", "HeartMuLa", "LeVo") if any(k[0] == n for k in rows)]
summ = {"per_system": {}, "ranking": {}, "compare": {}}
for n in names:
    ks = [k for k in rows if k[0] == n and len(rows[k]) == 2]
    summ["per_system"][n] = {"n": len(ks), **{f"{m}_{tag}": round(st.mean(rows[k][v][m] for k in ks), 4)
                                               for m in ("recall", "wer", "per") for v, tag in (("False", "off"), ("True", "on"))}}
for m in ("wer", "per"):
    for tag in ("off", "on"):
        summ["ranking"][f"{m}_{tag}"] = sorted(((n, summ["per_system"][n][f"{m}_{tag}"]) for n in names), key=lambda x: x[1])
for a, b in (("2B·B", "XL·B"), ("2B·B", "XL·C"), ("2B·C", "XL·B"), ("2B·C", "XL·C")):
    res = {}
    for m in ("wer", "per"):
        for v, tag in (("False", "off"), ("True", "on")):
            by = defaultdict(list)
            for k in rows:
                if k[0] == a and (b, k[1], k[2]) in rows:
                    by[k[1]].append(rows[(b, k[1], k[2])][v][m] - rows[k][v][m])
            d = [x for xs in by.values() for x in xs]
            res[f"{m}_{tag}"] = {"diff_mean": round(st.mean(d), 4), "ci95": boot(by)}
    summ["compare"][f"{b} − {a}"] = res
(R / "wer_per_summary.json").write_text(json.dumps(summ, indent=1, ensure_ascii=False))
print("system      n   recall off->on   WER off->on     PER off->on   (WER / PER lower is better)")
for n, v in summ["per_system"].items():
    print(f"{n:9s} {v['n']:3d}  {v['recall_off']:.3f}→{v['recall_on']:.3f}   {v['wer_off']:.3f}→{v['wer_on']:.3f}   {v['per_off']:.3f}→{v['per_on']:.3f}")
for k, v in summ["ranking"].items():
    print("ranking (best->worst)", k, [x[0] for x in v])
print("system comparisons (later - earlier; negative = later is better)")
for k, v in summ["compare"].items():
    print(f"{k}: WER off {v['wer_off']['diff_mean']:+.3f} {v['wer_off']['ci95']} on {v['wer_on']['diff_mean']:+.3f} {v['wer_on']['ci95']} | "
          f"PER off {v['per_off']['diff_mean']:+.3f} {v['per_off']['ci95']} on {v['per_on']['diff_mean']:+.3f} {v['per_on']['ci95']}")
