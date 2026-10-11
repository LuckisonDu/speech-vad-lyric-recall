#!/usr/bin/env python3
"""Lyric-recall / WER implementation used by all experiments, plus a parity check.

norm_words / recall / wer are imported by the other scripts in this folder.
main() is a parity check: it re-scores the stored transcripts ("text" column) of an
earlier evaluation's CSV with this implementation and checks that recall/WER match the
values stored in that CSV (difflib + jiwer reference implementation) value by value.
Inputs: EARLIER_CSV (columns song, cond, text, recall, wer, n_hyp) and per-song lyric
files META_DIR/<song>.meta.json with "ref_words". Writes nothing (prints a summary).
Supports the metric definitions in the paper's Setup section (recall = in-order matched
reference words / reference words; WER = word edit distance / reference words).
"""
import csv, difflib, json, re, sys
from pathlib import Path

# ===== Edit here =====
REPO = Path(__file__).resolve().parent.parent
EARLIER_CSV = Path("/path/to/earlier_eval/orig_asr.csv")   # earlier evaluation's CSV with stored transcripts and recall/wer
META_DIR = REPO / "data" / "lyrics_meta"                         # <song>.meta.json files, each with "ref_words" (normalized reference lyrics)
TRUNC_LEN = 1999                                            # transcripts stored with >= this many chars were truncated; skipped
TOL = 5e-5                                                  # tolerance for "identical" (values were stored rounded to 4 decimals)
# =====================

_BRACKETS = re.compile(r"\([^)]*\)|\[[^\]]*\]")


def norm_words(s):                     # lower-case, drop (...) and [...] spans, keep [a-z0-9'] words
    s = _BRACKETS.sub(" ", s.lower().replace("’", "'"))
    out = []
    for w in re.findall(r"[a-z0-9']+", s):
        w = w.strip("'")
        if w:
            out.append(w)
    return out


def recall(ref, hyp):                  # in-order matched words (difflib) / reference words
    if not ref:
        return float("nan")
    sm = difflib.SequenceMatcher(None, ref, hyp, autojunk=False)
    return sum(b.size for b in sm.get_matching_blocks()) / len(ref)


def wer(ref, hyp):
    """Word-level edit distance / number of reference words (jiwer.wer definition).
    An empty hypothesis is scored as the single word '<empty>', as in the reference implementation."""
    if not ref:
        return float("nan")
    h = hyp if hyp else ["<empty>"]
    prev = list(range(len(h) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i]
        for j, w in enumerate(h, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != w)))
        prev = cur
    return prev[-1] / len(ref)


def main():
    rows = list(csv.DictReader(open(EARLIER_CSV)))
    n_ok_r = n_ok_w = n_cmp = n_skip_trunc = 0
    worst_r = worst_w = 0.0
    bad = []
    for r in rows:
        mp = META_DIR / f"{r['song']}.meta.json"
        if not mp.exists():
            continue
        txt = r["text"]
        if len(txt) >= TRUNC_LEN:            # truncated when stored; cannot be re-scored
            n_skip_trunc += 1
            continue
        ref = json.loads(mp.read_text())["ref_words"]
        hyp = norm_words(txt)
        n_cmp += 1
        dr = abs(recall(ref, hyp) - float(r["recall"]))
        dw = abs(wer(ref, hyp) - float(r["wer"]))
        worst_r = max(worst_r, dr); worst_w = max(worst_w, dw)
        if dr <= TOL: n_ok_r += 1
        if dw <= TOL: n_ok_w += 1
        if dr > TOL or dw > TOL:
            bad.append((r["song"], r["cond"], round(dr, 5), round(dw, 5), len(ref), len(hyp), int(r["n_hyp"])))
    print(f"compared {n_cmp} rows (skipped {n_skip_trunc} rows whose stored transcript was truncated)")
    print(f"  recall identical {n_ok_r}/{n_cmp} = {n_ok_r/max(n_cmp,1):.1%}, max abs diff {worst_r:.2e}")
    print(f"  WER    identical {n_ok_w}/{n_cmp} = {n_ok_w/max(n_cmp,1):.1%}, max abs diff {worst_w:.2e}")
    if bad:
        print(f"  {len(bad)} mismatching rows, first 8:")
        for b in bad[:8]:
            print(f"     {b[0]:<24} {b[1]} d_recall={b[2]} d_wer={b[3]} n_ref={b[4]} n_hyp_recomputed={b[5]} stored={b[6]}")
    ok = n_cmp > 0 and n_ok_r == n_cmp and n_ok_w == n_cmp
    print("\n" + ("PASS: implementation matches" if ok else "FAIL: implementation does not match"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
