#!/usr/bin/env python3
"""De-duplicate vad_gen48.csv. Two recognition processes once appended to the same table at the same time,
so some keys (exp, arm, song, seed, vad) appear twice. Rule: keep the row that appears first in the file;
the statistics of how often the two rows disagree are written to a report (faster-whisper on GPU is not bit-reproducible
across re-runs). The original table is kept as BACKUP."""
import csv, collections, json, shutil, statistics as st
from pathlib import Path

# ===== Edit here =====
R = Path(__file__).resolve().parent.parent / "results"
CSV_PATH = R / "vad_gen48.csv"                         # table to de-duplicate (rewritten in place)
BACKUP = R / "vad_gen48_with_dups.csv"                 # copy of the table before de-duplication
REPORT = R / "vad_gen48_rerun_nondeterminism.json"     # disagreement statistics of the duplicated transcriptions
# =====================
shutil.copy2(CSV_PATH, BACKUP)
rows = list(csv.DictReader(open(CSV_PATH)))
fields = list(rows[0].keys())
seen, keep, pairs = {}, [], []
for r in rows:
    k = (r["exp"], r["arm"], r["song"], r["seed"], r["vad"])
    if k in seen:
        pairs.append((seen[k], r))
        continue
    seen[k] = r
    keep.append(r)
d = [abs(float(a["recall"]) - float(b["recall"])) for a, b in pairs]
rep = {"rows_before": len(rows), "rows_after": len(keep), "dup_pairs": len(pairs),
       "pairs_recall_differs": sum(x > 0 for x in d), "pairs_recall_diff_gt_0.05": sum(x > 0.05 for x in d),
       "mean_abs_recall_diff": round(st.mean(d), 4) if d else 0, "max_abs_recall_diff": max(d, default=0),
       "differs_by_vad": dict(collections.Counter(a["vad"] for (a, b), x in zip(pairs, d) if x > 0)),
       "rule": "keep first-written row per key"}
with open(CSV_PATH, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(keep)
json.dump(rep, open(REPORT, "w"), indent=1)
print(json.dumps(rep, ensure_ascii=False))
