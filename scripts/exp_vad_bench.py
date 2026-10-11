#!/usr/bin/env python3
"""Public-benchmark replication of the VAD on/off effect: Jam-ALT (20 English songs) + MUSDB-ALT (39 songs).

Whole songs (no 120 s crop), original mixtures. Same setting as the main experiment:
faster-whisper large-v3, fp16, beam_size=1, language="en", only vad_filter flipped.
Reference lyrics = the "text" field of each benchmark's meta.jsonl; metrics from verify_metrics.py
(words inside (...) and [...] removed, as elsewhere in the paper).
Inputs:  BENCH_DIR/{jamalt,musdbalt}/meta.jsonl and BENCH_DIR/<bench>/audio/<id>.*
Outputs: RESULTS_DIR/vad_bench.csv (PyAV decoding) and RESULTS_DIR/vad_bench_rerun.csv
         (librosa decoding, independent rerun); both resumable. The "text" column holds the transcript.
Supports: the public-benchmark replication (so readers can check the effect on public data).
"""
import csv, json, time
from pathlib import Path
import numpy as np
import audio_io
from verify_metrics import norm_words, recall, wer

# ===== Edit here =====
REPO = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO / "results"                   # output CSVs go here
BENCH_DIR = Path("/path/to/benchmarks")          # contains jamalt/ and musdbalt/, each with meta.jsonl and audio/<id>.<ext>
BENCHES = ("jamalt", "musdbalt")
# (output csv, audio decoding backend for audio_io: "av" or "librosa") - the two runs reported
RUNS = [
    ("vad_bench.csv", "av"),
    ("vad_bench_rerun.csv", "librosa"),
]
# =====================

FIELDS = ["bench", "id", "name", "vad", "dur", "recall", "wer", "n_ref", "n_hyp", "n_seg", "cr_max", "sec", "text"]


def run_one(model, items, out_name):
    out_csv = RESULTS_DIR / out_name
    done = set()
    if out_csv.exists():
        done = {(r["id"], r["vad"]) for r in csv.DictReader(open(out_csv))}
    new = not out_csv.exists()
    with open(out_csv, "a", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            wr.writeheader()
        for k, (b, m, au) in enumerate(items, 1):
            ref = norm_words(m["text"])
            audio = None
            for vad in (False, True):
                if (m["id"], str(vad)) in done:
                    continue
                if audio is None:
                    audio = audio_io.load_mono16k(au)
                t0 = time.time()
                segs, _ = model.transcribe(audio, language="en", beam_size=1, vad_filter=vad)
                segs = list(segs)
                txt = " ".join(x.text for x in segs)
                hyp = norm_words(txt)
                wr.writerow({"bench": b, "id": m["id"], "name": m.get("name", ""), "vad": vad,
                             "dur": round(len(audio) / 16000.0, 1), "recall": round(recall(ref, hyp), 4),
                             "wer": round(wer(ref, hyp), 4), "n_ref": len(ref), "n_hyp": len(hyp), "n_seg": len(segs),
                             "cr_max": round(max((x.compression_ratio for x in segs), default=0.0), 3),
                             "sec": round(time.time() - t0, 2), "text": txt.strip()[:8000]})
                f.flush()
            print(f"[{k}/{len(items)}] {b} {m['id']}", flush=True)
    print(f"wrote {out_csv}")


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    from faster_whisper import WhisperModel
    items = []
    for b in BENCHES:
        for line in open(BENCH_DIR / b / "meta.jsonl", encoding="utf-8"):
            m = json.loads(line)
            if m.get("language", "en") != "en":
                continue
            au = next(iter(sorted((BENCH_DIR / b / "audio").glob(m["id"] + ".*"))), None)
            if au is not None:
                items.append((b, m, au))
    print(f"{len(items)} songs", flush=True)
    model = WhisperModel("large-v3", device="cuda", compute_type="float16")
    for out_name, backend in RUNS:
        audio_io.BACKEND = backend
        run_one(model, items, out_name)


if __name__ == "__main__":
    main()
