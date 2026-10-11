#!/usr/bin/env python3
"""Default-settings check: what users typically run with faster-whisper, plus same-machine P0/P1 for HeartMuLa/LeVo.

Conditions (one row per song per condition, "cond" column):
  batched_default  BatchedInferencePipeline(model).transcribe(a, language="en"), everything else default
                   (faster-whisper 1.2.1: vad_filter=True, VadOptions(min_silence_duration_ms=160,
                   max_speech_duration_s=30), beam_size=5, batch_size=8)
  plain_default    WhisperModel.transcribe(a, language="en"), everything else default (vad_filter=False,
                   beam_size=5) - differs from batched_default only by the batched entry point + VAD
  p0 / p1          the paper's main setting: beam_size=1, vad_filter=True / False
                   (run only for HeartMuLa and LeVo, so all families are transcribed on the same hardware)
Songs: ACE-Step 384 + HeartMuLa and LeVo 96 each; first 120 s, 16 kHz mono (as in exp_vad_gen.py).
Inputs:  GEN_ROOT/runs/<exp>/<arm>/<song>@s<seed>.wav, META_DIR/<song>.meta.json ("ref_words").
Outputs: RESULTS_DIR/batched48.csv (p0, p1; librosa decoding; not shipped, not used by the paper) and RESULTS_DIR/batched48_default.csv
         (batched_default, plain_default; PyAV decoding); resumable. The "text" column holds the transcript.
Supports: the "library defaults" result (the batched default pipeline also turns VAD on).
"""
import csv, json, time
from pathlib import Path
import numpy as np
import audio_io
from verify_metrics import norm_words, recall, wer

# ===== Edit here =====
REPO = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO / "results"                   # output CSVs go here
GEN_ROOT = REPO / "data" / "audio" / "generated"           # runs/<exp>/<arm>/<song>@s<seed>.wav, 120 s generated songs you produce yourself
META_DIR = REPO / "data" / "lyrics_meta"              # <song>.meta.json with "ref_words"
ACE = ["vad48_2b/B", "vad48_2b/C", "vad48_xl/B", "vad48_xl/C"]   # ACE-Step 2B / XL arms
FAM = ["vad48_heart/H", "vad48_levo/L"]                          # HeartMuLa, LeVo
# (output csv, conditions to run, audio decoding backend for audio_io) - the two runs reported
RUNS = [
    ("batched48.csv", ["p0", "p1"], "librosa"),
    ("batched48_default.csv", ["batched_default", "plain_default"], "av"),
]
# =====================

FIELDS = ["cond", "exp", "arm", "song", "seed", "recall", "wer", "n_ref", "n_hyp", "dur_after_vad", "sec", "text"]


def run_one(m, bp, out_name, want):
    todo = []
    for ea in ACE + FAM:
        exp, arm = ea.split("/")
        root = GEN_ROOT / "runs" / exp / arm
        for w in sorted(root.glob("*.wav")):
            s, sd = w.stem.split("@s")
            conds = ["batched_default", "plain_default"] + (["p0", "p1"] if ea in FAM else [])
            todo += [(c, exp, arm, s, sd, w) for c in conds if c in want]
    out = RESULTS_DIR / out_name; done = set()
    if out.exists():
        done = {(r["cond"], r["exp"], r["arm"], r["song"], r["seed"]) for r in csv.DictReader(open(out))}
    new = not out.exists()
    with open(out, "a", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            wr.writeheader()
        for k, (c, exp, arm, s, sd, w) in enumerate(todo, 1):
            if (c, exp, arm, s, sd) in done:
                continue
            ref = json.loads((META_DIR / f"{s}.meta.json").read_text())["ref_words"]
            au = np.asarray(audio_io.load_mono16k(str(w), duration=120.0), dtype=np.float32)
            t0 = time.time()
            if c == "batched_default":
                segs, info = bp.transcribe(au, language="en")
            elif c == "plain_default":
                segs, info = m.transcribe(au, language="en")
            else:
                segs, info = m.transcribe(au, language="en", beam_size=1, vad_filter=(c == "p0"))
            txt = " ".join(x.text for x in segs); hyp = norm_words(txt)
            wr.writerow({"cond": c, "exp": exp, "arm": arm, "song": s, "seed": sd, "recall": round(recall(ref, hyp), 4),
                         "wer": round(wer(ref, hyp), 4), "n_ref": len(ref), "n_hyp": len(hyp),
                         "dur_after_vad": round(getattr(info, "duration_after_vad", 0) or 0, 2), "sec": round(time.time() - t0, 2),
                         "text": txt.strip()[:4000]})
            f.flush()
            if k % 50 == 0:
                print(f"[{k}/{len(todo)}]", flush=True)
    print("wrote", out, flush=True)


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    from faster_whisper import WhisperModel, BatchedInferencePipeline
    m = WhisperModel("large-v3", device="cuda", compute_type="float16"); bp = BatchedInferencePipeline(m)
    for out_name, conds, backend in RUNS:
        audio_io.BACKEND = backend
        run_one(m, bp, out_name, set(conds))


if __name__ == "__main__":
    main()
