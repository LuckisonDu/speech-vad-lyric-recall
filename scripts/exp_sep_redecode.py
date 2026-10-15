#!/usr/bin/env python3
"""Separate the vocals of every generated song and transcribe the separated vocals (VAD off or on).

Why: with VAD off, some generated songs still decode to non-lyrics ("Music", "Thank you.") and a
fallback to VAD-on recovers almost none of them; removing the accompaniment was the candidate fix.
All songs are processed (not only the flagged ones) so that we can also see whether normal songs get worse.
Separator: torchaudio HDEMUCS_HIGH_MUSDB_PLUS, 20 s chunks with 2 s left context (sep_util.py).
ASR: faster-whisper large-v3, fp16, beam 1, language=en (same as exp_vad_gen.py);
     VAD off  -> input "vocals_novad";  VAD on (Silero default 0.5) -> input "vocals_vad" (+ dur_after_vad).
Flagged songs (VAD off, < 0.5 recognised words per second in ../results/vad_gen48.csv) are processed first;
this only changes the processing order, not the results.

Reads  : GEN_ROOT/runs/<exp>/<arm>/<song>@s<seed>.wav (first 120 s), META_DIR/<song>.meta.json ("ref_words"),
         ../results/vad_gen48.csv (optional, only for ordering)
Writes : one CSV per entry of RUNS in ../results (sep_redecode48.csv, sep_vad48.csv, sep_vad48_fam.csv,
         sep_vad48_dr.csv, sep_redecode48_heart.csv, sep_redecode48_levo.csv); columns include the transcript ("text").
         Separated vocals are cached as GEN_ROOT/sep_vocals/<exp>/<arm>/<song>@s<seed>.vocals.wav
         (16 kHz mono); exp_energy_gate.py and exp_sep2.py read them.
Supports the paper's separated-vocals mechanism check (separation + VAD) and the cross-family rows.
Resumable: rows already in the output CSV are skipped.
"""
import csv, json, time
from pathlib import Path

from verify_metrics import norm_words, recall, wer

# ===== Edit here =====
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"   # output folder (also holds vad_gen48.csv)
GEN_ROOT = Path("data/audio/generated")   # contains runs/<exp>/<arm>/<song>@s<seed>.wav (generated songs you produce yourself)
SEP_DIR = GEN_ROOT / "sep_vocals"         # cache of separated vocals (written here, must be writable)
META_DIR = Path("data/lyrics_meta")       # "<song>.meta.json" files, each with "ref_words"
WIN_SEC = 120.0                           # only the first 120 s of every song is used
ACE = ["vad48_2b/B", "vad48_2b/C", "vad48_xl/B", "vad48_xl/C"]   # 4 ACE-Step systems
# (output csv, list of <exp>/<arm>, VAD on the separated vocals?) - the runs used for the paper
RUNS = [
    ("sep_redecode48.csv", ACE, False),
    ("sep_vad48.csv", ACE, True),
    ("sep_vad48_fam.csv", ["vad48_heart/H", "vad48_levo/L"], True),
    ("sep_vad48_dr.csv", ["vad48_diffrhythm/D"], True),
    ("sep_redecode48_heart.csv", ["vad48_heart/H"], False),
    ("sep_redecode48_levo.csv", ["vad48_levo/L"], False),
]
SEP_DEVICE = "cuda"       # where the separator runs ("cuda" or "cpu")
FLAGGED_ONLY = False      # True: only songs with < 0.5 words/s when VAD is off
SEP_ONLY = False          # True: only separate and cache vocals, no ASR (does not load Whisper)
# =====================

FIELDS = ["exp", "arm", "song", "seed", "input", "recall", "wer", "n_ref", "n_hyp", "n_seg", "cr_max", "sec", "text", "dur_after_vad"]


def main():
    import numpy as np, torch, torchaudio
    import audio_io
    from faster_whisper import WhisperModel

    import sep_util
    sep, sr_m, vi = sep_util.load_hdemucs(SEP_DEVICE)

    def separate(path, vp):
        """Return 16 kHz mono vocals (float32); read the cached file if present (sample-aligned chunking, see sep_util.py)."""
        return sep_util.separate_cached(path, vp, sep, sr_m, vi, SEP_DEVICE, WIN_SEC)

    model = None
    if not SEP_ONLY:
        model = WhisperModel("large-v3", device="cuda", compute_type="float16")
    print("models ready", flush=True)

    flagged = set()
    vg = RESULTS_DIR / "vad_gen48.csv"
    if vg.exists():
        flagged = {(r["exp"], r["arm"], r["song"], r["seed"]) for r in csv.DictReader(open(vg))
                   if r["src"] == "gen" and r["vad"] == "False" and int(r["n_hyp"]) / WIN_SEC < 0.5}

    for out_name, arms, use_vad in RUNS:
        todo = []
        for ea in arms:
            exp, arm = ea.split("/")
            for w in sorted((GEN_ROOT / "runs" / exp / arm).glob("*.wav")):
                song, seed = w.stem.split("@s")
                todo.append((exp, arm, song, seed, w))
        todo.sort(key=lambda x: (x[:4] not in flagged,))          # stable sort: flagged first
        if FLAGGED_ONLY:
            todo = [x for x in todo if x[:4] in flagged]
        print(f"{out_name}: {sum(x[:4] in flagged for x in todo)} flagged first, {len(todo)} total", flush=True)
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        out_csv = RESULTS_DIR / out_name
        done = set()
        if out_csv.exists():
            done = {(r["exp"], r["arm"], r["song"], r["seed"]) for r in csv.DictReader(open(out_csv))}
        new = not out_csv.exists()
        with open(out_csv, "a", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=FIELDS)
            if new:
                wr.writeheader()
            for k, (exp, arm, s, seed, path) in enumerate(todo, 1):
                if (exp, arm, s, seed) in done:
                    continue
                ref = json.loads((META_DIR / f"{s}.meta.json").read_text())["ref_words"]
                v = separate(path, SEP_DIR / exp / arm / f"{s}@s{seed}.vocals.wav")
                if SEP_ONLY:
                    print(f"[{k}/{len(todo)}] separated {exp}/{arm} {s}@s{seed}", flush=True)
                    continue
                t0 = time.time()
                segs, info = model.transcribe(v, language="en", beam_size=1, vad_filter=use_vad)
                segs = list(segs)
                txt = " ".join(x.text for x in segs)
                hyp = norm_words(txt)
                wr.writerow({"exp": exp, "arm": arm, "song": s, "seed": seed, "input": "vocals_vad" if use_vad else "vocals_novad",
                             "dur_after_vad": round(getattr(info, "duration_after_vad", 0) or 0, 2),
                             "recall": round(recall(ref, hyp), 4), "wer": round(wer(ref, hyp), 4),
                             "n_ref": len(ref), "n_hyp": len(hyp), "n_seg": len(segs),
                             "cr_max": round(max((x.compression_ratio for x in segs), default=0.0), 3),
                             "sec": round(time.time() - t0, 2), "text": txt.strip()[:4000]})
                f.flush()
                if k % 10 == 0 or k == len(todo):
                    print(f"[{k}/{len(todo)}] {exp}/{arm} {s}@s{seed}", flush=True)
        print(f"wrote {out_csv}")


if __name__ == "__main__":
    main()
