#!/usr/bin/env python3
"""Main experiment: speech VAD on/off when transcribing generated songs (and the original recordings).

For every generated song (first 120 s) and the original commercial recording of the same
48 songs, transcribe with faster-whisper large-v3 (fp16, beam_size=1, language="en") twice on
the SAME decoded audio, flipping only vad_filter (False / True), and score lyric recall / WER
against the reference lyrics.
Inputs:  GEN_ROOT/runs/<exp>/<arm>/<song>@s<seed>.wav, META_DIR/<song>.meta.json ("ref_words"),
         ORIG_DIR/<song>.flac.
Outputs: RESULTS_DIR/<csv> for each entry of RUNS (vad_gen48*.csv; resumable). The "text"
         column holds the transcript (truncated to 4000 chars) so analyses can be rerun.
Supports: the main VAD on/off result over the four generator families (ACE-Step 2B / XL,
HeartMuLa, LeVo, DiffRhythm) and the original-recording control (Results section, main table).
"""
import csv, json, time
from pathlib import Path

from verify_metrics import norm_words, recall, wer

# ===== Edit here =====
REPO = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO / "results"                   # output CSVs go here
GEN_ROOT = REPO / "data" / "audio" / "generated"           # runs/<exp>/<arm>/<song>@s<seed>.wav, 120 s generated songs you produce yourself
META_DIR = REPO / "data" / "lyrics_meta"              # <song>.meta.json with "ref_words" (normalized reference lyrics)
ORIG_DIR = Path("/path/to/original_recordings")  # <song>.flac commercial originals (not redistributable; obtain yourself)
WIN_SEC = 120.0                                  # seconds of audio transcribed per song
MODEL_NAME = "large-v3"                          # faster-whisper model
# (output csv, list of "<exp>/<arm>") - the configurations used for the paper's 48-song runs
RUNS = [
    ("vad_gen48.csv", ["vad48_2b/B", "vad48_2b/C", "vad48_xl/B", "vad48_xl/C"]),   # ACE-Step 2B and XL, two arms each
    ("vad_gen48_heart.csv", ["vad48_heart/H"]),                                     # HeartMuLa
    ("vad_gen48_levo.csv", ["vad48_levo/L"]),                                       # LeVo
    ("vad_gen48_dr.csv", ["vad48_diffrhythm/D"]),                                   # DiffRhythm
]
# =====================

FIELDS = ["src", "exp", "arm", "song", "seed", "vad", "recall", "wer", "n_ref", "n_hyp",
          "n_seg", "cr_max", "sec", "text"]


def items(arms):
    """(src, exp, arm, song, seed, audio path); generated songs first, then the originals of the same songs."""
    out, songs = [], []
    for ea in arms:
        exp, arm = ea.split("/")
        for w in sorted((GEN_ROOT / "runs" / exp / arm).glob("*.wav")):
            song, seed = w.stem.split("@s")
            out.append(("gen", exp, arm, song, int(seed), w))
            if song not in songs:
                songs.append(song)
    for s in songs:
        out.append(("orig", "orig", "orig", s, 0, ORIG_DIR / f"{s}.flac"))
    return out


def run_one(model, out_name, arms):
    import numpy as np, librosa
    todo = items(arms)
    print(f"{out_name}: {len(todo)} items (generated {sum(t[0] == 'gen' for t in todo)} / "
          f"original {sum(t[0] == 'orig' for t in todo)})", flush=True)
    out_csv = RESULTS_DIR / out_name
    done = set()
    if out_csv.exists():
        done = {(r["exp"], r["arm"], r["song"], r["seed"], r["vad"]) for r in csv.DictReader(open(out_csv))}
    new = not out_csv.exists()
    with open(out_csv, "a", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            wr.writeheader()
        for k, (src, exp, arm, s, seed, path) in enumerate(todo, 1):
            ref = json.loads((META_DIR / f"{s}.meta.json").read_text())["ref_words"]
            audio = None
            for vad in (False, True):
                if (exp, arm, s, str(seed), str(vad)) in done:
                    continue
                if audio is None:                       # same decoded audio for both conditions
                    audio, _ = librosa.load(str(path), sr=16000, mono=True, duration=WIN_SEC)
                    audio = np.asarray(audio, dtype=np.float32)
                t0 = time.time()
                segs, _info = model.transcribe(audio, language="en", beam_size=1, vad_filter=vad)
                segs = list(segs)
                sec = time.time() - t0
                txt = " ".join(x.text for x in segs)
                hyp = norm_words(txt)
                wr.writerow({"src": src, "exp": exp, "arm": arm, "song": s, "seed": seed, "vad": vad,
                             "recall": round(recall(ref, hyp), 4), "wer": round(wer(ref, hyp), 4),
                             "n_ref": len(ref), "n_hyp": len(hyp), "n_seg": len(segs),
                             "cr_max": round(max((x.compression_ratio for x in segs), default=0.0), 3),
                             "sec": round(sec, 2), "text": txt.strip()[:4000]})
                f.flush()
            print(f"[{k}/{len(todo)}] {src} {exp}/{arm} {s}@s{seed}", flush=True)
    print(f"wrote {out_csv}")


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    from faster_whisper import WhisperModel
    model = WhisperModel(MODEL_NAME, device="cuda", compute_type="float16")
    print("model ready", flush=True)
    for out_name, arms in RUNS:
        run_one(model, out_name, arms)


if __name__ == "__main__":
    main()
