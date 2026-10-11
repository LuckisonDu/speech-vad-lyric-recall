#!/usr/bin/env python3
"""Singing-oriented gate as a control for speech VAD (energy gating on separated vocals instead of a speech VAD,
following the idea of Syed et al. 2025).
  Gate = RMS on the HDemucs-separated vocals (16 kHz mono cached by exp_sep_redecode.py), 50 ms frames, 25 ms hop;
         frames above (95th percentile of this song - 30 dB) count as voiced; gaps < 1.0 s are merged,
         0.2 s padding on each side, segments shorter than 0.3 s are dropped.
  ASR  = the original mix (same input as P0/P1), transcribing only the gate segments (faster-whisper clip_timestamps);
         otherwise as exp_vad_gen.py (large-v3 fp16, beam 1, language=en, vad_filter=False).
  So the only difference from "Silero on the mix" is the gate itself. Rules were fixed before looking at results.
Also (RUN_JAMENDO): how many human-annotated words the same gate drops on the 20 Jam-ALT songs
(JamendoLyrics word timings; same protocol as exp_vad_truth.py).

Reads  : GEN_ROOT/<root>/<exp>/<arm>/<song>@s<seed>.wav (first 120 s), GEN_ROOT/sep_vocals/<exp>/<arm>/<song>@s<seed>.vocals.wav,
         META_DIR/<song>.meta.json ("ref_words"); Jam-ALT data under exp_vad_truth.ROOT.
Writes : ../results/energy_gate48.csv (7 systems, 672 songs; includes transcript column "text"),
         ../results/energy_gate_truth.csv. Summarised by analyze_energy_gate.py.
Supports the paper's energy-gate control numbers.
"""
import csv, json, time
from pathlib import Path
import numpy as np

# ===== Edit here =====
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"   # output folder
GEN_ROOT = Path("data/audio/generated")   # contains <root>/<exp>/<arm>/<song>@s<seed>.wav (generated songs you produce yourself) and sep_vocals/ (from exp_sep_redecode.py)
META_DIR = Path("data/lyrics_meta")       # "<song>.meta.json" files, each with "ref_words"
# (root folder under GEN_ROOT, <exp>/<arm>) for the 7 systems: 4 ACE-Step, HeartMuLa, LeVo, DiffRhythm
ARMS = [("runs", "vad48_2b/B"), ("runs", "vad48_2b/C"), ("runs", "vad48_xl/B"), ("runs", "vad48_xl/C"),
        ("runs", "vad48_heart/H"), ("runs", "vad48_levo/L"),
        ("runs", "vad48_diffrhythm/D")]
RUN_SONGS = True      # energy_gate48.csv (generated songs)
RUN_JAMENDO = True    # energy_gate_truth.csv (Jam-ALT word-level check)
# =====================

FIELDS = ["exp", "arm", "song", "seed", "recall", "wer", "n_ref", "n_hyp", "kept_sec", "n_seg", "sec", "text"]


def energy_segments(v, sr=16000, frame=0.05, hop=0.025, rel_db=30.0, merge=1.0, pad=0.2, min_len=0.3):
    fl, hl = int(frame * sr), int(hop * sr)
    if len(v) < fl:
        return []
    n = 1 + (len(v) - fl) // hl
    idx = np.arange(fl)[None, :] + hl * np.arange(n)[:, None]
    rms = np.sqrt(np.mean(v[idx] ** 2, axis=1) + 1e-12)
    db = 20 * np.log10(rms)
    act = db > (np.percentile(db, 95) - rel_db)
    segs, start = [], None
    for i, a in enumerate(act):
        t = i * hop
        if a and start is None:
            start = t
        if not a and start is not None:
            segs.append([start, t + frame]); start = None
    if start is not None:
        segs.append([start, n * hop + frame])
    merged = []
    for s, e in segs:
        if merged and s - merged[-1][1] < merge:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    dur = len(v) / sr
    out = [[max(0.0, s - pad), min(dur, e + pad)] for s, e in merged if e - s >= min_len]
    fin = []
    for s, e in out:                      # padding may create overlaps; merge once more
        if fin and s <= fin[-1][1]:
            fin[-1][1] = max(fin[-1][1], e)
        else:
            fin.append([s, e])
    return fin


def run_songs():
    import audio_io
    from faster_whisper import WhisperModel
    from verify_metrics import norm_words, recall, wer
    m = WhisperModel("large-v3", device="cuda", compute_type="float16")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "energy_gate48.csv"; done = set()
    if out.exists():
        done = {(r["exp"], r["arm"], r["song"], r["seed"]) for r in csv.DictReader(open(out))}
    new = not out.exists()
    with open(out, "a", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            wr.writeheader()
        k = 0
        for root, ea in ARMS:
            exp, arm = ea.split("/")
            for w in sorted((GEN_ROOT / root / exp / arm).glob("*.wav")):
                s, sd = w.stem.split("@s"); k += 1
                if (exp, arm, s, sd) in done:
                    continue
                ref = json.loads((META_DIR / f"{s}.meta.json").read_text())["ref_words"]
                voc = audio_io.load_mono16k(str(GEN_ROOT / "sep_vocals" / exp / arm / f"{s}@s{sd}.vocals.wav"))
                mix = np.asarray(audio_io.load_mono16k(str(w), duration=120.0), dtype=np.float32)
                segs = energy_segments(np.asarray(voc, dtype=np.float32)[: len(mix)])
                t0 = time.time()
                if segs:
                    it, _ = m.transcribe(mix, language="en", beam_size=1, vad_filter=False,
                                         clip_timestamps=[x for se in segs for x in se])
                    txt = " ".join(x.text for x in it)
                else:
                    txt = ""
                hyp = norm_words(txt)
                wr.writerow({"exp": exp, "arm": arm, "song": s, "seed": sd, "recall": round(recall(ref, hyp), 4),
                             "wer": round(wer(ref, hyp), 4), "n_ref": len(ref), "n_hyp": len(hyp),
                             "kept_sec": round(sum(e - b for b, e in segs), 2), "n_seg": len(segs),
                             "sec": round(time.time() - t0, 2), "text": txt.strip()[:4000]})
                f.flush()
                if k % 50 == 0:
                    print(f"[{k}]", flush=True)


def run_jamendo():
    from exp_vad_truth import separate, ROOT
    meta = {json.loads(l)["id"]: json.loads(l) for l in open(ROOT / "meta.jsonl", encoding="utf-8")}
    rows = []
    for i in sorted(k for k, v in meta.items() if v["language"] == "en"):
        words = [(float(r["word_start"]), float(r["word_end"])) for r in csv.DictReader(open(ROOT / "words" / f"{i}.csv"))]
        au = next(iter(sorted((ROOT / "audio").glob(i + ".*"))))
        segs = energy_segments(separate(au))
        keep = sum(sum(max(0.0, min(we, e) - max(ws, s)) for s, e in segs) >= 0.5 * max(we - ws, 1e-3) for ws, we in words)
        rows.append({"id": i, "n_words": len(words), "words_kept": keep, "frac_words_dropped": round(1 - keep / len(words), 4),
                     "kept_sec": round(sum(e - s for s, e in segs), 1)})
        print(rows[-1], flush=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "energy_gate_truth.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)


if __name__ == "__main__":
    if RUN_JAMENDO:
        run_jamendo()
    if RUN_SONGS:
        run_songs()
