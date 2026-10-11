#!/usr/bin/env python3
"""A third speech VAD implementation: WebRTC VAD (py-webrtcvad, GMM-based, from the Google WebRTC project).
Rules fixed before looking at results:
  mix at 16 kHz mono, 16-bit, 30 ms frames, aggressiveness=2 (middle of 0..3); frames judged as speech are joined
  into segments; post-processing identical to exp_energy_gate.py (gaps < 1.0 s merged, 0.2 s padding,
  segments < 0.3 s dropped) - only the "is there voice" step differs.
RUN_JAMENDO: how many human-annotated words this gate drops on the 20 Jam-ALT songs (JamendoLyrics word
  timings; same protocol as exp_vad_truth.py) -> ../results/webrtc_gate_truth.csv. This is the part used in the paper.
RUN_SONGS (off by default; not used for the paper's numbers): ASR on the original mix restricted to the kept
  segments (clip_timestamps; large-v3 fp16, beam 1, en, vad_filter=False) -> ../results/webrtc_gate48.csv.
  The kept-duration numbers on generated songs come from analyze_webrtc_kept.py, which imports webrtc_segments.
Dependency: webrtcvad-wheels (pip install webrtcvad-wheels).

Reads  : ROOT/meta.jsonl, ROOT/words/<id>.csv, ROOT/audio/<id>.* (Jam-ALT); for RUN_SONGS also generated songs
         under GEN_ROOT and META_DIR/<song>.meta.json.
Supports the paper's WebRTC VAD word-drop numbers.
"""
import csv, json, time
from pathlib import Path
import numpy as np
from exp_energy_gate import ARMS, FIELDS

# ===== Edit here =====
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"   # output folder
GEN_ROOT = Path("data/audio/generated")   # contains <root>/<exp>/<arm>/<song>@s<seed>.wav (generated songs; <root> as in ARMS)
META_DIR = Path("data/lyrics_meta")       # "<song>.meta.json" files, each with "ref_words"
ROOT = Path("data/jamendo_truth")         # Jam-ALT word-timing data, same layout as in exp_vad_truth.py
RUN_JAMENDO = True                        # webrtc_gate_truth.csv (used in the paper)
RUN_SONGS = False                         # webrtc_gate48.csv (optional ASR run, not used in the paper)
# =====================


def webrtc_segments(a, sr=16000, mode=2, frame_ms=30, merge=1.0, pad=0.2, min_len=0.3):
    import webrtcvad
    v = webrtcvad.Vad(mode)
    pcm = (np.clip(a, -1, 1) * 32767).astype("<i2").tobytes()
    n = int(sr * frame_ms / 1000); step = 2 * n; fr = frame_ms / 1000
    segs, start = [], None
    for i in range(len(pcm) // step):
        sp = v.is_speech(pcm[i * step:(i + 1) * step], sr)
        t = i * fr
        if sp and start is None:
            start = t
        if not sp and start is not None:
            segs.append([start, t]); start = None
    if start is not None:
        segs.append([start, (len(pcm) // step) * fr])
    merged = []
    for s, e in segs:
        if merged and s - merged[-1][1] < merge:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    dur = len(a) / sr; out = []
    for s, e in merged:
        if e - s < min_len:
            continue
        s, e = max(0.0, s - pad), min(dur, e + pad)
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


def run_songs():
    import audio_io
    from faster_whisper import WhisperModel
    from verify_metrics import norm_words, recall, wer
    m = WhisperModel("large-v3", device="cuda", compute_type="float16")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "webrtc_gate48.csv"; done = set()
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
                mix = np.asarray(audio_io.load_mono16k(str(w), duration=120.0), dtype=np.float32)
                segs = webrtc_segments(mix)
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
    import audio_io
    meta = {json.loads(l)["id"]: json.loads(l) for l in open(ROOT / "meta.jsonl", encoding="utf-8")}
    rows = []
    for i in sorted(k for k, v in meta.items() if v["language"] == "en"):
        words = [(float(r["word_start"]), float(r["word_end"])) for r in csv.DictReader(open(ROOT / "words" / f"{i}.csv"))]
        au = next(iter(sorted((ROOT / "audio").glob(i + ".*"))))
        segs = webrtc_segments(np.asarray(audio_io.load_mono16k(str(au)), dtype=np.float32))
        keep = sum(sum(max(0.0, min(we, e) - max(ws, s)) for s, e in segs) >= 0.5 * max(we - ws, 1e-3) for ws, we in words)
        rows.append({"id": i, "n_words": len(words), "words_kept": keep, "frac_words_dropped": round(1 - keep / len(words), 4),
                     "kept_sec": round(sum(e - s for s, e in segs), 1)})
        print(rows[-1], flush=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "webrtc_gate_truth.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)


if __name__ == "__main__":
    if RUN_JAMENDO:
        run_jamendo()
    if RUN_SONGS:
        run_songs()
