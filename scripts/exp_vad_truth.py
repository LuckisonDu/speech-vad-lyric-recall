#!/usr/bin/env python3
"""Ground-truth check of what speech VAD drops: is anybody singing in the audio that VAD removes?
Data: the 20 English Jam-ALT songs (same audio as exp_vad_bench.py) with human word-level timings from
JamendoLyrics (huggingface.co/datasets/jamendolyrics/jamendolyrics, annotations/words/<name>.csv:
word_start, word_end); Jam-ALT and JamendoLyrics share the same Jamendo tracks, matched by name.
Method: run the Silero VAD bundled with faster-whisper 1.2.1 on each full song (not cropped to 120 s) and keep
only its kept segments; an annotated word counts as dropped if its [start, end] overlaps the kept segments by
less than half of its duration. No ASR is run, so the result does not depend on Whisper.
Conditions:
  mix_default   original mix, VadOptions() defaults (threshold 0.5, min_silence 2000 ms, pad 400 ms) = paper P0
  mix_batched   original mix, min_silence_duration_ms=160, max_speech_duration_s=30 = BatchedInferencePipeline defaults
  mix_thr015    original mix, Silero threshold 0.15 (pad 400 ms, min_silence 800 ms, min_speech 100 ms, max 30 s)
  voc_default   HDemucs-separated vocals (as exp_sep_redecode.py), VadOptions() defaults

Reads  : ROOT/meta.jsonl, ROOT/words/<id>.csv, ROOT/audio/<id>.*; ../results/vad_bench.csv (optional, VAD on/off
         recall on the same 20 songs, from exp_vad_bench.py) for the correlation.
Writes : ../results/vad_truth.csv (one row per song x condition) and ../results/vad_truth_summary.json.
Supports the paper's "share of human-annotated sung words dropped by VAD" numbers and their correlation
with the recall loss.
"""
import csv, json
from pathlib import Path
import numpy as np

# ===== Edit here =====
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"   # output folder (also holds vad_bench.csv)
ROOT = Path("data/jamendo_truth")   # meta.jsonl (one JSON per line: "id", "language"), words/<id>.csv (word_start, word_end), audio/<id>.<ext> (Jam-ALT audio)
COND = {
    "mix_default": ("mix", {}),
    "mix_batched": ("mix", {"min_silence_duration_ms": 160, "max_speech_duration_s": 30}),
    "mix_thr015": ("mix", {"threshold": 0.15, "speech_pad_ms": 400, "min_silence_duration_ms": 800,
                           "min_speech_duration_ms": 100, "max_speech_duration_s": 30}),
    "voc_default": ("voc", {}),
}   # condition -> (input: "mix" or separated "voc", VadOptions overrides)
# =====================


def separate(path):
    import torch, torchaudio, audio_io
    b = torchaudio.pipelines.HDEMUCS_HIGH_MUSDB_PLUS
    sep = b.get_model().eval(); sr_m = b.sample_rate; vi = sep.sources.index("vocals")
    w = torch.from_numpy(audio_io.load_stereo(str(path), sr_m, None))
    ref = w.mean(0); w = (w - ref.mean()) / (ref.std() + 1e-8)
    CH, OV = int(20.0 * sr_m), int(2.0 * sr_m); out, pos = [], 0
    while pos < w.shape[1]:
        with torch.no_grad():
            v = sep(w[:, pos: pos + CH + OV].unsqueeze(0))[0][vi]
        if pos > 0:
            v = v[:, OV:]
        out.append(v[:, :CH] if pos + CH < w.shape[1] else v); pos += CH
    V = (torch.cat(out, 1).mean(0) * (ref.std() + 1e-8)).numpy()
    return audio_io.resample(V, sr_m, 16000).astype(np.float32)


def main():
    import audio_io
    from faster_whisper.vad import VadOptions, get_speech_timestamps
    meta = {json.loads(l)["id"]: json.loads(l) for l in open(ROOT / "meta.jsonl", encoding="utf-8")}
    ids = [i for i, m in meta.items() if m["language"] == "en"]
    rows = []
    for k, i in enumerate(sorted(ids), 1):
        words = [(float(r["word_start"]), float(r["word_end"])) for r in csv.DictReader(open(ROOT / "words" / f"{i}.csv"))]
        au = next(iter(sorted((ROOT / "audio").glob(i + ".*"))))
        mix = np.asarray(audio_io.load_mono16k(str(au), duration=None), dtype=np.float32)
        voc = None
        for c, (src, kw) in COND.items():
            if src == "voc" and voc is None:
                voc = separate(au)
            a = mix if src == "mix" else voc
            segs = [(s["start"] / 16000, s["end"] / 16000) for s in get_speech_timestamps(a, VadOptions(**kw))]
            def kept(ws, we):
                ov = sum(max(0.0, min(we, e) - max(ws, s)) for s, e in segs)
                return ov >= 0.5 * max(we - ws, 1e-3)
            n_keep = sum(kept(ws, we) for ws, we in words)
            rows.append({"id": i, "cond": c, "dur": round(len(a) / 16000, 1), "kept_sec": round(sum(e - s for s, e in segs), 1),
                         "n_words": len(words), "words_kept": n_keep, "frac_words_dropped": round(1 - n_keep / len(words), 4)})
        print(f"[{k}/{len(ids)}] {i} " + " ".join(f"{r['cond']}={r['frac_words_dropped']}" for r in rows[-len(COND):]), flush=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "vad_truth.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)

    # summary + comparison with vad_bench.csv (same 20 songs, large-v3 recall with VAD on/off)
    rec = {}
    bp = RESULTS_DIR / "vad_bench.csv"
    if bp.exists():
        for r in csv.DictReader(open(bp)):
            if r["bench"] == "jamalt":
                rec[(r["id"], r["vad"])] = float(r["recall"])
    S = {"n_songs": len(ids), "n_words": int(sum(r["n_words"] for r in rows if r["cond"] == "mix_default")), "cond": {}}
    from scipy.stats import spearmanr
    for c in COND:
        rr = [r for r in rows if r["cond"] == c]
        tot_w = sum(r["n_words"] for r in rr); tot_d = sum(r["n_words"] - r["words_kept"] for r in rr)
        d = {"words_dropped_pooled": round(tot_d / tot_w, 4), "median_song_frac_dropped": float(np.median([r["frac_words_dropped"] for r in rr])),
             "songs_over_20pct": sum(r["frac_words_dropped"] > 0.2 for r in rr), "median_kept_sec": float(np.median([r["kept_sec"] for r in rr])),
             "median_dur": float(np.median([r["dur"] for r in rr]))}
        if c == "mix_default" and rec:
            xs = [r["frac_words_dropped"] for r in rr if (r["id"], "True") in rec]
            ys = [rec[(r["id"], "False")] - rec[(r["id"], "True")] for r in rr if (r["id"], "True") in rec]
            rho, p = spearmanr(xs, ys); d["spearman_dropped_vs_recall_loss"] = [round(float(rho), 3), float(p)]; d["n_corr"] = len(xs)
        S["cond"][c] = d
    json.dump(S, open(RESULTS_DIR / "vad_truth_summary.json", "w"), indent=1)
    print(json.dumps(S, indent=1))


if __name__ == "__main__":
    main()
