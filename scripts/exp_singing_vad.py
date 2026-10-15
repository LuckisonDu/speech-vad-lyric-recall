#!/usr/bin/env python3
"""Singing-voice VAD control (robustness check): replace the speech VAD in front of Whisper by a singing detector.

D1 = per-frame "singing" probability of PANNs CNN14_DecisionLevelMax (AudioSet; not a model trained for singing activity
detection). The gating rule mirrors faster-whisper's default Silero setup: gap < 2.0 s merged, 0.4 s padding on both sides.
The probability threshold is calibrated on the ACE-Step arm-C songs (not part of the main comparison), seeds 1 and 2,
by maximising frame-level F1 against a singing ground truth (frames of the separated vocals with energy within 30 dB of
the 99th percentile); the calibrated value theta* is the main condition, theta* x 0.5 and theta* x 2 are sensitivity checks.
The analysis plan (including the calibration step) was fixed before the transcription stage was run.

Stages (STAGES below):
  probs  per-frame singing probability of every song -> OUT_DIR/probs/<exp>__<arm>__<song>@s<seed>.npy
  calib  threshold calibration on arm C, seeds 1-2 -> OUT_DIR/calib.json (-> results/singing_vad_calib.json)
  seg    kept intervals of Silero and of D1, their overlap, and singing-ground-truth statistics -> OUT_DIR/seg_stats.csv
  asr    transcribe the audio kept by D1 (Whisper large-v3, same settings as the paper) -> OUT_DIR/asr_<exps>_s<seeds>.csv
Inputs : GEN_ROOT/runs/<exp>/<arm>/<song>@s<seed>.wav, separated vocals under GEN_ROOT/sep_vocals(_htdemucs)/ (from
         exp_sep_redecode.py / exp_sep2.py), data/lyrics_meta/<song>.meta.json, the PANNs checkpoint and package.
Supports the singing-VAD robustness paragraph (summarised by analyze_singing_vad.py).
"""
import csv, json, sys, time
from pathlib import Path

# ===== Edit here =====
REPO = Path(__file__).resolve().parent.parent
GEN_ROOT = REPO / "data" / "audio" / "generated"              # runs/<exp>/<arm>/<song>@s<seed>.wav and sep_vocals*/ (generated songs you produce yourself)
OUT_DIR = REPO / "data" / "singing_vad"                       # probabilities, calibration, intermediate CSVs (written here)
META_DIR = REPO / "data" / "lyrics_meta"                      # "<song>.meta.json" files, each with "ref_words"
MANIFEST = REPO / "data" / "manifest.csv"                     # song ids are read from here
PANNS_PYLIB = REPO / "data" / "panns_pylib"                   # folder containing the panns_inference package (only models.py is used)
PANNS_CKPT = REPO / "data" / "Cnn14_DecisionLevelMax.pth"     # PANNs checkpoint
STAGES = ["probs", "calib", "seg", "asr"]                     # which stages to run
SEEDS_ASR = [1, 2]                                            # seeds transcribed (main analysis: 1, 2; extension: 3, 4, 5, 6)
ARMS_ASR = ["B"]                                              # arm(s) transcribed (no planner)
EXPS_ASR = ["vad48_2b", "vad48_xl"]                           # models transcribed
THRESH_ASR = ["d1_cal", "d1_calx0.5", "d1_calx2"]             # conditions transcribed: calibrated theta* (main), theta* x 0.5 and x 2 (sensitivity)
SEEDS_PROBS = [1, 2, 3, 4, 5, 6]                              # seeds for probabilities / interval statistics (arm B); arm C only seeds 1, 2
THRESH_SEG = ["d1_cal", "d1_calx0.5", "d1_calx2"]             # conditions in the interval statistics
CALIB_GRID = [0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3, 0.5]   # calibration grid
MIN_SIL_S, PAD_S = 2.0, 0.4                                   # same as faster-whisper's default VadOptions (min_silence 2000 ms, speech_pad 400 ms)
SING_CLASSES = [27, 32, 33, 34, 35]                           # AudioSet: Singing / Male / Female / Child / Synthetic singing
GT_FLOOR_DB = -30.0                                           # singing ground truth: vocal-stem frame energy >= -30 dB relative to the song's 99th percentile
WIN_SEC = 120.0
DEV = "cuda"                                                  # "cuda" or "cpu"
LIMIT = 0                                                     # > 0: only the first LIMIT items per stage (smoke test)
# =====================
sys.path.insert(0, str(PANNS_PYLIB))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import audio_io
from verify_metrics import norm_words, recall, wer

OUT = OUT_DIR
OUT.mkdir(parents=True, exist_ok=True)
GEN = GEN_ROOT
SONGS = sorted({r["song_id"] for r in csv.DictReader(open(MANIFEST))})
EXPS = ["vad48_2b", "vad48_xl"]


def items_probs():
    out = []
    for exp in EXPS:
        for arm, seeds in (("B", SEEDS_PROBS), ("C", [1, 2])):
            for s in SONGS:
                for sd in seeds:
                    out.append((exp, arm, s, sd))
    return out


def key(exp, arm, s, sd): return f"{exp}__{arm}__{s}@s{sd}"
def wavpath(exp, arm, s, sd): return GEN / "runs" / exp / arm / f"{s}@s{sd}.wav"


def shard(lst):
    return lst[:LIMIT] if LIMIT else lst


# ---------- Stage 1: PANNs per-frame singing probability ----------
def stage_probs():
    import torch
    import importlib.util, types                                # load models.py directly, bypassing the package __init__ (it needs matplotlib and downloads a label table)
    if "matplotlib" not in sys.modules:
        _mp = types.ModuleType("matplotlib"); _pp = types.ModuleType("matplotlib.pyplot"); _mp.pyplot = _pp
        sys.modules["matplotlib"] = _mp; sys.modules["matplotlib.pyplot"] = _pp
    _pk = types.ModuleType("panns_inference"); _pk.__path__ = [str(PANNS_PYLIB / "panns_inference")]
    sys.modules["panns_inference"] = _pk                        # empty shell package (no __init__ executed) so that models.py's relative imports work
    Cnn14_DecisionLevelMax = importlib.import_module("panns_inference.models").Cnn14_DecisionLevelMax
    m = Cnn14_DecisionLevelMax(sample_rate=32000, window_size=1024, hop_size=320, mel_bins=64, fmin=50, fmax=14000, classes_num=527)
    ck = torch.load(PANNS_CKPT, map_location="cpu", weights_only=False)
    m.load_state_dict(ck["model"]); m = m.to(DEV).eval()
    (OUT / "probs").mkdir(exist_ok=True)
    todo = shard(items_probs()); n = 0
    for it in todo:
        f = OUT / "probs" / (key(*it) + ".npy")
        if f.exists(): continue
        a = audio_io.load_stereo(str(wavpath(*it)), 32000, WIN_SEC).mean(0).astype(np.float32)
        with torch.no_grad():
            o = m(torch.from_numpy(a)[None].to(DEV))["framewise_output"][0].float().cpu().numpy()   # (T, 527)
        s = o[:, SING_CLASSES].max(1)
        np.save(f, s.astype(np.float16)); n += 1
        if n % 50 == 0: print("probs", n, "/", len(todo), flush=True)
    print("probs done", flush=True)


# ---------- Threshold calibration: arm C (not part of the main comparison), seeds 1-2 only; maximise frame-level F1 against the singing ground truth ----------
def thr_map():
    c = json.load(open(OUT / "calib.json"))["theta_star"]
    return {"d1_cal": c, "d1_calx0.5": c * 0.5, "d1_calx2": c * 2.0}


def stage_calib():
    from collections import defaultdict
    tp = defaultdict(int); fp = defaultdict(int); fn = defaultdict(int); n_songs = 0
    for exp in EXPS:
        for s in SONGS:
            for sd in (1, 2):
                pf = OUT / "probs" / (key(exp, "C", s, sd) + ".npy")
                gt, _ = gt_sing_mask(exp, "C", s, sd)
                if gt is None or not pf.exists(): continue
                pr = np.load(pf).astype(np.float32)
                idx = np.minimum((np.arange(len(gt)) * len(pr) / len(gt)).astype(int), len(pr) - 1)
                p10 = np.array([pr[int(i * len(pr) / len(gt)):max(int((i + 1) * len(pr) / len(gt)), int(i * len(pr) / len(gt)) + 1)].max() for i in range(len(gt))])
                for t in CALIB_GRID:
                    on = p10 >= t
                    tp[t] += int((on & gt).sum()); fp[t] += int((on & ~gt).sum()); fn[t] += int((~on & gt).sum())
                n_songs += 1
    f1 = {t: (2 * tp[t] / (2 * tp[t] + fp[t] + fn[t])) if (2 * tp[t] + fp[t] + fn[t]) else 0.0 for t in CALIB_GRID}
    best = max(CALIB_GRID, key=lambda t: f1[t])
    json.dump({"theta_star": best, "f1_by_threshold": {str(t): f1[t] for t in CALIB_GRID}, "precision": {str(t): tp[t] / max(tp[t] + fp[t], 1) for t in CALIB_GRID},
               "recall": {str(t): tp[t] / max(tp[t] + fn[t], 1) for t in CALIB_GRID}, "n_songs": n_songs, "set": "ACE-Step arm C, seeds 1-2, 2B+XL; ground truth = htdemucs_ft vocal-stem frames"},
              open(OUT / "calib.json", "w"), ensure_ascii=False, indent=1)
    print("calib theta*", best, "F1", round(f1[best], 3), flush=True)


# ---------- Interval tools ----------
def d1_intervals(s, thr, dur=WIN_SEC):
    """Per-frame probabilities -> kept intervals (seconds): frames >= thr -> merge gaps < MIN_SIL_S -> pad PAD_S on both sides -> merge overlaps."""
    fps = len(s) / dur
    on = np.asarray(s, dtype=np.float32) >= thr
    segs, i = [], 0
    while i < len(on):
        if on[i]:
            j = i
            while j < len(on) and on[j]: j += 1
            segs.append([i / fps, j / fps]); i = j
        else: i += 1
    merged = []
    for a, b in segs:
        if merged and a - merged[-1][1] < MIN_SIL_S: merged[-1][1] = b
        else: merged.append([a, b])
    pad = [[max(0.0, a - PAD_S), min(dur, b + PAD_S)] for a, b in merged]
    out = []
    for a, b in pad:
        if out and a <= out[-1][1]: out[-1][1] = max(out[-1][1], b)
        else: out.append([a, b])
    return out


def silero_intervals(audio16):
    from faster_whisper.vad import get_speech_timestamps, VadOptions
    ch = get_speech_timestamps(audio16, VadOptions())          # defaults: threshold 0.5, min_silence 2000, pad 400, min_speech 0
    return [[c["start"] / 16000.0, c["end"] / 16000.0] for c in ch]


def mask(iv, n=int(WIN_SEC * 10)):
    m = np.zeros(n, dtype=bool)
    for a, b in iv: m[int(round(a * 10)): int(round(b * 10))] = True
    return m


def gt_sing_mask(exp, arm, s, sd):
    """Singing ground truth (0.1 s frames): seeds 1-2 use htdemucs_ft vocals (as in the second-separator check), the others the HDemucs vocals."""
    for sub in (("sep_vocals_htdemucs",) if sd in (1, 2) else ()) + ("sep_vocals",):
        p = GEN / sub / exp / arm / f"{s}@s{sd}.vocals.wav"
        if p.exists():
            v = audio_io.load_mono16k(str(p), duration=WIN_SEC)
            n = int(WIN_SEC * 10); fr = np.zeros(n)
            for i in range(n):
                seg = v[i * 1600:(i + 1) * 1600]
                fr[i] = np.sqrt(np.mean(seg ** 2)) if len(seg) else 0.0
            db = 20 * np.log10(fr + 1e-9); ref = np.percentile(db, 99)
            return (db - ref) >= GT_FLOOR_DB, sub
    return None, None


# ---------- Stage 2: interval statistics ----------
def stage_seg():
    f = OUT / "seg_stats.csv"
    done = set()
    if f.exists():
        done = {(r["exp"], r["arm"], r["song"], r["seed"], r["det"]) for r in csv.DictReader(open(f))}
    fields = ["exp", "arm", "song", "seed", "det", "kept_sec", "silero_kept_sec", "inter_discard_sec", "union_discard_sec", "iou_discard",
              "gt_src", "gt_sing_sec", "disc_sing_frac", "sing_kept_recall", "silero_disc_sing_frac", "silero_sing_kept_recall"]
    new = not f.exists()
    todo = shard(items_probs())
    with open(f, "a", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=fields)
        if new: wr.writeheader()
        for k, (exp, arm, s, sd) in enumerate(todo, 1):
            pf = OUT / "probs" / (key(exp, arm, s, sd) + ".npy")
            if not pf.exists(): continue
            if all((exp, arm, s, str(sd), t) in done for t in THRESH_SEG): continue
            a16 = audio_io.load_mono16k(str(wavpath(exp, arm, s, sd)), duration=WIN_SEC)
            sil = silero_intervals(a16); msil = mask(sil)
            gt, gsrc = gt_sing_mask(exp, arm, s, sd)
            prob = np.load(pf).astype(np.float32)
            TM = thr_map()
            for det in THRESH_SEG:
                if (exp, arm, s, str(sd), det) in done: continue
                iv = d1_intervals(prob, TM[det]); md = mask(iv)
                dsil, dd = ~msil, ~md
                inter = int((dsil & dd).sum()) / 10; uni = int((dsil | dd).sum()) / 10
                row = {"exp": exp, "arm": arm, "song": s, "seed": sd, "det": det, "kept_sec": md.sum() / 10, "silero_kept_sec": msil.sum() / 10,
                       "inter_discard_sec": inter, "union_discard_sec": uni, "iou_discard": (inter / uni) if uni else 1.0, "gt_src": gsrc or ""}
                if gt is not None:
                    row["gt_sing_sec"] = gt.sum() / 10
                    row["disc_sing_frac"] = (gt & dd).sum() / dd.sum() if dd.sum() else ""
                    row["sing_kept_recall"] = (gt & md).sum() / gt.sum() if gt.sum() else ""
                    row["silero_disc_sing_frac"] = (gt & dsil).sum() / dsil.sum() if dsil.sum() else ""
                    row["silero_sing_kept_recall"] = (gt & msil).sum() / gt.sum() if gt.sum() else ""
                wr.writerow(row)
            fh.flush()
            if k % 50 == 0: print("seg", k, "/", len(todo), flush=True)
    print("seg done", flush=True)


# ---------- Stage 3: transcription after D1 gating (same Whisper settings as the paper) ----------
def stage_asr():
    from faster_whisper import WhisperModel
    model = WhisperModel("large-v3", device=DEV, compute_type="float16" if DEV == "cuda" else "int8")
    f = OUT / ("asr_" + "_".join(EXPS_ASR) + "_s" + "".join(str(x) for x in SEEDS_ASR) + ("" ) + ".csv")
    fields = ["exp", "arm", "song", "seed", "cond", "recall", "wer", "n_ref", "n_hyp", "kept_sec", "sec", "text"]
    done = set()
    if f.exists():
        done = {(r["exp"], r["arm"], r["song"], r["seed"], r["cond"]) for r in csv.DictReader(open(f))}
    new = not f.exists()
    todo = shard([(e, a, s, sd) for e in EXPS_ASR for a in ARMS_ASR for s in SONGS for sd in SEEDS_ASR])
    with open(f, "a", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=fields)
        if new: wr.writeheader()
        for k, (exp, arm, s, sd) in enumerate(todo, 1):
            conds = [t for t in THRESH_ASR if (exp, arm, s, str(sd), t) not in done]
            if not conds: continue
            ref = json.loads((META_DIR / f"{s}.meta.json").read_text())["ref_words"]
            a16 = audio_io.load_mono16k(str(wavpath(exp, arm, s, sd)), duration=WIN_SEC)
            prob = np.load(OUT / "probs" / (key(exp, arm, s, sd) + ".npy")).astype(np.float32)
            for c in conds:
                iv = d1_intervals(prob, thr_map()[c])
                kept = sum(b - a for a, b in iv)
                t0 = time.time()
                if kept < 0.5:
                    txt = ""
                else:
                    au = np.concatenate([a16[int(a * 16000): int(b * 16000)] for a, b in iv]).astype(np.float32)
                    segs, _ = model.transcribe(au, language="en", beam_size=1, vad_filter=False)
                    txt = " ".join(x.text for x in list(segs))
                hyp = norm_words(txt)
                wr.writerow({"exp": exp, "arm": arm, "song": s, "seed": sd, "cond": c, "recall": round(recall(ref, hyp), 4), "wer": round(wer(ref, hyp), 4),
                             "n_ref": len(ref), "n_hyp": len(hyp), "kept_sec": round(kept, 2), "sec": round(time.time() - t0, 2), "text": txt.strip()[:4000]})
                fh.flush()
            if k % 10 == 0: print("asr", k, "/", len(todo), flush=True)
    print("asr done", flush=True)


if __name__ == "__main__":
    for st in STAGES:
        {"probs": stage_probs, "calib": stage_calib, "seg": stage_seg, "asr": stage_asr}[st]()
    print("ALL_DONE", " ".join(STAGES), flush=True)
