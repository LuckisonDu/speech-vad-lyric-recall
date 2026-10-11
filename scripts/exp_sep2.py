#!/usr/bin/env python3
"""Robustness check with a second source separator for the separated-vocals mechanism check.
exp_sep_redecode.py uses torchaudio HDEMUCS_HIGH_MUSDB_PLUS; here the same analysis is redone with
Demucs v4 `htdemucs_ft` (demucs 4.0.1; the four sub-model checkpoints f7e0c4bc / d12395a8 / 92cfc3b6 / 04573f0d
are fetched into the torch hub cache). Separation settings = official demucs CLI defaults
(apply_model shifts=1, split=True, overlap=0.25; input normalised by the mono-mix mean/std and scaled back).
Input = first 120 s of each generated song, 44.1 kHz stereo; vocals = mean of the two channels,
resampled to 16 kHz mono and cached. Rules were fixed before looking at results.

RUN_SEP  separate the 384 ACE-Step songs, cache GEN_ROOT/sep_vocals_htdemucs/<exp>/<arm>/<song>@s<seed>.vocals.wav,
         and compute part A (no ASR): seconds kept by Silero (faster_whisper.vad get_speech_timestamps,
         default VadOptions) on the mix / HDemucs vocals / htdemucs_ft vocals, seconds kept by the energy gate
         (exp_energy_gate.energy_segments) on both vocal stems, and Silero on a controlled remix
         g=0.5: htdemucs_ft vocals + 0.5 x (mix - vocals)  -> ../results/sep2_kept.csv
RUN_ASR  part B: faster-whisper large-v3 (fp16, beam 1, language=en, as exp_sep_redecode.py / exp_energy_gate.py)
         mode=vocals_vad  new vocals + vad_filter=True (as exp_sep_redecode.py with VAD on)
         mode=egate       original mix, transcribing only the energy-gate segments computed on the new vocals
         mode=remix05_vad controlled remix g=0.5 + vad_filter=True (g=0 is vocals_vad; g=1 is the mix with VAD on)
         -> ../results/sep2_asr.csv (includes the transcript column "text"). Resumable.

Reads  : GEN_ROOT/runs/<exp>/<arm>/*.wav, GEN_ROOT/sep_vocals/... (HDemucs vocals cached by exp_sep_redecode.py),
         META_DIR/<song>.meta.json ("ref_words")
Supports the paper's second-separator robustness numbers (summarised by analyze_sep2.py).
"""
import csv, json, time
from pathlib import Path
import numpy as np

# ===== Edit here =====
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"   # output folder
GEN_ROOT = Path("data/audio/generated")   # contains runs/<exp>/<arm>/<song>@s<seed>.wav and sep_vocals/ (from exp_sep_redecode.py)
NEWDIR = GEN_ROOT / "sep_vocals_htdemucs"  # cache of htdemucs_ft vocals (written here)
META_DIR = Path("data/lyrics_meta")       # "<song>.meta.json" files, each with "ref_words"
WIN_SEC = 120.0                           # only the first 120 s of every song is used
ARMS = ["vad48_2b/B", "vad48_2b/C", "vad48_xl/B", "vad48_xl/C"]   # 4 ACE-Step systems
RUN_SEP = True                            # part A (separation + kept seconds)
RUN_ASR = True                            # part B (ASR)
DEVICE = "cuda"                           # device for the separator
MODES = ["vocals_vad", "egate", "remix05_vad"]   # ASR modes of part B
# =====================

KFIELDS = ["exp", "arm", "song", "seed", "silero_mix", "silero_hdemucs", "silero_htft", "energy_hdemucs", "energy_htft", "silero_remix05_htft", "sep_sec"]
AFIELDS = ["exp", "arm", "song", "seed", "mode", "recall", "wer", "n_ref", "n_hyp", "n_seg", "kept_sec", "sec", "text"]


def todo_list():
    t = []
    for ea in ARMS:
        exp, arm = ea.split("/")
        for w in sorted((GEN_ROOT / "runs" / exp / arm).glob("*.wav")):
            s, sd = w.stem.split("@s")
            t.append((exp, arm, s, sd, w))
    return t


def silero_kept(a):
    from faster_whisper.vad import VadOptions, get_speech_timestamps
    return round(sum(x["end"] - x["start"] for x in get_speech_timestamps(np.asarray(a, dtype=np.float32), VadOptions())) / 16000.0, 2)


def remix(mix, voc, g):
    n = min(len(mix), len(voc)); m, v = mix[:n], voc[:n]
    return np.clip(v + g * (m - v), -1, 1).astype(np.float32)


def run_sep(device):
    import torch, audio_io
    from demucs.apply import apply_model
    from demucs.pretrained import get_model
    from exp_energy_gate import energy_segments
    model = get_model("htdemucs_ft").eval().to(device)
    sr_m, vi = model.samplerate, list(model.sources).index("vocals")
    print("htdemucs_ft ready", sr_m, list(model.sources), flush=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "sep2_kept.csv"; done = set()
    if out.exists():
        done = {(r["exp"], r["arm"], r["song"], r["seed"]) for r in csv.DictReader(open(out))}
    new = not out.exists(); todo = todo_list()
    with open(out, "a", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=KFIELDS)
        if new:
            wr.writeheader()
        for k, (exp, arm, s, sd, w) in enumerate(todo, 1):
            if (exp, arm, s, sd) in done:
                continue
            vp = NEWDIR / exp / arm / f"{s}@s{sd}.vocals.wav"; t0 = time.time()
            if not vp.exists():
                wav = torch.from_numpy(audio_io.load_stereo(w, sr_m, WIN_SEC))
                ref = wav.mean(0); mu, sdv = ref.mean(), ref.std() + 1e-8
                with torch.no_grad():
                    src = apply_model(model, ((wav - mu) / sdv)[None], device=device, shifts=1, split=True, overlap=0.25, progress=False)[0]
                V = (src[vi] * sdv + mu).mean(0).cpu().numpy()
                vp.parent.mkdir(parents=True, exist_ok=True)
                audio_io.write_wav16(vp, audio_io.resample(V, sr_m, 16000), 16000)
            sep_sec = round(time.time() - t0, 1)
            vn = audio_io.load_mono16k(vp)
            vh = audio_io.load_mono16k(GEN_ROOT / "sep_vocals" / exp / arm / f"{s}@s{sd}.vocals.wav")
            mix = audio_io.load_mono16k(w, duration=WIN_SEC)
            seg = lambda v: round(sum(e - b for b, e in energy_segments(np.asarray(v, dtype=np.float32)[: len(mix)])), 2)
            wr.writerow({"exp": exp, "arm": arm, "song": s, "seed": sd, "silero_mix": silero_kept(mix), "silero_hdemucs": silero_kept(vh),
                         "silero_htft": silero_kept(vn), "energy_hdemucs": seg(vh), "energy_htft": seg(vn),
                         "silero_remix05_htft": silero_kept(remix(mix, vn, 0.5)), "sep_sec": sep_sec})
            f.flush()
            if k % 10 == 0 or k == len(todo):
                print(f"[{k}/{len(todo)}] {exp}/{arm} {s}@s{sd} sep {sep_sec}s", flush=True)


def run_asr(modes):
    import audio_io
    from faster_whisper import WhisperModel
    from verify_metrics import norm_words, recall, wer
    from exp_energy_gate import energy_segments
    m = WhisperModel("large-v3", device="cuda", compute_type="float16")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "sep2_asr.csv"; done = set()
    if out.exists():
        done = {(r["exp"], r["arm"], r["song"], r["seed"], r["mode"]) for r in csv.DictReader(open(out))}
    new = not out.exists(); todo = todo_list()
    with open(out, "a", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=AFIELDS)
        if new:
            wr.writeheader()
        for mode in modes:
            for k, (exp, arm, s, sd, w) in enumerate(todo, 1):
                if (exp, arm, s, sd, mode) in done:
                    continue
                vp = NEWDIR / exp / arm / f"{s}@s{sd}.vocals.wav"
                if not vp.exists():
                    continue
                ref = json.loads((META_DIR / f"{s}.meta.json").read_text())["ref_words"]
                vn = audio_io.load_mono16k(vp); t0 = time.time(); kept = 0.0
                if mode == "egate":
                    mix = audio_io.load_mono16k(w, duration=WIN_SEC)
                    segs = energy_segments(np.asarray(vn, dtype=np.float32)[: len(mix)])
                    kept = sum(e - b for b, e in segs)
                    if segs:
                        it, _ = m.transcribe(mix, language="en", beam_size=1, vad_filter=False, clip_timestamps=[x for se in segs for x in se])
                        it = list(it)
                    else:
                        it = []
                else:
                    a = vn if mode == "vocals_vad" else remix(audio_io.load_mono16k(w, duration=WIN_SEC), vn, 0.5)
                    it, info = m.transcribe(a, language="en", beam_size=1, vad_filter=True)
                    it = list(it); kept = getattr(info, "duration_after_vad", 0) or 0
                txt = " ".join(x.text for x in it); hyp = norm_words(txt)
                wr.writerow({"exp": exp, "arm": arm, "song": s, "seed": sd, "mode": mode, "recall": round(recall(ref, hyp), 4),
                             "wer": round(wer(ref, hyp), 4), "n_ref": len(ref), "n_hyp": len(hyp), "n_seg": len(it),
                             "kept_sec": round(kept, 2), "sec": round(time.time() - t0, 2), "text": txt.strip()[:4000]})
                f.flush()
                if k % 20 == 0:
                    print(f"[{mode} {k}/{len(todo)}]", flush=True)


if __name__ == "__main__":
    if RUN_SEP:
        run_sep(DEVICE)
    if RUN_ASR:
        run_asr(MODES)
