#!/usr/bin/env python3
"""Other ASR toolchains on the same 384 generated songs (ACE-Step, 4 systems x 48 songs x 2 seeds):
does a VAD / no-speech style default in other tools also drop sung vocals?

Each tool is run with its own *defaults* (only language "en" and the model are set), because the
question is "what happens if you use the tool the way its defaults suggest".

  openai    official openai-whisper, large-v3, transcribe() defaults (no VAD; has no_speech /
            compression-ratio / log-prob gates and temperature fallback)
  whisperx  WhisperX, large-v3, load_model() + transcribe() defaults (VAD always on, default pyannote)
  hf        transformers ASR pipeline, openai/whisper-large-v3, chunk_length_s=30 (no VAD; the setup
            used in an earlier evaluation)
  parakeet  NVIDIA Parakeet-TDT (NeMo), a non-Whisper ASR model, defaults
  fw_thr    faster-whisper with VAD on but a lower Silero threshold (0.3 / 0.1); otherwise as exp_vad_gen.py

Reads  : GEN_ROOT/runs/<exp>/<arm>/<song>@s<seed>.wav (first 120 s used)
         META_DIR/<song>.meta.json ("ref_words")
Writes : ../results/tool_<tool>[_<thr>].csv  (tool_openai, tool_whisperx, tool_hf, tool_parakeet,
         tool_fw_thr_0.3, tool_fw_thr_0.1). Columns include the raw transcript ("text").
Supports the paper's toolchain comparison (WhisperX / openai-whisper / transformers / Parakeet,
and the faster-whisper lower-threshold rows).
Each tool was installed in its own virtual environment; edit RUNS to run one tool per environment.
Resumable: rows already present in the output CSV are skipped.
"""
import csv, json, time
from pathlib import Path

from verify_metrics import norm_words, recall, wer

# ===== Edit here =====
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"   # output folder
GEN_ROOT = Path("data/audio/generated")   # contains runs/<exp>/<arm>/<song>@s<seed>.wav (generated songs you produce yourself)
META_DIR = Path("data/lyrics_meta")       # "<song>.meta.json" files, each with "ref_words" (reference lyric words of the 120 s window)
WIN_SEC = 120.0                           # only the first 120 s of every song is transcribed
ARMS = ["vad48_2b/B", "vad48_2b/C", "vad48_xl/B", "vad48_xl/C"]   # <exp>/<arm> folders (4 ACE-Step systems)
# (tool, Silero threshold or None) - the configurations used for the paper
RUNS = [("fw_thr", 0.3), ("fw_thr", 0.1), ("openai", None), ("whisperx", None), ("hf", None), ("parakeet", None)]
LIMIT = 0                                 # >0: smoke test on an evenly spaced subset, written to tool_<tag>_smoke.csv
# =====================

FIELDS = ["tool", "condition", "exp", "arm", "song", "seed", "recall", "wer", "n_ref", "n_hyp", "sec", "text"]


def load16k(path):
    import numpy as np, librosa
    a, _ = librosa.load(str(path), sr=16000, mono=True, duration=WIN_SEC)
    return np.asarray(a, dtype=np.float32)


def make(tool, thr):
    """Return (condition string, function audio16k -> text)."""
    if tool == "openai":
        import whisper
        m = whisper.load_model("large-v3", device="cuda")
        return f"openai-whisper {whisper.__version__} large-v3 defaults", lambda a: m.transcribe(a, language="en")["text"]
    if tool == "whisperx":
        import whisperx
        m = whisperx.load_model("large-v3", "cuda", compute_type="float16", language="en")
        ver = getattr(whisperx, "__version__", "?")
        return f"whisperx {ver} large-v3 defaults (VAD on)", lambda a: " ".join(s["text"] for s in m.transcribe(a, language="en")["segments"])
    if tool == "hf":
        import torch, transformers
        from transformers import pipeline
        p = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3", torch_dtype=torch.float16, device="cuda")
        return (f"transformers {transformers.__version__} pipeline whisper-large-v3 chunk30",
                lambda a: p({"raw": a, "sampling_rate": 16000}, chunk_length_s=30, generate_kwargs={"language": "en", "num_beams": 1})["text"])
    if tool == "parakeet":
        import nemo, nemo.collections.asr as nemo_asr
        name = "nvidia/parakeet-tdt-0.6b-v2"
        m = nemo_asr.models.ASRModel.from_pretrained(name)

        def run(a):
            o = m.transcribe([a], verbose=False)[0]
            return o.text if hasattr(o, "text") else str(o)
        return f"nemo {nemo.__version__} {name} defaults", run
    if tool == "fw_thr":
        import faster_whisper
        from faster_whisper import WhisperModel
        m = WhisperModel("large-v3", device="cuda", compute_type="float16")
        return (f"faster-whisper {faster_whisper.__version__} large-v3 beam1 vad_filter=True threshold={thr}",
                lambda a: " ".join(s.text for s in m.transcribe(a, language="en", beam_size=1, vad_filter=True,
                                                               vad_parameters={"threshold": thr})[0]))
    raise SystemExit(f"unknown tool {tool}")


def run_tool(tool, thr):
    tag = tool + (f"_{thr}" if tool == "fw_thr" else "")
    todo = []
    for ea in ARMS:
        exp, arm = ea.split("/")
        for w in sorted((GEN_ROOT / "runs" / exp / arm).glob("*.wav")):
            song, seed = w.stem.split("@s")
            todo.append((exp, arm, song, seed, w))
    if LIMIT:
        todo = todo[:: max(1, len(todo) // LIMIT)][:LIMIT]
    cond, fn = make(tool, thr)
    print(f"{cond}; {len(todo)} items", flush=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = RESULTS_DIR / (f"tool_{tag}.csv" if not LIMIT else f"tool_{tag}_smoke.csv")
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
            a = load16k(path)
            t0 = time.time()
            try:
                txt = fn(a)
            except Exception as e:
                print(f"FAILED {exp}/{arm} {s}@s{seed}: {type(e).__name__}: {str(e)[:200]}", flush=True)
                continue
            hyp = norm_words(txt)
            wr.writerow({"tool": tool, "condition": cond, "exp": exp, "arm": arm, "song": s, "seed": seed,
                         "recall": round(recall(ref, hyp), 4), "wer": round(wer(ref, hyp), 4), "n_ref": len(ref),
                         "n_hyp": len(hyp), "sec": round(time.time() - t0, 2), "text": txt.strip()[:4000]})
            f.flush()
            if k % 20 == 0 or k == len(todo):
                print(f"[{k}/{len(todo)}]", flush=True)
    print(f"wrote {out_csv}")


if __name__ == "__main__":
    for tool, thr in RUNS:
        run_tool(tool, thr)
