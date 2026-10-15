#!/usr/bin/env python3
"""Check on the authors' own public demo pages: does switching speech VAD on/off change published
song-generation system comparisons? English songs only.

Sources (authors' demo pages, downloaded as is, no processing):
  levo : https://levo-demo.github.io/   "Song Generation" table: Suno v4.5 / Mureka-O1 / YuE / DiffRhythm / ACE-Step / LeVo
         (table header order matches folders suno/mureka/yue/diffrhythm/acestep/overview, so LeVo = overview)
         paper Table 1 PER: Suno 21.6, Mureka 7.2, YuE 36.4, DiffRhythm 12.3, ACE-Step 37.1, LeVo 7.2
  dr2  : https://aslp-lab.github.io/DiffRhythm2.github.io/  text-prompt table DiffRhythm+ / ACE-Step / LeVo / DiffRhythm 2
         (+ audio-prompt table with DiffRhythm+ / LeVo / DiffRhythm 2 only, recorded as subset=audio)
         paper PER: DiffRhythm+ 0.15, ACE-Step 0.23, LeVo 0.19, DiffRhythm 2 0.13
         Rows hidden inside HTML comments (not visible on the page) are not used.

Two steps (STEPS at the top):
  prep : fetch each page, parse lyrics + per-system mp3 for every row, keep English rows, download politely
         (3 s pause between files) to DEMO_ROOT/<demo>/<system>/<id>.mp3; lyrics without section tags such as
         [verse] go to DEMO_ROOT/<demo>/lyrics/<id>.txt; list in DEMO_ROOT/<demo>/manifest.json. Needs internet.
  run  : transcribe whole songs (not cropped to 120 s) with the paper's settings:
         faster-whisper large-v3, fp16, beam 1, language=en;
         off    = vad_filter=False (P1)
         on     = vad_filter=True, default VadOptions (P0); kept seconds = duration_after_vad
         energy = exp_energy_gate.energy_segments on HDemucs-separated vocals (separation as exp_sep_redecode.py,
                  whole song), original mix + clip_timestamps
         recall / WER / normalisation from verify_metrics.py. Resumable per (demo, subset, system, id, cond).

Writes : ../results/demo_vad.csv (includes the transcript column "text"); summarised by analyze_demo_vad.py.
         Separated vocals cached under DEMO_ROOT/<demo>/sep_vocals/.
Supports the paper's demo-page check (LeVo demo page and DiffRhythm 2 demo page).
Note: demo samples are selected by the authors and include outputs of commercial systems; they are used only
for research evaluation here and the audio is not redistributed.
"""
import csv, html, json, re, subprocess, time
from pathlib import Path

# ===== Edit here =====
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"   # output folder
DEMO_ROOT = Path("data/demo_pages")   # downloaded demo audio + lyrics + manifest.json per demo (created by the prep step)
OUT_NAME = "demo_vad.csv"             # output CSV in RESULTS_DIR
STEPS = ["prep", "run"]               # "prep" downloads (needs internet), "run" transcribes (needs a GPU)
# =====================

DEMOS = {
    "levo": {"url": "https://levo-demo.github.io/",
             "pat": r"static/audio_sample/(suno|mureka|yue|diffrhythm|acestep|overview)/(\w+)\.mp3",
             "marker": "Click to show lyrics",
             "sysname": {"suno": "Suno v4.5", "mureka": "Mureka-O1", "yue": "YuE", "diffrhythm": "DiffRhythm",
                         "acestep": "ACE-Step", "overview": "LeVo"}},
    "dr2": {"url": "https://aslp-lab.github.io/DiffRhythm2.github.io/",
            "pat": r"samples/(text|audio)/(diffrhythmplus|acestep|levo|diffrhythm2)/(\w+)\.mp3",
            "marker": "Lyrics",
            "sysname": {"diffrhythmplus": "DiffRhythm+", "acestep": "ACE-Step", "levo": "LeVo", "diffrhythm2": "DiffRhythm 2"}},
}
TAG = re.compile(r"\[[^\]]*\]")
FIELDS = ["demo", "subset", "system", "id", "cond", "recall", "wer", "n_ref", "n_hyp", "dur", "kept_sec", "n_seg", "sec", "text"]


def page_items(page):
    """After removing HTML comments, yield ('text', line) / ('audio', src) in page order."""
    s = re.sub(r"<!--.*?-->", "", page, flags=re.S)
    s = re.sub(r"<script.*?</script>|<style.*?</style>", "", s, flags=re.S)
    for m in re.finditer(r"<(?:audio|source)[^>]*\ssrc=\"([^\"]+)\"[^>]*>|<[^>]+>|[^<]+", s):
        if m.group(1):
            yield ("audio", m.group(1))
        elif not m.group(0).startswith("<"):
            for ln in html.unescape(m.group(0)).split("\n"):
                ln = ln.strip()
                if ln:
                    yield ("text", ln)


def parse(demo, page):
    cfg = DEMOS[demo]; pat = re.compile(cfg["pat"])
    rows, buf, cur = {}, [], None
    for kind, v in page_items(page):
        if kind == "text":
            buf.append(v); continue
        m = pat.search(v)
        if not m:
            continue
        if cfg["marker"] in buf:                       # a new lyrics row starts
            k = len(buf) - 1 - buf[::-1].index(cfg["marker"])
            cur = "\n".join(buf[k + 1:])
            buf = []
        if demo == "levo":
            subset, folder, sid = "text", m.group(1), m.group(2)
        else:
            subset, folder, sid = m.group(1), m.group(2), m.group(3)
        key = (subset, sid)
        r = rows.setdefault(key, {"subset": subset, "id": sid, "lyrics_raw": cur, "audio": {}})
        assert r["lyrics_raw"] == cur, f"{demo} {key}: lyrics mismatch"
        r["audio"][folder] = v
    return list(rows.values())


def clean_lyrics(t):
    out = []
    for ln in t.split("\n"):
        ln = TAG.sub(" ", ln).strip()
        if ln:
            out.append(ln)
    return "\n".join(out)


def is_english(t):
    letters = [c for c in t if c.isalpha()]
    return bool(letters) and sum(c.isascii() for c in letters) / len(letters) > 0.95


def prep():
    for demo, cfg in DEMOS.items():
        d = DEMO_ROOT / demo; d.mkdir(parents=True, exist_ok=True)
        pg = d / "page.html"
        if not pg.exists():
            subprocess.run(["curl", "-sSfL", "-o", str(pg), cfg["url"]], check=True)
        rows = parse(demo, pg.read_text(encoding="utf-8"))
        man = []
        for r in rows:
            lyr = clean_lyrics(r["lyrics_raw"] or "")
            if not is_english(lyr):
                continue
            uid = f"{r['subset']}_{r['id']}" if demo == "dr2" else r["id"]
            (d / "lyrics").mkdir(exist_ok=True)
            (d / "lyrics" / f"{uid}.txt").write_text(lyr + "\n", encoding="utf-8")
            ent = {"uid": uid, "subset": r["subset"], "page_id": r["id"], "systems": {}}
            for folder, src in sorted(r["audio"].items()):
                dst = d / folder / f"{uid}.mp3"; dst.parent.mkdir(exist_ok=True)
                url = cfg["url"] + src
                if not dst.exists() or dst.stat().st_size == 0:
                    subprocess.run(["curl", "-sSfL", "--retry", "3", "-A", "Mozilla/5.0 (research eval)", "-o", str(dst), url], check=True)
                    time.sleep(3)
                ent["systems"][folder] = {"name": cfg["sysname"][folder], "url": url, "file": str(dst.relative_to(d)),
                                          "bytes": dst.stat().st_size}
            man.append(ent)
            print(demo, uid, sorted(ent["systems"]), flush=True)
        (d / "manifest.json").write_text(json.dumps({"source": cfg["url"], "fetched": time.strftime("%F"),
                                                     "songs": man}, ensure_ascii=False, indent=1), encoding="utf-8")


def run():
    import numpy as np, torch, torchaudio
    import audio_io
    from verify_metrics import norm_words, recall, wer
    from exp_energy_gate import energy_segments
    from faster_whisper import WhisperModel
    root = DEMO_ROOT
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / OUT_NAME
    done = set()
    if out.exists():
        done = {(r["demo"], r["subset"], r["system"], r["id"], r["cond"]) for r in csv.DictReader(open(out))}
    todo = []
    for demo in DEMOS:
        man = json.loads((root / demo / "manifest.json").read_text())
        for s in man["songs"]:
            for folder, e in s["systems"].items():
                todo.append((demo, s["subset"], folder, s["uid"], root / demo / e["file"], root / demo / "lyrics" / f"{s['uid']}.txt"))
    print(f"{len(todo)} songs (x off/on/energy), {len(done)} rows already done", flush=True)
    model = WhisperModel("large-v3", device="cuda", compute_type="float16")
    import sep_util
    sep, sr_m, vi = sep_util.load_hdemucs("cuda")

    def separate(path, vp):                       # as exp_sep_redecode.separate, but the whole song (no 120 s crop)
        return sep_util.separate_cached(path, vp, sep, sr_m, vi, "cuda", None)

    new = not out.exists()
    with open(out, "a", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            wr.writeheader()
        for k, (demo, subset, folder, uid, mp3, lyr) in enumerate(todo, 1):
            conds = [c for c in ("off", "on", "energy") if (demo, subset, folder, uid, c) not in done]
            if not conds:
                continue
            ref = norm_words(lyr.read_text(encoding="utf-8"))
            mix = np.asarray(audio_io.load_mono16k(str(mp3)), dtype=np.float32)
            dur = len(mix) / 16000
            for c in conds:
                t0 = time.time(); kept = dur; nseg = 0
                if c in ("off", "on"):
                    it, info = model.transcribe(mix, language="en", beam_size=1, vad_filter=(c == "on"))
                    segs = list(it); nseg = len(segs)
                    if c == "on":
                        kept = float(getattr(info, "duration_after_vad", 0) or 0)
                    txt = " ".join(x.text for x in segs)
                else:
                    voc = separate(mp3, root / demo / "sep_vocals" / folder / f"{uid}.vocals.wav")
                    assert abs(len(voc) - len(mix)) <= 160, (mp3, len(voc), len(mix))
                    gs = energy_segments(np.asarray(voc, dtype=np.float32)[: len(mix)])
                    kept = sum(e - s for s, e in gs); nseg = len(gs)
                    if gs:
                        it, _ = model.transcribe(mix, language="en", beam_size=1, vad_filter=False,
                                                 clip_timestamps=[x for se in gs for x in se])
                        txt = " ".join(x.text for x in it)
                    else:
                        txt = ""
                hyp = norm_words(txt)
                wr.writerow({"demo": demo, "subset": subset, "system": folder, "id": uid, "cond": c,
                             "recall": round(recall(ref, hyp), 4), "wer": round(wer(ref, hyp), 4),
                             "n_ref": len(ref), "n_hyp": len(hyp), "dur": round(dur, 2), "kept_sec": round(kept, 2),
                             "n_seg": nseg, "sec": round(time.time() - t0, 2), "text": txt.strip()[:6000]})
                f.flush()
            print(f"[{k}/{len(todo)}] {demo}/{subset}/{folder}/{uid}", flush=True)
    print("DEMO_VAD_DONE", flush=True)


if __name__ == "__main__":
    if "prep" in STEPS:
        prep()
    if "run" in STEPS:
        run()
