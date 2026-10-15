#!/usr/bin/env python3
"""Unit test of sep_util.py (CPU): separate one song with the corrected chunking and check
 (a) output duration == input duration (120.0 s);
 (b) for each 20 s block, the best lag (within +-3 s) between the RMS envelope of the vocals and of the mixture is 0;
 (c) agreement of the RMS envelope with htdemucs_ft vocals (assumed aligned: cached by exp_sep2.py).
The vocals cached by the earlier, mis-aligned chunking (if you have them in sep_vocals_old/) are evaluated the same way for comparison.
Output: results/sep_util_unittest.json
"""
# ===== Edit here =====
from pathlib import Path
REPO = Path(__file__).resolve().parent.parent
GEN = REPO / "data" / "audio" / "generated"                  # contains runs/<exp>/<arm>/<song>@s<seed>.wav, sep_vocals_old/, sep_vocals_htdemucs/
SONG = ("vad48_2b", "B", "runs")                              # (experiment, arm, runs sub-directory)
STEM = None                                                   # "<song>@s<seed>"; None = first song of that arm
OUT_JSON = REPO / "results" / "sep_util_unittest.json"
DEVICE = "cpu"
OLD_SUB = "sep_vocals_old"                                    # vocals from the earlier (mis-aligned) chunking, for comparison
# =====================
import json, time
import numpy as np


def env(x, sr=16000, hop=0.05):
    h = int(hop * sr); n = len(x) // h
    return np.sqrt(np.mean(x[: n * h].reshape(n, h) ** 2, axis=1) + 1e-12), hop


def lag_blocks(v, m, sr=16000, block=20.0, maxlag=3.0):
    ev, hop = env(v); em, _ = env(m)
    k = int(block / hop); L = int(maxlag / hop); res = []
    for b0 in range(0, len(em) - k + 1, k):
        a = np.log(em[b0: b0 + k]); best = None
        for d in range(-L, L + 1):           # v is d frames later than m => v[t+d] matches m[t]
            i0, i1 = b0 + d, b0 + d + k
            if i0 < 0 or i1 > len(ev):
                continue
            c = np.corrcoef(a, np.log(ev[i0:i1]))[0, 1]
            if best is None or c > best[1]:
                best = (d, c)
        res.append({"block_start_s": b0 * hop, "lag_s": round(best[0] * hop, 3), "corr": round(float(best[1]), 3)})
    return res


def env_corr(a, b):
    n = min(len(a), len(b)); ea, _ = env(a[:n]); eb, _ = env(b[:n])
    return round(float(np.corrcoef(np.log(ea), np.log(eb))[0, 1]), 4)


def main():
    import audio_io, sep_util
    exp, arm, sub = SONG
    wav = sorted((GEN / sub / exp / arm).glob("*.wav"))[0] if STEM is None else GEN / sub / exp / arm / f"{STEM}.wav"
    stem = wav.stem
    mix = audio_io.load_mono16k(wav, duration=120.0)
    t0 = time.time()
    sep, sr_m, vi = sep_util.load_hdemucs(DEVICE)
    new = sep_util.separate_16k(wav, sep, sr_m, vi, DEVICE, 120.0)
    t_sep = time.time() - t0
    old = audio_io.load_mono16k(GEN / OLD_SUB / exp / arm / f"{stem}.vocals.wav")
    ht = audio_io.load_mono16k(GEN / "sep_vocals_htdemucs" / exp / arm / f"{stem}.vocals.wav")
    r = {"song": f"{exp}/{arm}/{stem}", "sep_cpu_sec": round(t_sep, 1),
         "dur_mix": len(mix) / 16000, "dur_new": len(new) / 16000, "dur_old": len(old) / 16000, "dur_htft": len(ht) / 16000,
         "lag_new_vs_mix": lag_blocks(new, mix), "lag_old_vs_mix": lag_blocks(old, mix), "lag_htft_vs_mix": lag_blocks(ht, mix),
         "envcorr_new_htft": env_corr(new, ht), "envcorr_old_htft": env_corr(old, ht)}
    # the old file is 2 s late after 20 s: compare it as is, and realigned ([20,38) of the old file against [22,40) of htft)
    seg = slice(22 * 16000, 40 * 16000)
    r["envcorr_22_40_old_as_is_htft"] = env_corr(old[seg], ht[seg])
    r["envcorr_22_40_old_realigned_htft"] = env_corr(old[20 * 16000: 38 * 16000], ht[seg])
    r["envcorr_22_40_new_htft"] = env_corr(new[seg], ht[seg])
    assert abs(r["dur_new"] - r["dur_mix"]) < 1e-3, r["dur_new"]
    r["lag_new_vs_htft"] = lag_blocks(new, ht); r["lag_old_vs_htft"] = lag_blocks(old, ht)
    # criterion: blocks with envelope correlation >= 0.4 (the lag of weakly correlated blocks, e.g. a quiet last block, cannot be estimated) must have lag 0; same against htft
    assert all(abs(b["lag_s"]) < 1e-9 for b in r["lag_new_vs_mix"] if b["corr"] >= 0.4), r["lag_new_vs_mix"]
    assert all(abs(b["lag_s"]) < 1e-9 for b in r["lag_new_vs_htft"] if b["corr"] >= 0.4), r["lag_new_vs_htft"]
    Path(OUT_JSON).write_text(json.dumps(r, ensure_ascii=False, indent=1))
    print(json.dumps(r, ensure_ascii=False, indent=1))
    print("UNITTEST_OK")


if __name__ == "__main__":
    main()

