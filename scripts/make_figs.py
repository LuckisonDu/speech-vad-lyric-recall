#!/usr/bin/env python3
"""Paper figures, drawn only from numeric files in ../results.

Reads  : ../results/vad_gen48_summary.json (Fig. 1), vad_stats48.csv, vad_stats48_heart_levo.csv,
         vad_gen48.csv, vad_gen48_heart.csv, vad_gen48_levo.csv (Fig. 2),
         vad_sweep48_summary.json (Fig. 4), multiverse_summary.json (Fig. 5)
Writes : FIG_DIR/fig1_flip.pdf, fig2_kept.pdf, fig4_sweep.pdf, fig5_multiverse.pdf
Fig. 1: mean recall of the four ACE-Step systems with VAD off / on (ranking flip).
Fig. 2: VAD kept seconds vs recall with VAD on (left) / off (right), three families.
Fig. 4: threshold sweep, median seconds kept on the mixture and on separated vocals.
Fig. 5: evaluation multiverse, verdict per system pair x setting.
(The per-song VAD timeline example figure is not included: it needs audio-derived data.)
"""
import csv, json
from collections import defaultdict
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

# ===== Edit here =====
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"   # folder holding the result CSV/JSON files
FIG_DIR = Path(__file__).resolve().parent.parent / "figs"          # output folder for the PDF figures
# =====================

R = RESULTS_DIR; F = FIG_DIR; F.mkdir(exist_ok=True)
plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.linewidth": 0.6, "pdf.fonttype": 42})
SYS = {("vad48_2b", "B"): "ACE 2B, no planner", ("vad48_2b", "C"): "ACE 2B, planner", ("vad48_xl", "B"): "ACE XL, no planner",
       ("vad48_xl", "C"): "ACE XL, planner"}
COL = {"ACE 2B, no planner": "#4C78A8", "ACE 2B, planner": "#72B7B2", "ACE XL, no planner": "#E45756", "ACE XL, planner": "#F58518"}

# Fig. 1: mean recall of the four systems under the two settings (dumbbell plot), shows the ranking flip
g = json.loads((R / "vad_gen48_summary.json").read_text())["per_system"]
key = {"ACE 2B, no planner": "2B·B", "ACE 2B, planner": "2B·C", "ACE XL, no planner": "XL·B", "ACE XL, planner": "XL·C"}
fig, ax = plt.subplots(figsize=(3.3, 1.9))
for i, (lab, s) in enumerate(key.items()):
    a, b = g[s]["recall_off_mean"], g[s]["recall_on_mean"]
    ax.plot([0, 1], [a, b], "-o", color=COL[lab], ms=4, lw=1.4, label=lab)
ax.set_xticks([0, 1]); ax.set_xticklabels(["VAD off", "VAD on (default threshold)"]); ax.set_xlim(-0.25, 1.25)
ax.set_ylabel("mean lyric recall"); ax.set_ylim(0, 0.9); ax.grid(axis="y", lw=0.3, alpha=0.5)
ax.legend(frameon=False, fontsize=6.5, loc="lower left")
fig.tight_layout(pad=0.3); fig.savefig(F / "fig1_flip.pdf"); plt.close(fig)

# Fig. 2: VAD kept seconds vs recall with VAD on (left) and off (right); three families, large dots = system means
SYS2 = dict(SYS); SYS2[("vad48_heart", "H")] = "HeartMuLa"; SYS2[("vad48_levo", "L")] = "LeVo"
COL2 = dict(COL); COL2["HeartMuLa"] = "#54A24B"; COL2["LeVo"] = "#B279A2"
kept = {}
for fn in ("vad_stats48.csv", "vad_stats48_heart_levo.csv"):
    for r in csv.DictReader(open(R / fn)):
        if r["src"] == "gen":
            kept[(r["exp"], r["arm"], r["song"], r["seed"])] = float(r["kept_sec"])
rec = defaultdict(dict)
for fn in ("vad_gen48.csv", "vad_gen48_heart.csv", "vad_gen48_levo.csv"):
    for r in csv.DictReader(open(R / fn)):
        if r["src"] == "gen":
            rec[(r["exp"], r["arm"], r["song"], r["seed"])][r["vad"]] = float(r["recall"])
fig, axs = plt.subplots(1, 2, figsize=(3.3, 2.35), sharex=True, sharey=True)
for k, v in rec.items():
    lab = SYS2[k[:2]]
    for ax, vad in zip(axs, ("True", "False")):
        ax.scatter(kept[k], v[vad], s=2.5, color=COL2[lab], alpha=0.45, lw=0)
for (e, a_), lab in SYS2.items():
    ks = [k for k in rec if k[:2] == (e, a_)]
    mk = sum(kept[k] for k in ks) / len(ks)
    for ax, vad in zip(axs, ("True", "False")):
        ax.scatter(mk, sum(rec[k][vad] for k in ks) / len(ks), s=26, color=COL2[lab], edgecolor="k", lw=0.6, zorder=5, label=lab if vad == "True" else None)
for ax, t in zip(axs, ("recall with VAD on", "recall with VAD off")):
    ax.set_title(t, fontsize=7.5); ax.set_xlim(-3, 123); ax.set_ylim(-0.03, 1.03); ax.grid(lw=0.3, alpha=0.5)
axs[0].set_ylabel("lyric recall")
fig.legend(loc="lower center", ncol=3, fontsize=5.6, frameon=False, handletextpad=0.2, columnspacing=0.8, bbox_to_anchor=(0.55, 0.0))
fig.text(0.55, 0.165, "seconds kept by the VAD (of 120 s)", ha="center", fontsize=7.5)
fig.tight_layout(pad=0.3, rect=(0.02, 0.2, 1, 1)); fig.savefig(F / "fig2_kept.pdf"); plt.close(fig)

# Fig. 4: threshold sweep, median seconds kept on the mixture and on separated vocals
sw = json.loads((R / "vad_sweep48_summary.json").read_text())
thr = ["0.1", "0.2", "0.3", "0.4", "0.5"]
fig, axs = plt.subplots(1, 2, figsize=(3.3, 1.6), sharey=True)
for lab, s in key.items():
    axs[0].plot([float(t) for t in thr], [sw[s]["mix_kept_median"][t] for t in thr], "-o", ms=2.5, lw=1, color=COL[lab], label=lab)
    axs[1].plot([float(t) for t in thr], [sw[s]["vocals_kept_median"][t] for t in thr], "-o", ms=2.5, lw=1, color=COL[lab])
for ax, t in zip(axs, ("mixture", "separated vocals")):
    ax.set_title(t, fontsize=7.5); ax.set_xticks([0.1, 0.3, 0.5]); ax.grid(lw=0.3, alpha=0.5); ax.set_ylim(0, 95)
axs[0].set_ylabel("median seconds kept"); fig.text(0.55, 0.01, "VAD threshold", ha="center", fontsize=7.5)
axs[0].legend(frameon=False, fontsize=5.5, loc="upper right")
fig.tight_layout(pad=0.3, rect=(0, 0.06, 1, 1)); fig.savefig(F / "fig4_sweep.pdf"); plt.close(fig)

# Fig. 5: evaluation multiverse, verdict for 6 system pairs x 20 settings (A better / n.s. / B better)
mv = json.loads((R / "multiverse_summary.json").read_text())
pipes = mv["pipelines_no_vad"] + mv["pipelines_vad"]
cols = [(p, m) for m in ("recall", "wer") for p in pipes]
cols = [c for c in cols if c[0] in mv["pipelines_no_vad"]] + [c for c in cols if c[0] in mv["pipelines_vad"]]
pairs = list(mv["pairs"].keys())
M = np.zeros((len(pairs), len(cols)))
for c in mv["cells"]:
    i = pairs.index(c["pair"]); j = cols.index((c["pipe"], c["metric"]))
    M[i, j] = 1 if c["verdict"] == c["a"] else (-1 if c["verdict"] == c["b"] else 0)
fig, ax = plt.subplots(figsize=(3.3, 1.75))
ax.imshow(M, cmap=ListedColormap(["#E45756", "#E8E8E8", "#4C78A8"]), vmin=-1, vmax=1, aspect="auto")
nv = sum(1 for c in cols if c[0] in mv["pipelines_no_vad"])
ax.axvline(nv - 0.5, color="k", lw=1.2)
short = {"fw VAD off": "fw off", "fw off + check": "fw+chk", "all vocals": "vocals", "openai-whisper": "openai", "transformers": "HF",
         "Parakeet": "Parakeet", "fw VAD 0.5": "fw .5", "fw VAD 0.3": "fw .3", "fw VAD 0.1": "fw .1", "WhisperX": "WhX"}
ax.set_xticks(range(len(cols))); ax.set_xticklabels([f"{short[p]}{'' if m == 'recall' else '*'}" for p, m in cols], rotation=90, fontsize=5)
ax.set_yticks(range(len(pairs))); ax.set_yticklabels([p.replace("+pl", "+P").replace(" vs ", " vs ") for p in pairs], fontsize=6)
ax.set_xticks(np.arange(-.5, len(cols), 1), minor=True); ax.set_yticks(np.arange(-.5, len(pairs), 1), minor=True)
ax.grid(which="minor", color="w", lw=0.6); ax.tick_params(which="minor", length=0)
ax.text(nv / 2 - 0.5, -0.9, "no VAD", ha="center", fontsize=6.5); ax.text(nv + (len(cols) - nv) / 2 - 0.5, -0.9, "with a VAD", ha="center", fontsize=6.5)
fig.tight_layout(pad=0.2); fig.savefig(F / "fig5_multiverse.pdf"); plt.close(fig)
print("figs ->", sorted(p.name for p in F.glob("*.pdf")))
