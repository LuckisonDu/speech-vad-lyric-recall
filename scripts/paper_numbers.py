#!/usr/bin/env python3
"""Recompute every number in the paper from the result files and write them as LaTeX macros.

Reads:  RESULTS_DIR/*.json and a few per-song CSVs (see the J("...") and open(RESULTS_DIR / ...) calls).
Writes: OUT_DIR/numbers.tex (one \\newcommand{\\N<Name>}{value} per number, included by main.tex),
        OUT_DIR/supp_s5_rows.tex and OUT_DIR/supp_s8_rho_rows.tex (supplement table rows).
Macro names contain letters only (LaTeX macro names cannot contain digits).
Usage:  python3 paper_numbers.py
"""
import csv, json, random, statistics as st
from collections import defaultdict
from pathlib import Path

# ===== Edit here =====
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"   # folder with the summary JSONs and per-song CSVs
OUT_DIR = Path(__file__).resolve().parent.parent / "paper"          # where numbers.tex and the supplement rows are written
# =====================
OUT_DIR.mkdir(parents=True, exist_ok=True)
R = RESULTS_DIR
HERE = OUT_DIR
J = lambda f: json.loads((R / f).read_text())
out = []


def put(name, val, fmt="{:.3f}"):
    s = fmt.format(val) if isinstance(val, (int, float)) and fmt else str(val)
    out.append(f"\\newcommand{{\\N{name}}}{{{s}}}")


def sgn(x, fmt="{:+.3f}"):
    return fmt.format(x).replace("-", "$-$").replace("+", "$+$")


def pv(p):
    s = f"{p:.2g}"
    if "e" in s:
        b, e = s.split("e"); return f"${b}\\times10^{{{int(e)}}}$"
    return f"${s}$"


def ci(c, fmt="{:+.3f}"):
    return f"[{sgn(c[0], fmt)}, {sgn(c[1], fmt)}]"


SYS = {"2B·B": "TwoBB", "2B·C": "TwoBC", "XL·B": "XLB", "XL·C": "XLC", "orig": "Orig"}

# ---------- Main experiment: VAD off/on, ACE-Step 2B/XL with/without planner (analyze_vad_gen48.py) ----------
g = J("vad_gen48_summary.json")
for s, m in SYS.items():
    v = g["per_system"][s]
    put(f"{m}OffMean", v["recall_off_mean"]); put(f"{m}OnMean", v["recall_on_mean"])
    put(f"{m}OffMed", v["recall_off_median"]); put(f"{m}OnMed", v["recall_on_median"])
    put(f"{m}BrokenOn", v["broken_by_vad_on"], "{}"); put(f"{m}BrokenOff", v["broken_by_vad_off"], "{}")
    put(f"{m}ZeroOn", v["zero_words_on"], "{}"); put(f"{m}N", v["n"], "{}")
    put(f"{m}Keep", 100 * v["keep_median"], "{:.1f}")
c = g["compare"]
for key, m in (("XL·B − 2B·B", "XLBvTwoBB"), ("XL·C − 2B·B", "XLCvTwoBB"), ("XL·B − 2B·C", "XLBvTwoBC"),
               ("XL·C − 2B·C", "XLCvTwoBC"), ("2B·C − 2B·B", "TwoBCvTwoBB"), ("XL·C − XL·B", "XLCvXLB")):
    x = c[key]
    put(f"{m}OffD", sgn(x["vad_off"]["diff_mean"]), None); put(f"{m}OffCI", ci(x["vad_off"]["ci95"]), None)
    put(f"{m}OnD", sgn(x["vad_on"]["diff_mean"]), None); put(f"{m}OnCI", ci(x["vad_on"]["ci95"]), None)
    put(f"{m}OffP", pv(x['vad_off']['wilcoxon_p_by_song']), None)
    put(f"{m}OnP", pv(x['vad_on']['wilcoxon_p_by_song']), None)
    put(f"{m}OffWins", f"{x['vad_off']['b_wins']}/{x['vad_off']['a_wins']}", None)
    put(f"{m}OnWins", f"{x['vad_on']['b_wins']}/{x['vad_on']['a_wins']}", None)
    put(f"{m}IntD", sgn(x["interaction_on_minus_off"]["diff_mean"]), None); put(f"{m}IntCI", ci(x["interaction_on_minus_off"]["ci95"]), None)
    put(f"{m}Flips", x["winner_flips"], "{}"); put(f"{m}Pairs", x["pairs"], "{}")
tot = sum(g["per_system"][s]["broken_by_vad_on"] for s in ("2B·B", "2B·C", "XL·B", "XL·C"))
put("GenBrokenOnTotal", tot, "{}")

# ---------- Mechanism: retained seconds, threshold sweep, vocal ratio (analyze_vad_stats.py, analyze_vad_sweep.py, analyze_toolchains.py) ----------
k = J("vad_stats48_summary.json")
for s, m in SYS.items():
    v = k["per_system"][s]
    put(f"{m}KeptMed", v["kept_sec_median"], "{:.1f}"); put(f"{m}KeptZero", v["kept_0s"], "{}"); put(f"{m}KeptLtTen", v["kept_lt_10s"], "{}")
put("RhoKeptRecallOn", k["gen_spearman_keptsec_vs_recall_on"]); put("RhoKeptRecallOff", k["gen_spearman_keptsec_vs_recall_off"])
put("RhoKeptWordKeep", k["gen_spearman_keptsec_vs_wordkeep"])
put("XLBKeptLess", -k["xlB_minus_2bB_kept_sec"]["mean"], "{:.1f}"); put("XLBKeptLessPairs", k["xlB_minus_2bB_kept_sec"]["xl_less"], "{}")
sw = J("vad_sweep48_summary.json")
for s, m in SYS.items():
    for t in ("0.1", "0.3", "0.5"):
        tt = {"0.1": "One", "0.3": "Three", "0.5": "Five"}[t]
        put(f"{m}MixKept{tt}", sw[s]["mix_kept_median"][t], "{:.1f}")
        if sw[s]["vocals_kept_median"][t] is not None:
            put(f"{m}VocKept{tt}", sw[s]["vocals_kept_median"][t], "{:.1f}")
tc = J("toolchains_summary.json")
for s, m in (("2B·B", "TwoBB"), ("2B·C", "TwoBC"), ("XL·B", "XLB"), ("XL·C", "XLC")):
    put(f"{m}VR", tc["vocal_ratio_db_median"][s]["whole"], "{:.1f}")
put("RhoVRKept", tc["spearman_vocalratio_vs_keptsec"])

# ---------- Other recognizers / toolchains (analyze_toolchains.py) ----------
TOOLS = {"faster-whisper VAD off": "FwOff", "faster-whisper VAD on (threshold 0.5)": "FwOnFive", "faster-whisper VAD on (threshold 0.3)": "FwOnThree",
         "faster-whisper VAD on (threshold 0.1)": "FwOnOne", "openai-whisper default": "Oai", "WhisperX default (VAD always on)": "Wx",
         "transformers pipeline": "Hf", "Parakeet-TDT (non-Whisper)": "Pk"}
for key, m in TOOLS.items():
    v = tc[key]
    for s, sm in (("2B·B", "TwoBB"), ("2B·C", "TwoBC"), ("XL·B", "XLB"), ("XL·C", "XLC")):
        put(f"{m}{sm}", v["mean"][s])
    put(f"{m}D", sgn(v["XL·B−2B·B"]["mean"]), None); put(f"{m}CI", ci(v["XL·B−2B·B"]["ci95"]), None)
    put(f"{m}Rank", " $>$ ".join({"2B·B": "2B", "2B·C": "2B+planner", "XL·B": "XL", "XL·C": "XL+planner"}[x] for x in v["ranking"]), None)

# ---------- WER / PER instead of recall (analyze_wer_per.py) ----------
wp = J("wer_per_summary.json")
for s, m in (("2B·B", "TwoBB"), ("2B·C", "TwoBC"), ("XL·B", "XLB"), ("XL·C", "XLC"), ("HeartMuLa", "Heart"), ("LeVo", "Levo")):
    v = wp["per_system"][s]
    for mm in ("wer", "per"):
        put(f"{m}{mm.upper()}Off", v[f"{mm}_off"]); put(f"{m}{mm.upper()}On", v[f"{mm}_on"])
x = wp["compare"]["XL·B − 2B·B"]
for mm in ("wer", "per"):
    for tag in ("off", "on"):
        put(f"XLBvTwoBB{mm.upper()}{tag.capitalize()}D", sgn(x[f"{mm}_{tag}"]["diff_mean"]), None)
        put(f"XLBvTwoBB{mm.upper()}{tag.capitalize()}CI", ci(x[f"{mm}_{tag}"]["ci95"]), None)

# ---------- Second generator family (HeartMuLa), from per-song CSV ----------
p = defaultdict(dict)
for r in csv.DictReader(open(R / "vad_gen48_heart.csv")):
    p[(r["src"], r["song"], r["seed"])][r["vad"]] = r
hk = [kk for kk in p if kk[0] == "gen" and len(p[kk]) == 2]
off = [float(p[kk]["False"]["recall"]) for kk in hk]; on = [float(p[kk]["True"]["recall"]) for kk in hk]
put("HeartN", len(hk), "{}"); put("HeartOffMean", st.mean(off)); put("HeartOnMean", st.mean(on))
put("HeartOffMed", st.median(off)); put("HeartOnMed", st.median(on))
put("HeartBrokenOn", sum(a >= .6 and b < .3 for a, b in zip(off, on)), "{}"); put("HeartWorse", sum(b < a for a, b in zip(off, on)), "{}")

# ---------- Cross-family comparison (analyze_cross_family.py) ----------
cf = J("cross_family_summary.json")
for s, m in (("HeartMuLa", "Heart"), ("LeVo", "Levo"), ("2B·B", "TwoBB"), ("2B·C", "TwoBC"), ("XL·B", "XLB"), ("XL·C", "XLC")):
    v = cf["per_system"][s]
    put(f"CF{m}KeptMean", v["kept_sec_mean"], "{:.1f}"); put(f"CF{m}KeptMed", v["kept_sec_median"], "{:.1f}")
    put(f"CF{m}LowOn", v["below03_on"], "{}")
    if s in ("HeartMuLa", "LeVo"):
        put(f"{m}CFOffMean", v["recall_off_mean"]); put(f"{m}CFOnMean", v["recall_on_mean"])
        put(f"{m}CFBrokenOn", v["broken_on"], "{}"); put(f"{m}CFWorse", v["worse_on"], "{}"); put(f"{m}CFN", v["n"], "{}")
        put(f"{m}PTwoMean", v["P2_mean"]); put(f"{m}AllVocMean", v["all_vocals_mean"])
        put(f"{m}Flagged", v["flagged"], "{}"); put(f"{m}Rescued", v["rescued"], "{}")
        for t in ("P0", "P1", "P2"):
            put(f"{m}{ {'P0': 'PZero', 'P1': 'POne', 'P2': 'PTwo'}[t] }Low", v["below03"][t], "{}")
for key, m in (("LeVo − HeartMuLa", "LvH"), ("LeVo − 2B·C", "LvTwoBC"), ("LeVo − XL·B", "LvXLB"), ("LeVo − XL·C", "LvXLC"), ("LeVo − 2B·B", "LvTwoBB")):
    for mm, v in cf["compare"][key].items():
        tag = "".join(x.capitalize() for x in mm.split("_"))   # recall_off -> RecallOff
        put(f"{m}{tag}D", sgn(v["diff_mean"]), None); put(f"{m}{tag}CI", ci(v["ci95"]), None)
        put(f"{m}{tag}P", pv(v["wilcoxon_p_by_song"]), None)
put("CFRhoKeptOn", cf["spearman_kept_vs_recall_on_all"]); put("CFRhoKeptOff", cf["spearman_kept_vs_recall_off_all"])
put("CFNItems", cf["n_items_all"], "{}")
put("LvHKeptMore", cf["levo_minus_heart_kept"]["mean"], "{:.1f}"); put("LvHKeptMorePairs", cf["levo_minus_heart_kept"]["levo_more"], "{}")

# ---------- Public benchmarks Jam-ALT / MUSDB-ALT, two machines (exp_vad_bench.py) ----------
bm = J("vad_bench_summary.json")
for b, m in (("jamalt", "Jam"), ("musdbalt", "Mus"), ("all", "Bench")):
    v = bm[b]
    put(f"{m}N", v["n"], "{}"); put(f"{m}OffMed", v["recall_off_median"]); put(f"{m}OnMed", v["recall_on_median"])
    put(f"{m}OffMean", v["recall_off_mean"]); put(f"{m}OnMean", v["recall_on_mean"]); put(f"{m}Zero", v["zero_on"], "{}")
    put(f"{m}Broken", v["broken_on"], "{}"); put(f"{m}D", sgn(v["diff_mean"]), None); put(f"{m}CI", ci(v["ci95"]), None)
    put(f"{m}WerOff", v["wer_off_mean"]); put(f"{m}WerOn", v["wer_on_mean"]); put(f"{m}DurMed", v["dur_median"], "{:.0f}")
bv = defaultdict(dict)
for r in csv.DictReader(open(R / "vad_bench_rerun.csv")):
    bv[r["id"]][r["vad"]] = r
o2 = [float(v["False"]["recall"]) for v in bv.values()]; n2 = [float(v["True"]["recall"]) for v in bv.values()]
put("BenchTwoOffMed", st.median(o2)); put("BenchTwoOnMed", st.median(n2))
put("BenchTwoZero", sum(int(v["True"]["n_hyp"]) == 0 for v in bv.values()), "{}")
put("BenchTwoBroken", sum(a >= .6 and b < .3 for a, b in zip(o2, n2)), "{}")

# ---------- Fix pipeline P0/P1/P2 (analyze_p2.py) ----------
pp = J("p2_summary.json")["per_system"]
a = pp["all"]
put("PZeroMean", a["P0_mean"]); put("POneMean", a["P1_mean"]); put("PTwoMean", a["P2_mean"])
put("PZeroLow", a["below_0.3"]["P0"], "{}"); put("POneLow", a["below_0.3"]["P1"], "{}"); put("PTwoLow", a["below_0.3"]["P2"], "{}")
put("Flagged", a["flagged"], "{}"); put("Rescued", a["rescued"], "{}"); put("FlaggedPOne", a["flagged_recall_P1_mean"]); put("FlaggedPTwo", a["flagged_recall_P2_mean"])
put("AllVocMean", a["all_vocals_mean"]); put("UnflagVocD", sgn(a["unflagged_voc_minus_mix_mean"]), None)
put("UnflagVocWorse", a["unflagged_voc_worse"], "{}"); put("UnflagVocBetter", a["unflagged_voc_better"], "{}")
put("XLFlagged", pp["XL·B"]["flagged"] + pp["XL·C"]["flagged"], "{}"); put("XLRescued", pp["XL·B"]["rescued"] + pp["XL·C"]["rescued"], "{}")
put("TwoBFlagged", pp["2B·B"]["flagged"] + pp["2B·C"]["flagged"], "{}"); put("TwoBRescued", pp["2B·B"]["rescued"] + pp["2B·C"]["rescued"], "{}")
put("FlaggedPct", 100 * a["flagged"] / a["n"], "{:.0f}")

# ---------- Sample size / seeds (analyze_sampling.py) ----------
sp = J("sampling_summary.json")
W = {5: "Five", 10: "Ten", 20: "Twenty", 30: "Thirty"}
for vad in ("vad_off", "vad_on"):
    for sd, st_ in (("one_seed", "One"), ("two_seeds", "Two")):
        for n, w in W.items():
            v = 100 * sp["p_positive"][f"{vad}_{sd}_n{n}"]
            put(f"Samp{'Off' if vad == 'vad_off' else 'On'}{st_}{w}", "0" if v == 0 else f"{v:.1f}", None)   # write "0" (not "<0.1") when no resample was positive
put("SeedDisagreeOff", sp["vad_off_seed_disagree"], "{}")
# ---------- (WER/PER rows for all systems are produced above) ----------
# ---------- Controlled remix of separated vocals + residual (exp_remix.py) ----------
rm = J("remix48_summary.json")
GW = {"0.0": "Zero", "0.25": "Quarter", "0.5": "Half", "1.0": "Full", "2.0": "Double"}
for s, m in (("2B·B", "TwoBB"), ("2B·C", "TwoBC"), ("XL·B", "XLB"), ("XL·C", "XLC")):
    for g, w in GW.items():
        put(f"Remix{m}{w}", rm["median_kept"][s][g], "{:.1f}")
for g, w in GW.items():
    put(f"RemixGap{w}", sgn(rm["xlB_minus_2bB_mean"][g], "{:+.1f}"), None)
# ---------- Listening test (aggregated summary only) ----------
LISTEN = "listen_final.json"   # aggregated listening-test results (no per-rater answers)
ls = J(LISTEN)
put("LRaters", ls["n_raters"], "{}"); put("LItems", ls["n_items"], "{}"); put("LAlpha", ls["alpha_ordinal"])
put("LClear", ls["QA"]["all"]["n_clear"], "{}")
for t, m in (("P0", "PZero"), ("P1", "POne"), ("P2", "PTwo")):
    put(f"LClearLow{m}", ls["QA"]["all"][f"clear_but_{t}_lt0.3"], "{}")
    put(f"LRho{m}", ls["QB"]["spearman"][t])
    put(f"LAuc{m}", ls["QB"]["auc_clear_vs_not"][t])
    put(f"LSTwoClearLow{m}", ls["QA"]["S2"][f"clear_but_{t}_lt0.3"], "{}")
put("LSTwoN", ls["QA"]["S2"]["n"], "{}"); put("LSTwoClear", ls["QA"]["S2"]["n_clear"], "{}")
put("LSTwoUnclear", ls["QA"]["S2"]["n"] - ls["QA"]["S2"]["n_clear"], "{}")
put("LSOneN", ls["QA"]["S1"]["n"], "{}"); put("LSOneClear", ls["QA"]["S1"]["n_clear"], "{}")
for k, m in (("P2-P0", "PTwoPZero"), ("P1-P0", "POnePZero"), ("P2-P1", "PTwoPOne")):
    v = ls["QB"]["diff_ci95"][k]; put(f"LD{m}", sgn(v["diff"]), None); put(f"LD{m}CI", ci(v["ci"]), None)
rs = ls["reasons"]; put("LWhyN", sum(rs.values()), "{}")
for k, m in (("slurred", "Slurred"), ("masked", "Masked"), ("none", "None"), ("other", "Other")):
    put(f"LWhy{m}", rs.get(k, 0), "{}")
fc = ls["flag_check"]; put("LFlagged", fc["n_flagged"], "{}"); put("LFlaggedClear", fc["flagged_but_clear"], "{}")
put("LFlaggedClearStillLow", fc["flagged_after_P2_still_low_but_clear"], "{}")

# ---------- Evaluation-pipeline multiverse (analyze_multiverse.py) ----------
mv = J("multiverse_summary.json")
put("MVN", mv["n_universes"], "{}"); put("MVDistinct", mv["distinct_rankings"], "{}")
put("MVDistinctNoVad", mv["distinct_rankings_no_vad"], "{}"); put("MVDistinctVad", mv["distinct_rankings_vad"], "{}")
put("MVPairsRev", mv["pairs_reversed_significantly"], "{}"); put("MVPairsRevNoVad", mv["pairs_reversed_significantly_no_vad"], "{}")
put("MVNoVadN", len(mv["pipelines_no_vad"]) * 2, "{}"); put("MVVadN", len(mv["pipelines_vad"]) * 2, "{}")
put("MVTopTwoBC", mv["top_system_counts"].get("2B+pl", 0), "{}")
pp = mv["pairs"]
x = pp["2B vs XL"]["all"]; put("MVXLWins", x["XL"], "{}"); put("MVTwoBWins", x["2B"], "{}"); put("MVTwoBXLns", x["n.s."], "{}")
x = pp["2B+pl vs XL+pl"]; put("MVPlVadSig", x["vad"]["2B+pl"], "{}"); put("MVPlNoVadSig", x["no_vad"]["2B+pl"], "{}")
put("MVPairsMulti", sum(1 for v in pp.values() if sum(1 for k, c in v["all"].items() if k != "n" and c > 0) >= 2), "{}")

# ---------- Mediation / decomposition (analyze_mediation.py) ----------
md = J("mediation_summary.json")
put("MedRTwo", md["r2"], "{:.2f}"); put("MedRTwoSys", md["r2_with_system_dummies"], "{:.2f}"); put("MedRTwoNoKept", md["r2_system_and_off_without_kept"], "{:.2f}")
for key, m in (("XL − 2B", "XLvTwoBB"), ("LeVo − HeartMuLa", "LvH")):
    v = md["pairs"][key]
    put(f"Med{m}Gap", sgn(v["observed_on_gap"]), None); put(f"Med{m}Cov", sgn(v["coverage_part"]), None); put(f"Med{m}CovCI", ci(v["coverage_ci95"]), None)
    put(f"Med{m}Intr", sgn(v["intrinsic_part"]), None); put(f"Med{m}IntrCI", ci(v["intrinsic_ci95"]), None)
    put(f"Med{m}Res", sgn(v["residual"]), None); put(f"Med{m}ResCI", ci(v["residual_ci95"]), None)
    put(f"Med{m}Share", 100 * v["coverage_share_of_gap"], "{:.0f}")
sh = [v["coverage_share_of_gap"] for v in md["pairs"].values()]
put("MedShareMin", 100 * min(sh), "{:.0f}"); put("MedShareMax", 100 * max(sh), "{:.0f}"); put("MedNPairs", len(sh), "{}")

# ---------- Reversal criterion (analyze_criterion.py) ----------
cr = J("criterion_summary.json")
put("CritN", cr["all"]["n"], "{}"); put("CritOK", cr["all"]["sign_correct"], "{}")
put("CritRevObs", cr["all"]["reversals_observed"], "{}"); put("CritRevPred", cr["all"]["reversals_predicted"], "{}")
put("CritFiveN", cr["default_threshold_six_systems"]["n"], "{}"); put("CritFiveOK", cr["default_threshold_six_systems"]["sign_correct"], "{}")
put("CritFiveRev", cr["default_threshold_six_systems"]["reversals_observed"], "{}")
ms = cr["misses"][0] if cr["misses"] else None
if ms:
    put("CritMissObs", sgn(ms["on_diff_observed"]), None)

# ---------- Preregistered test of the criterion (criterion_prereg_result.json) ----------
pr = J("criterion_prereg_result.json")
put("PreHits", pr["hits"], "{}"); put("PreN", pr["n"], "{}"); put("PreHardHits", pr["hard_hits"], "{}"); put("PreHardN", pr["hard_n"], "{}")
pr_rev_pred = sum(r["predicted_reversal"] for r in pr["rows"]); pr_rev_both = sum(r["predicted_reversal"] and r["observed_reversal"] for r in pr["rows"])
put("PreRevPred", pr_rev_pred, "{}"); put("PreRevBoth", pr_rev_both, "{}")
pr_miss = [r for r in pr["rows"] if not r["hit"]]; assert len(pr_miss) == 1 and pr_miss[0]["observed_reversal"]
put("PreMissObs", sgn(pr_miss[0]["observed_on_diff"]), None)

# ---------- Consistency of remix g=1 with the original mix (remix48.csv vs vad_stats48.csv, per song) ----------
_vs = {(r["exp"], r["arm"], r["song"], r["seed"]): float(r["kept_sec"]) for r in csv.DictReader(open(R / "vad_stats48.csv")) if r["src"] == "gen"}
_rm = [(float(r["kept_g1.0"]), _vs[(r["exp"], r["arm"], r["song"], r["seed"])]) for r in csv.DictReader(open(R / "remix48.csv"))]
put("RemixSame", sum(abs(a - b) < 0.01 for a, b in _rm), "{}"); put("RemixN", len(_rm), "{}")
put("RemixMaxDiff", max(abs(a - b) for a, b in _rm), "{:.1f}")

# ---------- Listening-test reasons by model size (descriptive) ----------
rb = J("listen_final_reasons_by_size.json")
for sz, m in (("2B", "TwoB"), ("XL", "XL")):
    put(f"LWhy{m}Masked", rb[sz]["masked"], "{}"); put(f"LWhy{m}NotMost", rb[sz]["not_most"], "{}")
    put(f"LWhy{m}MaskedPct", 100 * rb[sz]["masked_share_of_all"], "{:.0f}")

# ---------- WhisperX across families (exploratory) ----------
wxs = J("whisperx_families_summary.json")
put("WxLevoRankOff", wxs["rank_vad_off"].index("LeVo") + 1, "{}"); put("WxLevoRankWx", wxs["rank_whisperx"].index("LeVo") + 1, "{}")
v = wxs["pairs"]["LeVo − 2B+pl"]
put("WxLvTwoBCD", sgn(v["whisperx"]["diff"]), None); put("WxLvTwoBCCI", ci(v["whisperx"]["ci95"]), None); put("WxLvTwoBCOffD", sgn(v["vad_off"]["diff"]), None)
v = wxs["pairs"]["LeVo − HeartMuLa"]
put("WxLvHD", sgn(v["whisperx"]["diff"]), None); put("WxLvHCI", ci(v["whisperx"]["ci95"]), None)

# ---------- Listening test, leave-one-rater-out robustness ----------
lo = J("listen_final_leave_one_out.json"); lo_ = [v for k, v in lo.items() if k != "None"]
put("LooSTwoMin", min(v["s2_clear"] for v in lo_), "{}"); put("LooSTwoMax", max(v["s2_clear"] for v in lo_), "{}")
put("LooLowMin", min(v["clear_p0_low"] for v in lo_), "{}"); put("LooLowMax", max(v["clear_p0_low"] for v in lo_), "{}")

# ---------- Batched pipeline defaults (exp_batched.py) and same-machine rerun ----------
bs_ = J("batched_summary.json"); v = bs_["pairs"]["XL − 2B"]
put("BtPlainD", sgn(v["plain_default"][0]), None); put("BtPlainCI", ci(v["plain_default"][1:]), None)
put("BtBatchD", sgn(v["batched_default"][0]), None); put("BtBatchCI", ci(v["batched_default"][1:]), None)
put("BtXLKept", bs_["kept_median_batched"]["XL"], "{:.1f}"); put("BtTwoBKept", bs_["kept_median_batched"]["2B"], "{:.1f}")
sm = J("same_machine_summary.json")
put("SmMaxDiff", max(abs(c["machine_a"] - c["machine_b"]) for n in sm["heart_levo_machine_a_vs_machine_b"].values() for c in n.values()), "{:.3f}")

# ---------- VAD on separated vocals (exp_sep_redecode.py, analyze_sep_vad.py) ----------
sv = J("sep_vad_summary.json")
for sy, m in (("2B", "TwoBB"), ("2B+P", "TwoBC"), ("XL", "XLB"), ("XL+P", "XLC"), ("HeartMuLa", "Heart"), ("LeVo", "Levo")):
    put(f"Sv{m}Kept", sv["systems"][sy]["kept_median"], "{:.0f}"); put(f"Sv{m}Rec", sv["systems"][sy]["recall_vocals_vad"])
for pr, m in (("XL - 2B", "XLBvTwoBB"), ("LeVo - HeartMuLa", "LvH"), ("XL+P - 2B+P", "XLCvTwoBC")):
    v = sv["pairs"][pr]["vocals_vad"]; put(f"Sv{m}D", sgn(v["d"]), None); put(f"Sv{m}CI", ci(v["ci"]), None)
put("SvNoFlip", sum(1 for v in sv["pairs"].values() if v["vocals_vad"]["sig"] != "0" and v["mix_off"]["sig"] != "0" and v["vocals_vad"]["sig"] != v["mix_off"]["sig"]), "{}")

# ---------- Human word timings: Jam-ALT x JamendoLyrics (exp_vad_truth.py) ----------
vt = J("vad_truth_summary.json")
put("TrN", vt["n_songs"], "{}"); put("TrWords", f"{vt['n_words']:,}".replace(",", "{,}"), None)
put("TrMixDrop", 100 * vt["cond"]["mix_default"]["words_dropped_pooled"], "{:.0f}")
put("TrMixDropMed", 100 * vt["cond"]["mix_default"]["median_song_frac_dropped"], "{:.0f}")
put("TrProdDrop", 100 * vt["cond"]["mix_thr015"]["words_dropped_pooled"], "{:.0f}")
put("TrVocDrop", 100 * vt["cond"]["voc_default"]["words_dropped_pooled"], "{:.0f}")
put("TrRho", vt["cond"]["mix_default"]["spearman_dropped_vs_recall_loss"][0])

# ---------- Energy gate on separated vocals (exp_energy_gate.py, analyze_energy_gate.py) ----------
eg = J("energy_gate_summary.json")
ks = [v["kept_median"] for v in eg["systems"].values()]
put("EgKeptMin", min(ks), "{:.0f}"); put("EgKeptMax", max(ks), "{:.0f}")
v = eg["pairs"]["XL - 2B"]["energy"]; put("EgXLBvTwoBBD", sgn(v["d"]), None); put("EgXLBvTwoBBCI", ci(v["ci"]), None)
v = eg["pairs"]["LeVo - HeartMuLa"]["energy"]; put("EgLvHD", sgn(v["d"]), None); put("EgLvHCI", ci(v["ci"]), None)
put("EgContra", eg["contradictions_vs_off"], "{}"); put("EgPairs", len(eg["pairs"]), "{}")
put("EgTruthDrop", 100 * eg["truth"]["words_dropped_pooled"], "{:.0f}")

# ---------- Fourth family DiffRhythm (energy_gate_summary.json) ----------
dr = eg["pairs"]
for pr, m in (("DiffRhythm - HeartMuLa", "DrH"), ("DiffRhythm - LeVo", "DrL"), ("DiffRhythm - 2B", "DrTwoB"), ("DiffRhythm - XL", "DrXL")):
    for c, cm in (("mix_off", "Off"), ("mix_on", "On"), ("vocals_vad", "Voc"), ("energy", "Eg")):
        put(f"{m}{cm}D", sgn(dr[pr][c]["d"]), None); put(f"{m}{cm}CI", ci(dr[pr][c]["ci"]), None)
put("DrN", eg["systems"]["DiffRhythm"]["n"], "{}")
put("DrOffMean", eg["systems"]["DiffRhythm"]["recall_mix_off"]); put("DrOnMean", eg["systems"]["DiffRhythm"]["recall_mix_on"])
put("EgPairsAll", len(dr), "{}")
put("VocManufactured", sum(1 for v in dr.values() if v["mix_off"]["sig"] == "0" and v["vocals_vad"]["sig"] != "0"), "{}")

# ---------- Third speech VAD: WebRTC mode 2, retained fraction only (exp_webrtc_gate.py, analyze_webrtc_kept.py) ----------
import statistics
wk = {}
for r in csv.DictReader(open(R / "webrtc_kept48.csv")):
    wk.setdefault(r["exp"] + "/" + r["arm"], []).append(float(r["kept_sec"]) / float(r["dur"]))
meds = [statistics.median(v) for v in wk.values()]
put("WrKeptMin", 100 * min(meds), "{:.0f}"); put("WrKeptMax", 100 * max(meds), "{:.0f}"); put("WrSys", len(wk), "{}")
wt = list(csv.DictReader(open(R / "webrtc_gate_truth.csv")))
put("WrTruthDrop", 100 * (1 - sum(int(x["words_kept"]) for x in wt) / sum(int(x["n_words"]) for x in wt)), "{:.0f}")

# ---------- Public demo-page samples (exp_demo_vad.py, analyze_demo_vad.py) ----------
dm = J("demo_vad_summary.json")["groups"]["levo/text"]; su = dm["systems"]["suno"]
put("DemoN", dm["n_songs"], "{}"); put("DemoSys", len(dm["systems"]), "{}")
put("DemoSunoKept", su["median_kept_on"], "{:.1f}"); put("DemoSunoDur", su["median_dur"], "{:.0f}")
put("DemoSunoOff", su["recall_off"], "{:.2f}"); put("DemoSunoOn", su["recall_on"], "{:.2f}"); put("DemoSunoEg", su["recall_energy"], "{:.2f}")
put("DemoTauOff", dm["rank"]["kendall_vs_paper_recall_off"], "{:.2f}"); put("DemoTauOn", dm["rank"]["kendall_vs_paper_recall_on"], "{:.2f}")
put("DemoSunoRankOff", dm["rank"]["recall_off"].index("Suno") + 1, "{}"); put("DemoSunoRankOn", dm["rank"]["recall_on"].index("Suno") + 1, "{}")

# ---------- Second separator htdemucs_ft (exp_sep2.py, analyze_sep2.py) ----------
s2 = J("sep2_summary.json"); km = s2["kept_median"]
ht = [km[k]["silero_htft"] for k in ("2B", "2B+P", "XL", "XL+P")]
put("SepTwoKeptMin", min(ht), "{:.0f}"); put("SepTwoKeptMax", max(ht), "{:.0f}")
v = s2["pairs"]["XL - 2B"]["ht_vocals_vad"]; put("SepTwoXLD", sgn(v["d"]), None); put("SepTwoXLCI", ci(v["ci"]), None)
put("SepTwoR", s2["separator_agreement_per_song"]["silero"]["pearson_r"], "{:.2f}")

# ---------- Supplement S5 decomposition table rows from mediation_summary.json ----------
_md = J("mediation_summary.json"); _rows = []
_nm = {"XL − 2B": "XL $-$ 2B", "XL+pl − 2B": "XL+planner $-$ 2B", "XL − 2B+pl": "XL $-$ 2B+planner",
       "XL+pl − 2B+pl": "XL+planner $-$ 2B+planner", "LeVo − HeartMuLa": "LeVo $-$ HeartMuLa", "LeVo − 2B+pl": "LeVo $-$ 2B+planner"}
for _k, _v in _md["pairs"].items():
    _rows.append(f"{_nm.get(_k, _k)} & {sgn(_v['observed_on_gap'])} & {sgn(_v['coverage_part'])} {ci(_v['coverage_ci95'])} & "
                 f"{sgn(_v['intrinsic_part'])} {ci(_v['intrinsic_ci95'])} & {sgn(_v['residual'])} {ci(_v['residual_ci95'])} \\\\")
(HERE / "supp_s5_rows.tex").write_text("% Auto-generated by paper_numbers.py from results/mediation_summary.json; do not edit\n" + "\n".join(_rows) + "\n", encoding="utf-8")
put("MedNSongs", int(_md["model"].split("fit on ")[1].split()[0]), "{}")

# ---------- Supplement S8 per-system correlations (cross_family_summary.json) ----------
_cf = J("cross_family_summary.json")["spearman_kept_by_system"]
_sn = {"2B·B": "ACE 2B", "2B·C": "ACE 2B+planner", "XL·B": "ACE XL", "XL·C": "ACE XL+planner", "HeartMuLa": "HeartMuLa", "LeVo": "LeVo"}
(HERE / "supp_s8_rho_rows.tex").write_text("% Auto-generated by paper_numbers.py from results/cross_family_summary.json; do not edit\n" + "\n".join(
    f"{_sn.get(_k, _k)} & {_v['n']} & {_v['on']:.3f} & {_v['off']:.3f} \\\\".replace("-", "$-$") for _k, _v in _cf.items()) + "\n", encoding="utf-8")
put("SysRhoOnMin", min(v["on"] for v in _cf.values())); put("SysRhoOnMax", max(v["on"] for v in _cf.values()))
put("SysRhoOffMin", sgn(min(v["off"] for v in _cf.values())), None); put("SysRhoOffMax", sgn(max(v["off"] for v in _cf.values())), None)

(HERE / "numbers.tex").write_text("% Auto-generated by paper_numbers.py; do not edit\n" + "\n".join(out) + "\n")
print(f"wrote {len(out)} macros -> {HERE / 'numbers.tex'}")
