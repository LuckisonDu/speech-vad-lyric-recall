# Speech VAD Can Reverse ASR-Based Lyric-Recall Rankings of Generated Songs

Author: Luckison Du

Evaluation code and numeric results for the paper. Song-generation papers often score lyric
intelligibility by transcribing generated songs with Whisper and comparing the transcript with the
input lyrics. We show that turning on a speech voice-activity detector (VAD) in front of the
recognizer (e.g. `vad_filter=True` in faster-whisper) discards large, system-dependent parts of the
sung audio and can reverse the ranking of generators.

## Contents

```
scripts/      experiment and analysis scripts (Python 3, MIT license)
results/      summary JSONs and per-song numeric CSVs (no transcripts, no lyrics)
data/         manifest of the generated songs (ids and settings only) and benchmark ids
paper/        numbers.tex and supplement table rows as produced by scripts/paper_numbers.py
LICENSE       MIT (code)
```

Every script has an `# ===== Edit here =====` block at the top holding all paths and settings (no
command-line arguments). Run scripts from the repository root, e.g. `python3 scripts/paper_numbers.py`.

## Recompute the paper's numbers

```
python3 scripts/paper_numbers.py
```

This reads only `results/` and writes `paper/numbers.tex` (one `\newcommand{\N<Name>}{value}` per
number quoted in the paper) plus `paper/supp_s5_rows.tex` and `paper/supp_s8_rho_rows.tex`. It needs
only the Python standard library.

The analysis scripts below can also be rerun from the shipped CSVs (they regenerate the summary
JSONs; `analyze_vad_gen48.py` needs SciPy for the Wilcoxon p-values):
`analyze_vad_gen48, analyze_vad_stats, analyze_vad_sweep, analyze_toolchains, analyze_p2,
analyze_sampling, analyze_sep_vad, analyze_energy_gate, analyze_sep2, analyze_criterion,
analyze_multiverse, analyze_mediation, analyze_demo_vad, analyze_remix`.
`analyze_wer_per.py` and `analyze_cross_family.py` recompute WER/PER from transcripts and reference
lyrics, so they need the full per-item CSVs (with the `text` column) produced by rerunning the
experiments, plus your own lyric files.

## Script -> result -> paper mapping

| Paper result | Experiment script (audio -> per-song CSV) | Analysis script (-> summary) |
|---|---|---|
| Main VAD off/on comparison, 4 ACE-Step configurations + originals | `exp_vad_gen.py` -> `vad_gen48.csv` | `analyze_vad_gen48.py` -> `vad_gen48_summary.json` |
| HeartMuLa, LeVo, DiffRhythm families | `exp_vad_gen.py` -> `vad_gen48_heart/levo/dr.csv` | `analyze_cross_family.py` -> `cross_family_summary.json` |
| Retained seconds (mechanism) | `exp_vad_stats.py` -> `vad_stats48*.csv` | `analyze_vad_stats.py` -> `vad_stats48_summary.json` |
| VAD threshold sweep | `exp_vad_sweep.py` -> `vad_sweep48*.csv` | `analyze_vad_sweep.py` -> `vad_sweep48_summary.json` |
| Other recognizers / toolchains, vocal ratio | `exp_toolchains.py` -> `tool_*.csv`; `exp_vocal_ratio.py` -> `vocal_ratio48.csv` | `analyze_toolchains.py` -> `toolchains_summary.json` |
| WER / PER | (uses `vad_gen48*.csv` transcripts) | `analyze_wer_per.py` -> `wer_per_summary.json` |
| Separated-vocal decoding and VAD (HDemucs) | `exp_sep_redecode.py` -> `sep_redecode48*.csv`, `sep_vad48*.csv` | `analyze_p2.py` -> `p2_summary.json`; `analyze_sep_vad.py` -> `sep_vad_summary.json` |
| Second separator (htdemucs_ft) | `exp_sep2.py` -> `sep2_asr.csv`, `sep2_kept.csv` | `analyze_sep2.py` -> `sep2_summary.json` |
| Controlled remix (vocals + g x residual) | `exp_remix.py` -> `remix48.csv` | `analyze_remix.py` -> `remix48_summary.json` |
| Energy gate on separated vocals | `exp_energy_gate.py` -> `energy_gate48.csv`, `energy_gate_truth.csv` | `analyze_energy_gate.py` -> `energy_gate_summary.json` |
| WebRTC VAD (retained fraction) | `exp_webrtc_gate.py` -> `webrtc_gate_truth.csv` | `analyze_webrtc_kept.py` -> `webrtc_kept48.csv` |
| Human word timings (Jam-ALT x JamendoLyrics) | `exp_vad_truth.py` -> `vad_truth.csv`, `vad_truth_summary.json` | - |
| Public benchmarks (Jam-ALT, MUSDB-ALT), two machines | `exp_vad_bench.py` -> `vad_bench.csv`, `vad_bench_rerun.csv` | `vad_bench_summary.json` (*) |
| Batched pipeline defaults | `exp_batched.py` -> `batched48_default.csv` | `batched_summary.json` (*) |
| Sample size / seeds | (uses `vad_gen48.csv`) | `analyze_sampling.py` -> `sampling_summary.json` |
| Evaluation multiverse (figure) | (uses `tool_*.csv`, `sep_redecode48.csv`, ...) | `analyze_multiverse.py` -> `multiverse_summary.json` |
| Mediation / decomposition | (uses the CSVs above) | `analyze_mediation.py` -> `mediation_summary.json` |
| Reversal criterion | (uses the CSVs above) | `analyze_criterion.py` -> `criterion_summary.json`; preregistered test `criterion_prereg_predictions.json` -> `criterion_prereg_result.json` (*) |
| Public demo-page samples | `exp_demo_vad.py` -> `demo_vad.csv` | `analyze_demo_vad.py` -> `demo_vad_summary.json` |
| WhisperX across families, same-machine rerun | `exp_toolchains.py` -> `tool_whisperx*.csv` | `whisperx_families_summary.json`, `same_machine_summary.json` (*) |
| Listening test | - | `listen_final*.json` (aggregated; see below) |
| All numbers in the paper | - | `paper_numbers.py` -> `paper/numbers.tex` |
| Figures | - | `make_figs.py` -> `figs/fig1_flip.pdf`, `fig2_kept.pdf`, `fig5_multiverse.pdf` (paper) and `fig4_sweep.pdf`, from `results/`. The per-song VAD timeline example figure is not included because it is drawn from audio. |

Shared modules: `verify_metrics.py` (word normalization, lyric recall, WER; checked value-by-value
against an earlier implementation), `audio_io.py` (audio loading; librosa or PyAV backend).

(*) These summaries were computed in interactive analysis sessions and are shipped as-is; the
per-item CSVs they were computed from are included.

## Rerunning the experiments

Audio is not distributed. Generate songs yourself with the settings in `data/manifest.csv`, crop to the
first 120 s, and place them as `data/audio/generated/runs/<exp>/<arm>/<song_id>@s<seed>.wav`, with
reference lyric words in `data/lyrics_meta/<song_id>.meta.json`. Then run the experiment scripts and the
analysis scripts. Software used for the paper:

| Component | Version |
|---|---|
| Python | 3 |
| faster-whisper | 1.2.1 (CTranslate2 backend), model Whisper large-v3, fp16, `beam_size=1`, `language="en"` |
| Silero VAD | v6, as bundled with faster-whisper 1.2.1 (`vad_filter=True`, default parameters unless noted) |
| WebRTC VAD | `webrtcvad-wheels` 2.0.14.post1, mode 2 |
| Source separation | torchaudio `HDEMUCS_HIGH_MUSDB_PLUS`; demucs 4.0.1 `htdemucs_ft` |
| openai-whisper | 20250625 |
| WhisperX | 3.8.6 (pyannote VAD) |
| transformers | 5.18.0 (Whisper large-v3 pipeline, 30 s chunks) |
| NVIDIA NeMo | 3.0.0, `nvidia/parakeet-tdt-0.6b-v2` |
| librosa / PyAV | audio decoding (backend chosen in `audio_io.py`) |
| g2p_en | phoneme conversion for PER |
| NumPy, SciPy | statistics |
| matplotlib | figures (`make_figs.py`) |

Generators: ACE-Step 1.5 (2B `acestep-v15-turbo`, XL `acestep-v15-xl-turbo`, planner
`acestep-5Hz-lm-1.7B`), HeartMuLa-oss-3B, LeVo (SongGeneration v2-large), DiffRhythm 1.2 full; see
`data/manifest.csv` for per-song settings.

## Not included, and why

* **Lyrics and audio.** The lyrics are from commercial songs and the generated audio is conditioned on
  them, so neither is redistributed. Transcript columns are removed from all CSVs for the same reason;
  song ids are pseudonymous (`song001`-`song048`).
* **Listening-test raw answers and rater information.** Only aggregated results (`listen_final.json`,
  leave-one-rater-out counts, reasons by model size) are included; the per-rater answers and the
  listening interface are not.
* **Demo-page transcripts.** Only numeric scores are included for the public demo-page samples.
* **Supplement.** The supplementary PDF is distributed with the paper, not here.
* One unrelated exploratory block (a visual-domain analogy) was removed from `paper_numbers.py`; its
  macros are not used in the paper or the supplement.
