# Speech VAD Can Reverse ASR-Based Lyric-Recall Rankings of Generated Songs

Author: Luckison Du

Evaluation code and numeric results for the paper (version v38m of the manuscript). Song-generation papers often score lyric
intelligibility by transcribing generated songs with Whisper and comparing the transcript with the
input lyrics. We show that turning on a speech voice-activity detector (VAD) in front of the
recognizer (e.g. `vad_filter=True` in faster-whisper) discards large, system-dependent parts of the
sung audio and can reverse the ranking of generators.

## Contents

```
scripts/      experiment and analysis scripts (Python 3, MIT license)
results/      summary JSONs and per-song numeric CSVs (no transcripts, no lyrics); results/singing_vad/ holds the singing-voice VAD tables
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
analyze_multiverse, analyze_mediation, analyze_demo_vad, analyze_remix, analyze_singing_vad,
analyze_rerun_noise`.
We reran these 16 scripts from the shipped CSVs before this release and compared the regenerated
summaries value by value with the shipped summaries (see "Checks performed before this release" below).
`analyze_wer_per.py` and `analyze_cross_family.py` recompute WER/PER from transcripts and reference
lyrics, so they need the full per-item CSVs (with the `text` column) produced by rerunning the
experiments, plus your own lyric files.

## Seeds and what changed since the first release

* ACE-Step 1.5 (four configurations: 2B / XL, without / with planner) now has **six seeds** (1-6, 48 lyric windows each,
  1728 manifest rows in total); HeartMuLa, LeVo and DiffRhythm have four seeds (1-4). Seeds 3-6 of ACE-Step were generated
  on other (cloud) GPUs than seeds 1-2 with the same code revision, weights and settings; the VAE decoding tile size, which the
  official code otherwise picks from the free GPU memory, was pinned for seeds 3-6 (`vae_decode_tile` in `data/manifest.csv`).
  Regenerating a seed on a different GPU does not give bit-identical audio (we measured SNRs of 58 dB and 24 dB against a
  regeneration on the original machine for two configurations), so reruns reproduce the statistics, not the waveforms.
* Which seeds enter which analysis: the main comparison (`vad_gen48`), retained seconds, WER/PER, the separated-vocals-then-VAD check,
  the energy gate, the mediation analysis and the cross-family comparison use all available seeds (cross-family pairs only the seeds
  present on both sides, 1-4). The analyses that need extra recognizer outputs that exist only for seeds 1-2 (other toolchains,
  the fix pipeline `analyze_p2.py`, the evaluation multiverse, the reversal criterion, the second separator) are restricted to
  seeds 1-2 through `ACE_SEEDS` at the top of each script, so that all settings are compared on the same 384 songs.
  The threshold sweep, the controlled remix and the vocal-ratio tables were not extended and cover seeds 1-2 only.
* **Corrected vocal separation.** The chunked HDemucs separation used in the first release placed each chunk 2 s late after the
  first chunk and lost 2 s at every 20 s boundary (a 120 s song came out as 118 s). `scripts/sep_util.py` is the single corrected
  implementation (sample-aligned, length asserted); all separation-dependent scripts use it, and every result that reads
  separated vocals (`sep_*`, `energy_gate*`, `remix48`, `vocal_ratio48`, the vocal columns of `vad_stats48` and `vad_sweep48`,
  `vad_truth`, `demo_vad`, `sep2_kept`) was recomputed with it. `scripts/test_sep_util.py` and `results/sep_util_unittest.json` show the
  block-wise lag check on one song. Numbers of the first release (commit b902307) that depend on separated vocals are superseded.
* New: `exp_singing_vad.py` / `analyze_singing_vad.py` (singing-voice VAD control with PANNs, threshold calibrated on arm C),
  `analyze_rerun_noise.py` / `dedupe_vad_gen48.py` (re-run noise from duplicated transcriptions of identical audio),
  the third listening test (aggregated summaries only), and the macros that quote them in `paper/numbers.tex`.

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
| Singing-voice VAD control (PANNs) | `exp_singing_vad.py` -> `singing_vad/asr_*.csv`, `singing_vad/seg_stats.csv`, `singing_vad_calib.json` | `analyze_singing_vad.py` -> `singing_vad_summary.json` |
| GPU re-run noise | `dedupe_vad_gen48.py` (-> `vad_gen48_with_dups.csv`, `vad_gen48_rerun_nondeterminism.json`) | `analyze_rerun_noise.py` -> `rerun_noise_summary.json` |
| Separation unit test | `test_sep_util.py` -> `sep_util_unittest.json` | - |
| Listening tests | - | `listen_final*.json`, `listen3_summary.json`, `listen3_part3_extra.json` (aggregated; see below) |
| All numbers in the paper | - | `paper_numbers.py` -> `paper/numbers.tex` |
| Figures | - | `make_figs.py` -> `figs/fig1_flip.pdf`, `fig2_kept.pdf`, `fig5_multiverse.pdf` (paper) and `fig4_sweep.pdf`, from `results/`. The per-song VAD timeline example figure is not included because it is drawn from audio. |

Shared modules: `verify_metrics.py` (word normalization, lyric recall, WER; checked value-by-value
against an earlier implementation), `audio_io.py` (audio loading; librosa or PyAV backend), `sep_util.py` (corrected
chunked HDemucs separation).

## Reproducibility notes

* **GPU transcription is not bit-reproducible.** faster-whisper (CTranslate2, fp16) run twice on the same audio with the same
  settings gives different transcripts for a fraction of the songs. `results/rerun_noise_summary.json` quantifies this from duplicated
  transcriptions of identical audio and compares it with the main effect (the noise of a 288-pair mean is small next to the effect).
  Expect a rerun to differ in the last digits and the recognizer-dependent summaries to agree within this noise, not bit for bit.
  The analysis scripts, by contrast, are deterministic given the shipped CSVs (fixed bootstrap seed 101).
* Seeds 3-6 of ACE-Step were generated, and part of their recognition was run, on other machines than seeds 1-2 (same package versions,
  same model weights verified by checksum). The XL regenerations differ from the originals more than the 2B ones (see above); we treat
  this as GPU numerical differences, not as a change of the experiment.
* Absolute scores of different generator families were recognized on different machines and are indicative only.

## Checks performed before this release

* Rerunning the 16 analysis scripts listed above on the shipped CSVs regenerates the shipped summary JSONs value by value
  (differences below 1e-9; SciPy is needed for the Wilcoxon p-values).
* `paper_numbers.py` regenerates `paper/numbers.tex` from the shipped results.
* `analyze_wer_per.py`, `analyze_cross_family.py` and the listening analyses are not rerunnable from this repository (they need
  transcripts, reference lyrics, or the raw listening answers, which are not distributed); their summaries are shipped as computed.

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
  leave-one-rater-out counts, reasons by model size, `listen3_summary.json`, `listen3_part3_extra.json`) are included; raters
  appear only as codes (R1-R4, L1-L4). The per-rater answers, free-text reasons, the listening interface and the analysis script
  of the third test (it reads the raw answers) are not.
* **Demo-page transcripts.** Only numeric scores are included for the public demo-page samples.
* **Supplement.** The supplementary PDF is distributed with the paper, not here.
* One unrelated exploratory block (a visual-domain analogy) was removed from `paper_numbers.py`; its
  macros are not used in the paper or the supplement.
