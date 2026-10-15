# Evaluation set

Lyrics and audio are **not redistributed**: the lyrics come from commercial songs, and the generated
audio is conditioned on them. This folder only identifies the evaluation items.

* `manifest.csv` - one row per generated song (1728 rows):
  `song_id` (pseudonymous lyric-window id, `song001`-`song048`, identical to the `song` column in `../results/*.csv`),
  `exp` / `arm` (run and configuration names used in the result files),
  `generator`, `model`, `planner` (ACE-Step arm `B` = no planner, arm `C` = with the 5 Hz LM planner),
  `seed`, `crop_sec` (all audio is cropped to the first 120 s before evaluation),
  `gen_params` (generation settings recorded at generation time).
  * ACE-Step 1.5 2B and XL: 48 lyric windows x 6 seeds x 2 arms each (1152 songs). Seeds 3-6 were generated on other GPUs; their rows also
    record the pinned VAE decoding tile size (`vae_decode_tile`).
  * HeartMuLa-oss-3B, LeVo (SongGeneration v2-large), DiffRhythm 1.2 full: 48 x 4 seeds each (576 songs).
* `benchmark_ids.csv` - ids used in the result files for the public benchmarks (Jam-ALT English songs,
  MUSDB-ALT), mapped to the track names of those public datasets.

The 48 lyric windows were drawn from a set of English commercial songs, stratified into four equal
recall strata measured on the original recordings (sampling seed 20260923). Every song is identified
only by its pseudonymous id; the original titles and artists are withheld.

To rerun the experiments you need to generate your own audio with the listed models and settings
and place it as `data/audio/generated/runs/<exp>/<arm>/<song_id>@s<seed>.wav`, with reference lyric
words in `data/lyrics_meta/<song_id>.meta.json` (`{"ref_words": [...]}`).
