# Track 1: Noise Event Detection

Team sonicboom

This folder holds the code of our Track 1 systems for the IndoML 2026 Datathon
([Codabench competition 17825](https://www.codabench.org/competitions/17825/)).

We have two systems. They share the same model family (ATST-Frame encoder with a RED interval head):

- **C8** is our model-only system: one learned pipeline for every clip.
- **950232** is our leaderboard entry (4th of 82). It adds a router, template matching for the synthetic clips,
  library copies in natural clips, a WavLM-Large end refiner and a few decoding rules.

## 1. Task and score

Each clip is 16 kHz mono speech from the Vaani project. Some clips are natural recordings, some are synthetic
(clean speech with pasted noise events). For each clip, a system writes the time spans (onset and offset in
seconds) of the noise events. The event class is not scored.

Score = event F1 + segment Dice (maximum 2.0).

- **Event F1**: a predicted event matches a reference event when its onset and its offset are both within
  20 % of the reference length (at least 50 ms).
- **Segment Dice**: overlap of the predicted and reference event frames on a 10-ms grid.

The test set has 5,517 clips (3,437 natural, 2,080 synthetic). The organizer validation release has 2,501 labelled
clips (1,806 natural, 695 synthetic; the synthetic clips come with their clean references).

## 2. Systems and results

| System | Codabench upload | Score | Event F1 | Dice | Final rank |
| --- | --- | --- | --- | --- | --- |
| 950232 (leaderboard entry) | 950232 | 1.6112 | 0.7911 | 0.8201 | 4th of 82 |
| C8 (model-only) | 977262 | 1.5925 | 0.7795 | 0.8130 | not on the board |

The event F1 of 950232 (0.7911) is the best event F1 on the final board.

### C8 (model-only)

C8 runs one pipeline on every clip. It reads no file name, no audio peak, no template and no noise library at
test time (`inference/audit_guard.py` checks this on every run).

1. **Four members**, each an ATST-Frame encoder with a RED interval head (per 40-ms frame and 7 classes: a presence
   value and the distances to the event onset and offset): **P2-L3, S1-L3, R10-L2, PSR10B-L2** (section 6).
2. Each member runs at **4 input shifts** (0, 10, 20 and 30 ms of leading zeros). This gives 16 event sets.
3. **Per set**: the P2 decoder at presence ≥ 0.75, events ≥ 0.08 s; then a refiner prunes overlapping events.
   An empty set is decoded again at 0.65.
4. **Vote**: class-agnostic clustering at IoU 0.55. A cluster is kept when
   best score × (supporting sets / 16) ≥ 0.5. The refined events all carry score 1.0, so a cluster needs 8 of the
   16 sets, and its endpoints are the plain means of its members.
5. **Empty clip**: R10-L2 decoded at 0.55 (support ≥ 2 of its 4 shifts), then PSR10B-L2 the same way, then one
   whole-clip event [0, D].

### 950232 (leaderboard entry)

950232 uses the base members P2, S1, R10 and PSR10B (24 event sets) and adds:

- a **peak router**: a clip is natural when its int16 peak reaches full scale (section 8.1);
- **synthetic clips**: template matching with the noise library, master-file extents, a whitened matched filter,
  natural-cut kernels and a label-grid snap (section 8.3);
- **natural clips**: the vote with four extra rules (LTL6, LACC, LPAD8, LD1R), a WavLM-Large end refiner (E0),
  library copies (Q1) and two more fallback donors (section 8.2).

## 3. Folder map

```
atst_red/                  model code
  indoml/                  ATST-Frame + RED head package: features, heads, training, decoding, metrics, data split
  context_crop.py          the P2 loader (load_parent) and the context-crop window logic
  mixed_training.py        window streams and the fp32-head rule of the mixed strong/weak training
  cuda_adapter.py          native-bundle checks and the CUDA model adapter
data_prep/
  vaani/                   download, decode and audit the Vaani training release; stage the training inputs
  kaggle_inputs/           input datasets of the training kernels (folds, Gold pools, Silver labels, test-natural, LIBSYN)
  recipe/                  transcript audit and the Silver / Bronze training recipe
  gold_synth/              "generated" training clips built from Gold material
  libsyn/                  LIBSYN: noise-library templates pasted on clean speech beds
training/
  p2/                      P2 full-data training (reference only, see section 10)
  s1/                      S1 trainer (Apple GPU)
  kernels/                 Kaggle kernels: R10, PSR10B and the training rounds L / L2 / L3; fp16 -> fp32 cast
inference/
  indoml_pipeline/         shared package: P2 inference, refiner, fusion, templates, router, scorer, durations
  export_frames.py         per-frame outputs of one member at one input shift
  vote.py                  the C8 vote (with --c4: the 950232 natural rules)
  audit_guard.py           runs vote.py under an audit
  make_zip.py              writes the Codabench zip and checks the format
  t1score.py               local scorer (the organizer formulas) with per-subset statistics
system_950232/             the additions of 950232
  peak_router.py           natural / synthetic router from the audio peak
  job2/                    master-file extents of the synthetic branch
  wmf/                     whitened matched filter (tile GLRT) of the synthetic branch
  copies/                  library copies in natural clips (Q1) and natural-cut kernels (Q2)
  sr1/                     SR1 label-grid snap of the synthetic events
  e0/                      WavLM-Large E0 end refiner (local stage, Kaggle training and prediction)
  donors/                  fallback donors R10S1 (Apple GPU trainer) and SOUP2 (weight average)
requirements.txt
```

## 4. Setup

### 4.1 Python

```
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` pins the model environment (torch 2.8.0). The data-preparation scripts ran in a second
environment; the file lists its packages and versions as comments. The Kaggle kernels run on the Kaggle GPU image
(2 × T4, Python 3.12.13, torch 2.10.0+cu128) without extra installs.

Run the scripts from this folder (`track1_event_detection/`). Scripts that use the `indoml` package need
`PYTHONPATH=atst_red`; the `indoml_pipeline` commands run from `inference/`.

### 4.2 ATST-Frame code and checkpoint

```
git clone https://github.com/fschmid56/PretrainedSED.git
git -C PretrainedSED checkout 1aa47e482f7e89904cba2338999345025d8b4e36
curl -L -o ATST-F_strong_1.pt https://github.com/fschmid56/PretrainedSED/releases/download/v0.0.1/ATST-F_strong_1.pt
shasum -a 256 ATST-F_strong_1.pt     # fbf2577958e3648d55ee8cea7e0e5260c4505fb0c13964e1ec81bc367cee5eda
```

### 4.3 The native bundle (`T1_NATIVE_BUNDLE`; Kaggle dataset `native-bundle`)

```
native-bundle/
  asset-manifest.json          {"parent_checkpoint_sha256": <sha256 of assets/parent-native-e5.pt>,
                                "assets": [{"path": ..., "bytes": ..., "sha256": ...} for each file below]}
  assets/ATST-F_strong_1.pt    section 4.2
  assets/parent-native-e5.pt   the E5 checkpoint (section 6.1); its model_config.head_config defines the RED head
  source/PretrainedSED/        the clone of section 4.2
```

### 4.4 The P2 parent package (`T1_P2_CHECKPOINT`; Kaggle dataset `p2-parent`)

```
p2-parent/
  context_crop.py              = atst_red/context_crop.py
  mixed_training.py            = atst_red/mixed_training.py
  parent-p2.pt                 P2, sha256 419c441e911419b33fa066d67ffa43bb63a8c6b3c659175ccb565f0ecfb83373
                               (model_state digest 3fa453a39509b804f80e628c53643c346824373b63c81b0deb5d877bc14a64e9)
```

`context_crop.load_parent` rebuilds the architecture from the native bundle and loads `parent-p2.pt`. Every member
has exactly this architecture; a member checkpoint only replaces `model_state`.

### 4.5 Environment variables

The scripts take their inputs from these variables (or from command-line arguments where they have them).

| Variable | Used by | Meaning |
| --- | --- | --- |
| `T1_WORK` | data_prep, e0 | work folder for every data product (default: `.`) |
| `T1_VAL_META` | data_prep | `validationMetadata.json` of the organizer validation release |
| `INDOML_VALIDATION_ZIP`, `INDOML_TEST_ZIP` | data_prep, inference, system_950232 | the organizer zips (`validation.zip`, `input_data.zip`) |
| `INDOML_WORK` | indoml_pipeline, system_950232 | pipeline work folder (templates, durations, audio) |
| `T1_NATIVE_BUNDLE`, `T1_P2_CHECKPOINT` | inference, training/s1 | sections 4.3 and 4.4 |
| `T1_FRAMES` | vote.py | frame store root (same as `--frames-root`) |
| `T1_ORGANIZER_AUDIT` | stage_full_inputs.py | the organizer validation manifests (section 5.3) |
| `T1_FOLDS`, `T1_GOLD_SYNTH_MANIFEST` | kaggle_inputs, training/s1, job2 | `folds.json`; `manifest.jsonl` of gold_synth.py |
| `T1_AUDIO_PREFIX_OLD`, `T1_AUDIO_PREFIX_NEW` | gold_synth | optional rewrite of the audio path prefix in the Gold manifests |
| `T1_LIBRARY`, `T1_MASTERS`, `T1_BEDS`, `T1_TRAIN_LENGTHS`, `T1_G01`, `T1_STORE` | data_prep/libsyn | LIBSYN inputs (section 5.6) |
| `T1_LIBSYN_ROOT` | build_libsyn_dataset.py | folder with the generator outputs `v2/` and `ds-libsyn-v1/` |
| `T1_VAANI_FULL`, `T1_VAANI_INPUTS`, `T1_GOLD_POOLS`, `T1_ORGANIZER_AUDIO`, `T1_COMMON` | training/s1 | local copies of the kernel inputs (section 6.2) |
| `T1_DS_<KEY>`, `ATST_SCALE_INPUT_ROOTS` | training/kernels | dataset folder names; extra input roots (section 6.3) |
| `T1_MAC_INPUT` | system_950232/donors | a folder laid out like `/kaggle/input` with the kernel datasets |
| `T1_WAVLM_ASSETS` | system_950232/e0 | folder with the pinned microsoft/wavlm-large files |
| `T1_JOB2_OUT`, `T1_WMF_CACHE`, `T1_SR1_INPUTS`, `T1_BASE_PREDICTIONS`, `T1_OFFICIAL_SCORER_DIR` | system_950232 | outputs, caches and stored inputs of section 8 |
| `KAGGLE_DATASET_OWNER` | dataset builders | Kaggle user name written into `dataset-metadata.json` |
| `T1_VALIDATION_DIR`, `T1_GUARD_TMP` | t1score.py, audit_guard.py | extracted validation folder; temporary folder |

## 5. Data preparation

The training data is the Vaani Noise-Event Dataset on Hugging Face (revision
`be488e2ac12fd62bef46b9f83e3a5feded575333`, 182 parquet files, 17.7 GB). It has three tiers: **Gold** (verified
event timestamps), **Silver** (unverified timestamps) and **Bronze** (clip tags only). The organizer validation
and test releases come from the competition page and are not redistributed here.

```
export T1_WORK=/path/to/work
export T1_VAL_META=/path/to/validation/validationMetadata.json
export INDOML_VALIDATION_ZIP=/path/to/validation.zip
export INDOML_TEST_ZIP=/path/to/input_data.zip
export INDOML_WORK=/path/to/pipeline-work      # templates, durations, frames of the inference package
```

### 5.1 Vaani training release

```
python data_prep/vaani/download_full.py      # 182 shards -> $T1_WORK/vaani-release (Hugging Face login if asked)
python data_prep/vaani/prepare_full.py       # -> $T1_WORK/vaani-full: manifest.jsonl (90,630 clips) + 16 kHz WAV
PYTHONPATH=atst_red python -m indoml.split_geographic --manifest $T1_WORK/vaani-full/manifest.jsonl \
    --output $T1_WORK/vaani-split            # geographic split (seed 42, validation share 0.125)
```

`vaani-gold/{train,val}.jsonl` are the rows of `vaani-split/{train,val}.jsonl` with
`quality == "verified_timestamps"` (9,663 and 1,443 Gold clips).

### 5.2 ATST features and the inner split (P2 lineage)

```
for split in train val; do
  PYTHONPATH=atst_red python -m indoml.pretrained --source-root PretrainedSED --checkpoint ATST-F_strong_1.pt \
      --manifest $T1_WORK/vaani-gold/$split.jsonl --output $T1_WORK/atst-gold-$split \
      --device mps --batch-size 4 --threads 4
done
python data_prep/vaani/prepare_inner.py      # -> $T1_WORK/inner-split/{train,val}.jsonl (8,213 / 1,439 clips)
```

### 5.3 Training inputs (`vaani-inputs`)

```
python data_prep/vaani/audit_full_release.py         # -> $T1_WORK/full-release-audit/training-audit.jsonl
T1_ORGANIZER_AUDIT=/path/to/organizer-audit python data_prep/vaani/stage_full_inputs.py
                                                     # -> $T1_WORK/full-input-staging/input-upload
```

`training.jsonl` has 90,635 rows: 11,095 Gold clips with strong labels and 79,540 clips with clip tags only
(16 Gold, 61,640 Silver, 17,884 Bronze). `T1_ORGANIZER_AUDIT/input-audit/` holds three manifests of the organizer
validation release: `inference.jsonl` (clip_id, audio_path, duration), `references.jsonl` (the labelled events)
and `official-audio.jsonl` (the audio inventory). The validation audio is packed only for evaluation inside the
kernels; it is never a training row.

### 5.4 Folds, generated clips and the kernel datasets

```
python data_prep/kaggle_inputs/make_folds.py --val-meta $T1_VAL_META --out $T1_WORK/folds.json
python data_prep/gold_synth/gold_stats.py            # gold_stats.pkl (beds and pieces of the Gold clips)
python data_prep/gold_synth/measure_levels.py        # val_levels.pkl (piece-to-bed levels, 695 synthetic val clips)
python data_prep/gold_synth/gold_synth.py --out $T1_WORK/gold-synth --seed 20260916 --count-a 6000 --count-b 3000
export T1_FOLDS=$T1_WORK/folds.json T1_GOLD_SYNTH_MANIFEST=$T1_WORK/gold-synth/manifest.jsonl
python data_prep/kaggle_inputs/prepare_gold_pools.py # "gold-pools": Gold split, generated clips (Gold beds only)
python data_prep/kaggle_inputs/prepare_silver.py     # "silver-pools": Silver timestamps, generated-clip sessions
python data_prep/kaggle_inputs/prepare_test_audio.py # "test-natural": unlabelled test clips (see section 10)
```

`folds.json` groups the 1,806 natural validation clips by (state, district). The kernels use it to drop every
training row that shares a recording session with a natural validation clip.

### 5.5 Silver and Bronze recipe (`bronze-recipe`)

```
python data_prep/recipe/audit_main.py        # tag vocabulary and transcript links -> data_prep/recipe/out/
python data_prep/recipe/build_recipe.py      # out/recipe_silver.jsonl, recipe_bronze_strong.jsonl, recipe_bronze_neg.jsonl
```

The recipe gives each Silver event a mode (`full`, `presence`, `ignore`), turns the Bronze clips with a wrap-all
transcript tag into whole-clip events (1,887 rows outside the validation sessions), and lists the absent
categories of each Bronze clip.

### 5.6 LIBSYN

LIBSYN pastes the noise-library templates on clean speech beds of the training release. The library has 203
templates: the exact noise waveforms of the 695 synthetic validation clips (noisy minus clean, cut at the labels,
then clustered). The beds are 25,280 event-free speech spans (24.7 h) of the Gold and Silver clips, cleaned with
DeepFilterNet3 and masked with Silero VAD.

```
export T1_STORE=/path/to/store                 # int16 audio stores and residual caches
cd inference && python -m indoml_pipeline.templates library && cd ..   # $INDOML_WORK/templates/library.pkl
python data_prep/libsyn/m01_masters.py       # masters.pkl: start-aligned cuts of one long noise file
python data_prep/libsyn/g01_val_tracks.py    # g01_val_tracks.pkl: the noise track of every synthetic val clip
python data_prep/libsyn/valstats.py          # valstats.json: levels, zero runs, vocal slots, categories
python data_prep/libsyn/train_float.py       # train_float.jsonl: training-release clip lengths
```

The beds come from the Track 2 scripts `track2_noise_removal/training/data/build_banks.py` and
`build_clean_beds.py` (see the Track 2 README, section 6.2); `T1_BEDS` points to their `clean_beds/` output.

```
export T1_LIBRARY=$INDOML_WORK/templates/library.pkl T1_BEDS=/path/to/clean_beds
python data_prep/libsyn/gen_libsyn.py --out libsyn/v1 --start 0 --count 2000 --keep-beds
python data_prep/libsyn/calibrate.py --dir libsyn/v1 --tag 000000-001999       # bands B1-B9 below
python data_prep/libsyn/gen_libsyn.py --out libsyn/v1 --start 2000 --count 14000
python data_prep/libsyn/gen_libsyn.py --out libsyn/v2 --start 16000 --count 2000 --keep-beds
python data_prep/libsyn/calibrate.py --dir libsyn/v2 --tag 016000-017999
python data_prep/libsyn/gen_libsyn.py --out libsyn/v2 --start 18000 --count 30000
T1_LIBSYN_ROOT=libsyn python data_prep/kaggle_inputs/build_libsyn_dataset.py  # "libsyn-v2" (clips 16,000-47,999)
```

LIBSYN-v1 (clips 0-15,999) is packed in the same layout as `libsyn-v2` (audio zip, manifest, sessions file and
`val-natural-libcopy-66.txt`, the 66 natural validation clips that hold an exact library copy). Clip i depends only
on (seed, i); the default seed is 20261006.

Generator rules:

| Rule | What it does |
| --- | --- |
| L1 length | clip length D drawn from the training-release clip lengths inside [3.0, 15.2] s |
| L2 bed | beds of one session joined in random order (then same district, then any session), joined by exact-zero gaps |
| L3 silence | exact-zero runs added in the lowest-energy frames, with the validation count and length laws; vocal events are inserted into a zero slot |
| L4 count | 1 to 5 events, P(n) proportional to the validation histogram 110 / 78 / 124 / 340 / 43 |
| L5 template | class first (vocal 30.3 %, noise 68.0 %, master 1.7 %), then a uniform template of the class |
| L6 placement | no overlap; gaps ≥ 0.2 s; events start ≥ 0.25 s and end ≤ D − 0.251 s; masters start at their first sample and are cut at the clip end |
| L7 levels | vocal: −SNR from N(3.97, 0.44) dB; noise: SNR from N(mean of the template on validation, 4.956 dB), kept in [−12.24, 15.49] dB |
| L8 mixture | bed + events, then one gain so that the int16 peak is 32,440; 16 kHz FLAC |
| L9 labels | onset = paste start (rounded down to 1 ms); offset = onset + template length (masters: D) |
| L10 volume | 16,000 clips in v1, 32,000 in v2; deterministic per clip; session file for the session filter |
| L11 provenance | bed ids, template ids, gains and levels of every clip |

Calibration bands (`calibrate.py`, measured again from the written audio): B1 count shares within ±0.03 of
validation; B2 every placement rule holds; B3 vocal levels; B4 noise levels in range; B5 share of exact zeros;
B6 peak 32,440 ± 1; B7 exactness (mixture = bed + template within 1 LSB); B8 length range; B9 bed sources (no
validation session, no test file opened).

### 5.7 Kaggle input datasets

| Dataset (default folder name) | Built by |
| --- | --- |
| `vaani-inputs` | `data_prep/vaani/stage_full_inputs.py` |
| `native-bundle`, `p2-parent` | sections 4.3 and 4.4 |
| `gold-pools`, `silver-pools`, `test-natural` | `data_prep/kaggle_inputs/prepare_*.py` |
| `bronze-recipe` | `data_prep/recipe/build_recipe.py` (the files in `out/`) |
| `libsyn-v1`, `libsyn-v2` | `data_prep/kaggle_inputs/build_libsyn_dataset.py` |
| `member-inits` | the start weights of a training round (section 6.4) |

The kernels find each dataset by folder name under `/kaggle/input`. `T1_DS_<KEY>` changes a name (for example
`T1_DS_INPUTS`), and `ATST_SCALE_INPUT_ROOTS` adds other roots.

## 6. Training of the members

All members have the P2 architecture: the ATST-Frame encoder (12 blocks) and the RED head. The objective is P2's:
presence (class weight 5), boundary (weight 100) and interval IoU (weight 1) on strong rows, and a positive-only tag
term on weak rows. Optimizer: AdamW (weight decay 1e-4) with 13 depth groups (decay 0.95), gradient clip 5,
frequency warp (p 0.5, scale 0.6-1.0).

### 6.1 P2 (reference)

| Step | Code | Settings | Kept |
| --- | --- | --- | --- |
| 1. frozen features | `indoml.pretrained` (section 5.2) | ATST-F_strong_1 on the Gold clips | |
| 2. RED head on frozen features | `indoml.intervals train --mode red` | inner split; 10 epochs, patience 4, batch 16, positive-weight cap 5.0; then continued | epoch 16 |
| 3. encoder adaptation | `indoml.atst_intervals train --head-checkpoint` | encoder lr 1e-6, head lr 1e-4, batch 8, seed 45 | epoch 0 |
| 4. E5 | `indoml.atst_intervals train --initial-native-checkpoint` | 6 epochs, seed 49, frequency warp p 0.5, scale 0.6-1.0, warp seed 50 | epoch 5 = `parent-native-e5.pt` |
| 5. P2 | `training/p2/full_train.py` (Kaggle) | strong labels of all Gold clips + clip tags of the other clips; 16 strong + 32 weak clips per update; warmup / hold / decay 2,486 / 2,486 / 14,916 updates | update 4,972 = `parent-p2.pt` |

P2 is selected by its score on the 2,501 validation clips (1.158590 at update 4,972).

### 6.2 S1 (Apple GPU)

S1 continues P2 with Silver timestamps as strong labels.

```
export T1_NATIVE_BUNDLE=/path/to/native-bundle T1_P2_CHECKPOINT=/path/to/parent-p2.pt
export T1_VAANI_FULL=$T1_WORK/vaani-full T1_VAANI_INPUTS=/path/to/vaani-inputs T1_GOLD_POOLS=/path/to/gold-pools
export T1_ORGANIZER_AUDIO=/path/to/organizer-audio T1_COMMON=/path/to/common   # val/*.wav; val_meta.json, scorer.py
python training/s1/train_s1.py --arm S1 --updates 4000 --out runs/S1
```

Per update: 4 Gold-train clips + 4 Silver clips (Silver weight 0.4) with P2's full objective, and 16 tag-only
clips (weight 0.1). Encoder 1e-6, head 1e-4, warmup then cosine. 4,000 updates (1.9 h on an Apple M-series GPU;
encoder on MPS, head on CPU, fp32). The member is the u4000 checkpoint (`S1-004000-student.pt`). Every Silver
and Bronze row from a session of a natural validation clip is removed first. `train_s1.py` applies the torch 2.8 MPS
LayerNorm patch (`wavlm_mps_patch.py`).

### 6.3 R10 and PSR10B (Kaggle, 2 × T4)

Each kernel is one Kaggle script with the datasets of section 5.7 attached (internet off). It trains two arms,
one per GPU, with the same draws.

- `training/kernels/kernel_r10.py`, arm **R10** (cuda:1; arm A on cuda:0): P2 continued. Per update 8 Gold +
  2 generated clips (weight 1.0); 8 Silver-stream clips (weight 0.4) with per-event modes `full`, `presence`,
  `ignore` and `onset` (the Silver stream includes the 1,887 Bronze wrap-all rows); 32 tag-only clips (weight 0.1)
  plus a Bronze absent-category term. Encoder peak 1e-5 (warmup 1,000 updates), head 1e-4, plateau to D, cosine
  to 0.05 × peak at T. Our run: T 15,500, D 12,500. The member is the checkpoint at update 15,500.
- `training/kernels/kernel_psr10b.py`, arm **PSR10B** (cuda:1; arm PSR10 on cuda:0): the same, but every Silver
  event uses P2's full objective, plus Bronze. The member is the checkpoint at update 15,000.

The kernels keep fp16 weights. We cast the members to fp32 before the next round:

```
python training/kernels/cast_ckpt_fp32.py R10/ckpt-015500.pt R10-u15500-fp32.pt
python training/kernels/cast_ckpt_fp32.py PSR10B/ckpt-015000.pt PSR10B-u15000-fp32.pt
```

### 6.4 Training rounds L, L2 and L3 (`training/kernels/kernel_rounds.py`)

`kernel_rounds.py` is `kernel_psr10b.py` plus three switches at the top of the file. Set `OM_MODE = True`, then
`OM_STAGE` (and `OM3_PAIR` for stage 3), and run one Kaggle session per row. The `member-inits` dataset holds the
start weights under the file names below (the kernel checks their sha256).

| `OM_STAGE` | Round | Arms (cuda:0, cuda:1) | Start weights | Strong batch A per update |
| --- | --- | --- | --- | --- |
| 1 | L | R10-L, PSR10B-L | `R10-u15500-fp32.pt`, `PSR10B-u15000-fp32.pt` | 8 Gold + 1 generated + 5 LIBSYN-v1 |
| 2 | L | P2-L, S1-L | `parent-p2.pt`, `S1-004000-student.pt` | 8 Gold + 1 generated + 5 LIBSYN-v1 |
| 3, `OM3_PAIR = 'A'` | L2 | R10-L2, PSR10B-L2 | `R10-L-u4000-fp32.pt`, `PSR10B-L-u4000-fp32.pt` | 8 Gold + 1 generated + 8 LIBSYN-v2 |
| 3, `OM3_PAIR = 'B'` | L2 | P2-L2, S1-L2 | `P2-Lb-u4000-fp32.pt`, `S1-Lb-u4000-fp32.pt` (the stage-2 weights) | 8 Gold + 1 generated + 8 LIBSYN-v2 |
| 5 | L3 | P2-L3, S1-L3 | `P2-L2-u4000-fp32.pt`, `S1-L2-u4000-fp32.pt` | 8 Gold + 1 generated + 8 LIBSYN-v2 (second pass) |

Every round, every update also has 8 Silver clips (weight 0.4) and 32 tag-only clips (weight 0.1). Each round
runs 4,000 updates (plateau end D 3,000) from fresh AdamW, encoder peak 5e-6 and head peak 5e-5 (warmup 200).
P2-L* arms use the R10 arm spec and S1-L* arms the PSR10B arm spec. A round evaluates on the 2,501 validation
clips, writes fp32 weights at D and T, and makes no test prediction. The C8 members are the u4000 weights of
P2-L3, S1-L3, R10-L2 and PSR10B-L2.

### 6.5 950232 fallback donors R10S1 and SOUP2

R10S1 is a second R10 run on the Mac (seed offset 10,000), with the kernel recipe ported line for line
(`psr10_port.py` imports `kernel_psr10b.py` and checks its sha256):

```
export T1_MAC_INPUT=/path/to/mac-input     # the kernel datasets, laid out like /kaggle/input
mkdir -p system_950232/donors/runs
echo '{"kind": "mac_psr10_precision_v1", "amp": "bf16"}' > system_950232/donors/runs/PRECISION.json
python system_950232/donors/mac_train.py --run R10S1 --arm R10 --seed-offset 10000 --T 15500 --D 12500 \
    --amp bf16 --precision-file system_950232/donors/runs/PRECISION.json --encoder-device mps
python system_950232/donors/export_frames.py --run-dir system_950232/donors/runs/R10S1 --which T --encoder-device mps
python system_950232/donors/make_soup.py --a R10-u15500-fp32.pt --b system_950232/donors/runs/R10S1/final-015500.pt \
    --out SOUP2.pt --report SOUP2-report.json
```

SOUP2 is the fp32 average of R10 and R10S1 (R10S1 rounded to fp16 first, like R10).

## 7. Inference and the C8 vote

Inference runs on CPU. Put the member checkpoints of the v1.0 release (`P2L3.pt`, `S1L3.pt`, `R10L2.pt`,
`PSR10BL2.pt`) in `members/` and extract the test audio:

```
unzip input_data.zip -d test_input                 # test_input/audio/*.wav
cd inference && python -c "from indoml_pipeline.build_submission import durations_for; durations_for('test')" && cd ..
cp $INDOML_WORK/test-durations.json .              # wav-header durations of the 5,517 clips
python -c "import json; print('\n'.join(sorted(k[:-4] for k in json.load(open('test-durations.json')))))" > test_clips.txt
```

Per-frame outputs of every member at the 4 input shifts (in samples: 0, 160, 320, 480):

```
for m in P2L3 S1L3 R10L2 PSR10BL2; do
  for s in 0 160 320 480; do
    python inference/export_frames.py --checkpoint members/$m.pt --audio test_input/audio --shift $s \
        --out frames/$m/test/frames_shift$s.npz
  done
done
```

The C8 vote, run under the audit, and the zip:

```
python inference/audit_guard.py --split test --clips test_clips.txt --durations test-durations.json \
    --frames-root frames --members P2L3,S1L3,R10L2,PSR10BL2 --no-e0 --fb-donors R10L2,PSR10BL2 \
    --out C8.predictions.jsonl
python inference/make_zip.py --pred C8.predictions.jsonl --zip C8.zip --audio-zip input_data.zip \
    --durations test-durations.json
```

`audit_guard.py` runs `vote.py` twice and writes `C8.predictions.guard.json`. It fails when the process opens a
forbidden file (template, router, copy or label files, or any audio), imports a template, router or matched-filter
module, or when the output changes after every clip id is replaced by an opaque token. `vote.py` alone takes the
same arguments. `make_zip.py` writes a zip with one `predictions.jsonl` and checks every clip and every event
against the WAV headers in the test zip.

On the validation release, `t1score.py` gives the organizer score with per-subset numbers:
`T1_VALIDATION_DIR=/path/to/validation python inference/t1score.py predictions.jsonl`.

## 8. The 950232 additions

The code of this section is the code that built 950232, with paths turned into arguments and variables. It is
reference code: some steps ran on stored intermediate files (section 10).

### 8.1 Peak router

`system_950232/peak_router.py` reads only the int16 samples: a clip is natural when its maximum is 32767 or its
minimum is −32768 (cast to int32 first; `np.abs` on int16 overflows at −32768). On validation it is right for
2,501 of 2,501 clips; on test it gives 3,437 natural and 2,080 synthetic clips. This works because the organizers
peak-normalised every natural clip to full scale; we report it as a data artefact, not as a method.

```
python system_950232/peak_router.py input_data.zip            # self-test and the route counts
python -c "import sys; sys.path.insert(0, 'system_950232'); import peak_router as r; rows, _ = r.route_zip('input_data.zip'); print('\n'.join(sorted(c for c, v in rows.items() if v['route'] == 'natural')))" > natural_clips.txt
```

### 8.2 Natural clips

Members: P2, S1, R10 (twice), PSR10B (twice) at 4 shifts = 24 event sets (base checkpoints: P2 u4972, S1 u4000,
R10 u15500, PSR10B u15000). Stage order:

| Stage | What it does |
| --- | --- |
| N1 + LTL6 | per set: decoder 0.75 and refiner as in C8; LTL6: in clips with D ≥ 6 s, a candidate ≥ 2 s with min(score, peak) ≥ 0.60 gets score and peak 0.75 |
| N2 + LACC + LPAD8 | vote over the 24 sets (support ≥ 12 of 24); LACC: clusters ≥ 1 s need 11; LPAD8: in clips with D ≥ 8 s, kept events ≥ 1 s get 7.5 % of their length at each end |
| N3 | clamp to [0, D], round to 0.1 ms |
| N4 E0 pairing | one-to-one pairs with the E0 events at IoU ≥ 0.5; base events shorter than 2 s take the 50/50 average |
| N5 Q1 copies | a library template found in the clip (tile-GLRT S3 ≥ θ_k) replaces the overlapping events with the template extent |
| N6 fallback | empty clip: R10, then R10S1, then SOUP2 (each at 0.55, support ≥ 2 of 4), then [0, D] |
| post LD1R | clips with D ≥ 8 s and no event ≥ 0.9 D: if ≥ 6 of the 8 P2 / S1 sets have a candidate ≥ 0.9 D with min(score, peak) ≥ 0.65, the longest event becomes [0, D] (not on Q1 clips) |

Frames for the 6 roles come from `inference/export_frames.py` (for P2: `--checkpoint parent-p2.pt`) and, for R10S1,
from `system_950232/donors/export_frames.py`. Then:

```
python inference/vote.py --split test --clips natural_clips.txt --durations test-durations.json --frames-root frames \
    --c4 --e0 e0-test-predictions.jsonl --fb-donors R10,R10S1,SOUP2 --out nat.predictions.jsonl
python system_950232/copies/apply_q1.py --final nat.predictions.jsonl --n4 nat.predictions.N4.jsonl \
    --durations test-durations.json --theta system_950232/wmf/theta.json \
    --scan-stores system_950232/copies/cache/scan_testnat system_950232/wmf/cache/scan_testsyn \
    --out nat_q1.predictions.jsonl --copies-out q1_copies.json
```

`vote.py` has no N5. `apply_q1.py` replaces the line of every clip with a copy by the N5 result of its N4 line. This
equals the deployed order because N6 and LD1R never act on a clip with a copy. The Q1 thresholds θ_k come from
natural audio only (`wmf/w08_null.py`: the 1,806 natural validation clips and 4,000 natural training clips). The
scan rows come from `copies/ncscan.py` (`build-testnat`, then `scan testnat <workers>`) and, for the clips that the
file-name split calls synthetic, from the WMF scan of section 8.3.

**E0 end refiner** (WavLM-Large, frozen, mean of its 25 hidden layers, with a RED head):

```
T1_WAVLM_ASSETS=/path/to/wavlm-large python system_950232/e0/frozen_wavlm_red.py run --output e0-stage1
python system_950232/e0/prepare_e0_inputs.py --out e0-inputs --head e0-stage1/head-seed73 --test-audit /path/to/test-audit
```

Stage 1 trains the head on frozen features of the inner split (seed 73, epoch 5 kept). The Kaggle jobs
`full_wavlm_job.py` (training on the full training release, 8 weak passes, selection on the validation clips) and
`predict_wavlm_job.py` (test prediction) need internet: they install torch 2.8.0 and transformers 4.57.1 and
download WavLM-Large at a pinned revision. Their `predictions.jsonl` is the `--e0` input of `vote.py`.

### 8.3 Synthetic clips

| Step | Code | What it does |
| --- | --- | --- |
| S0 template decode | `inference/indoml_pipeline/templates.py`, `build_submission.py` | NCC scan with the 203 library templates; greedy decode at per-template thresholds fitted on validation |
| J2 master extents | `system_950232/job2/` | a paste of a long noise file starts at the file start and ends at min(onset + file length, clip end): machine 5.437 s, child 5.211 s, bird ≥ 9.077 s; a prefix scan finds truncated pastes (threshold 0.5 per master) |
| WMF | `system_950232/wmf/` | whitened matched filter with a tile GLRT (S3): removes events whose template is not there, re-chooses templates and adds pastes; thresholds from natural audio only |
| Q2 | `system_950232/copies/q2_*.py`, `kjob.py`, `run_cand.sh` | 2,849 kernels cut from labelled natural validation events, searched in the free regions; rule and thresholds from validation only |
| SR1 | `system_950232/sr1/sr1lib.py` | snaps each event to the label grid: one integer offset per template, fit on validation (at most 1.1 ms) |

Commands, in order (each script's docstring gives its inputs):

```
export T1_JOB2_OUT=/path/to/job2-out T1_STORE=/path/to/store
cd inference
python -m indoml_pipeline.templates all                         # library, exemplars, validation scan, thresholds
python -m indoml_pipeline.templates scan --split test
python -m indoml_pipeline.build_submission --split test --natural <natural lines> --out $INDOML_WORK/test/predictions.jsonl
cd ../system_950232/job2
python audio_store.py val test
python m03_val_extent.py; python m04_test_extent.py; python m05_prefix_scan.py
python m06_prefix_events.py; python m07_job2_apply.py          # -> $T1_JOB2_OUT/base_test_job2.pkl
python -c "import os, pickle; d = os.environ['T1_JOB2_OUT']; pickle.dump({'pred': pickle.load(open(d + '/base_test_job2.pkl', 'rb'))}, open(d + '/final_job2only.pkl', 'wb'))"
python s40_emit_v2.py $T1_JOB2_OUT/final_job2only.pkl $T1_JOB2_OUT/test_synthetic_predictions_v2_job2only.jsonl
cd ../wmf
python w00_stores.py valsyn valclean valnat train testsyn; python w01_base.py
python w02_scan.py <store> <workers>                            # each store
python w03b_requests.py; python w03_atlag.py <store> <request.pkl> <out.pkl> <workers>
python w04_events.py; python w07_val.py; python w08_null.py     # theta.json
python w09_test.py '<config json>'
cd ../copies
python q2_kernels.py; bash run_cand.sh; python q2_null.py; python q2_test.py
```

S0 and the later steps work on the 2,082 clips that the file-name router of `indoml_pipeline` calls synthetic
(`build_submission.py` decodes them; the natural lines of that file are not used). The final lines follow the peak
router (section 10). For the rescan file of `s40_emit_v2.py`, no template is added in 950232, so its `pred` is
the event table of `m07_job2_apply.py`. Deployed counts on
the test set: N4 changes 2,890 lines; N5 adds 87 copies in 77 clips; N6 fills 61 clips (R10 38, R10S1 7, SOUP2 2,
floor 14); SR1 changes 2,011 of the 2,080 synthetic lines. The 950232 file is the union of the natural and
synthetic lines, sorted by clip id (`make_zip.py` checks it).

## 9. Pre-trained dependencies

| Item | Use | Source | Licence | sha256 |
| --- | --- | --- | --- | --- |
| ATST-F_strong_1.pt (ATST-Frame) | encoder of every member | github.com/fschmid56/PretrainedSED, release v0.0.1 | MIT | `fbf2577958e3648d55ee8cea7e0e5260c4505fb0c13964e1ec81bc367cee5eda` |
| PretrainedSED code | ATST-Frame model code (cloned, not included) | github.com/fschmid56/PretrainedSED @ `1aa47e482f7e89904cba2338999345025d8b4e36` | MIT, © 2024 Florian Schmid | (git commit) |
| WavLM-Large | frozen encoder of E0 (950232 only) | huggingface.co/microsoft/wavlm-large @ `c1423ed94bb01d80a3f5ce5bc39f6026a0f4828c` | model card (the microsoft/UniSpeech licence) | `pytorch_model.bin` `fdee460e529396ddb2f8c8e8ce0ad74cfb747b726bc6f612e666c7c1e1963c9d`; `config.json` `a3d8fe831aaf63d725b54a8ac36f3549cd4365c5086774b2c89cabbc6f9e129d`; `preprocessor_config.json` `60ca5a31e13f69ee2fbf147504c8676db5f6398fd7a6b12294341dff838edfcf` |
| DeepFilterNet3 | cleans the LIBSYN speech beds (Track 2 scripts) | github.com/Rikorose/DeepFilterNet @ `d375b2d8309e0935d165700c91da9de862a99c31`, `models/DeepFilterNet3.zip` | MIT OR Apache-2.0 | zip `49c52edc8947ae1f9bf50d81530beaf3a2c3245aeaf34b6f31ff535cd22284d2`; `model_120.ckpt.best` `23b92884f63ccf54bb026014604625ab231657b6480df65db4095c4c171e6003` |
| Silero VAD v5 | speech mask of the beds (Track 2 scripts) | github.com/snakers4/silero-vad @ `1e261b036686cd0017d500ee96acd1c4ba572a9d`, `src/silero_vad/data/silero_vad.jit` | MIT | `e1122837f4154c511485fe0b9c64455f7b929c96fbb8d79fbdb336383ebd3720` |
| Vaani Noise-Event Dataset | training data (Gold, Silver, Bronze) | huggingface.co/datasets/ARTPARK-IISc/Vaani-Noise-Event-Dataset @ `be488e2ac12fd62bef46b9f83e3a5feded575333` | CC BY 4.0 (dataset card; access request) | 182 parquet files |
| Organizer validation release | validation; noise library; LIBSYN statistics; folds; model selection | competition page | competition terms (not redistributed) | |
| Organizer test inputs | inference only (no labels) | competition page | competition terms (not redistributed) | |

We use no other pre-trained model.

### 9.1 Our weights (v1.0 release)

The [v1.0 release](https://github.com/Ayush-004/indoml-2026-datathon/releases/tag/v1.0) holds our Track 1 weights (`SHA256SUMS.txt` there has the full sha256
values). Each `.pt` file is a dict with the fp32 weights in `model_state` and the run settings beside them.

| File | System | What it is | Where it goes | sha256 (first 16) |
| --- | --- | --- | --- | --- |
| `P2L3.pt` | C8 | member P2-L3, update 4,000 | `members/` (section 7) | `daf63ace65a8a286` |
| `S1L3.pt` | C8 | member S1-L3, update 4,000 | `members/` | `a0687743a2ee3350` |
| `R10L2.pt` | C8 | member R10-L2, update 4,000; first empty-clip donor | `members/` | `d535feed15d46071` |
| `PSR10BL2.pt` | C8 | member PSR10B-L2, update 4,000; second empty-clip donor | `members/` | `bd17083832711887` |
| `parent-native-e5.pt` | C8, 950232 | E5: the architecture and the RED head config | `native-bundle/assets/` (section 4.3) | `819dcf00eacf6fdc` |
| `parent-p2.pt` | 950232 | P2, update 4,972; the start of every member | `p2-parent/` (section 4.4) | `419c441e911419b3` |
| `S1-004000-student.pt` | 950232 | S1, update 4,000 | frame export (section 8.2) | `7ca81a17579969cc` |
| `R10-u15500-fp32.pt` | 950232 | R10, update 15,500; first fallback donor | frame export | `e0684e5951505371` |
| `PSR10B-u15000-fp32.pt` | 950232 | PSR10B, update 15,000 | frame export | `0fdae2609cdb9a3a` |
| `R10S1-final-015500.pt` | 950232 | R10S1, update 15,500 (section 6.5); second fallback donor | frame export | `3d603bef3c1998e9` |
| `SOUP2.pt` | 950232 | mean of R10 and R10S1 (section 6.5); third fallback donor | frame export | `76ba6eaccac073b4` |
| `e0-head.zip` | 950232 | E0 head `selected.pt` (for frozen WavLM-Large) and its protocol files | input of `e0/predict_wavlm_job.py` | `fe8bd70e164f4449` |

Eight files are saved again for this release with shorter metadata: local file paths are cut to the file names,
and `R10S1-final-015500.pt` keeps only the model weights and the run settings (no optimizer, scaler or RNG
state). Their tensors are bit-identical to the files we used. `parent-p2.pt`, `S1-004000-student.pt`,
`R10-u15500-fp32.pt` and the E0 files are byte-identical. `training/kernels/kernel_rounds.py` pins the sha256 of
the released `PSR10B-u15000-fp32.pt`.

## 10. Limitations

- **No data.** The repository has no audio, labels or predictions; our checkpoints are in the v1.0 release
  (section 9.1). Several scripts check the sha256 of their inputs against our files (for example the training
  audit, the recipe files and the round start weights); a rebuild with other inputs stops at these checks until
  the constants are updated.
- **Hardware.** P2, R10, PSR10B, the rounds and E0 ran as Kaggle sessions on 2 × T4. S1, R10S1 and the donor
  frame export ran on an Apple M-series GPU (MPS, torch 2.8, with the LayerNorm patch). The vote is CPU and numpy.
- **Reference-only parts.**
  - `training/p2/full_train.py` needs `fusion.py` and the mixed and BEATs bundles of our P2 run, which are not
    in this repository. Steps 2-4 of the P2 lineage (section 6.1) ran with earlier revisions of the `indoml`
    package; the step-3 checkpoint is not kept.
  - The organizer-audit manifests of section 5.3 and the E0 test manifests come from a local audit of the
    organizer zips; that script is not included (the fields are given above).
  - `mac_train.py` mentions a small queue supervisor, a precision helper (`precision.py`) and a `tests/` folder;
    they are not included. The precision file of section 6.5 has the content that helper wrote.
  - 950232 used stored intermediate files. The synthetic lines are the stored lines of 2,079 clips plus one clip
    that the peak router sends to the synthetic branch and that took a per-clip path; the per-clip script, the
    SR1 driver and the final assembly script are not included. Three natural test clips of 950232 were built by a
    separate step that `vote.py` does not reproduce. `vote.py` matches the deployed natural builder on the 1,740
    natural validation clips without a library copy; we have no record of a test run of `vote.py` itself.
  - `job2/m06_prefix_events.py` takes the master thresholds from natural peaks of validation and test clips
    (label-free). A fit without test audio gives the same thresholds (0.5 for all three masters).
- **Other notes.**
  - `prepare_test_audio.py` uses the file-name router only to choose the clips of the `test-natural` dataset. The
    kernels use that dataset for a no-overlap check and for test predictions that no submission uses.
  - `train_c2.py` also has an optional consistency stream on unlabelled test audio. S1 trains with this stream
    off (weight 0) and reads no test audio. No member of C8 or 950232 uses it.
  - The library templates, the LIBSYN statistics, the gold_synth level statistics and the recipe vocabulary come
    from the labelled validation release; no test label exists or is used.
  - `t1score.py` and `indoml_pipeline/score.py` re-implement the organizer reference scorer; `copies/nclib.py`
    loads the organizer scorer files from `T1_OFFICIAL_SCORER_DIR` (not included).
  - Code names in docstrings: "K-A" is the R10 kernel and its pools, "K1" the per-event Silver modes, "PS-R10" the
    PSR10 / PSR10B kernel, "OM1" to "OM5" the round stages, and "generated" the gold_synth clips.
- **LIBSYN and the test set.** Copies of library templates also occur in natural test clips (87 copies in 77
  clips), so LIBSYN can teach those events (worth up to about +0.004).

## 11. Licence

Our code is under the MIT licence of this repository (see `LICENSE` in the repository root). PretrainedSED
(MIT, © 2024 Florian Schmid) is cloned at setup and is not part of this folder. The other models and datasets keep
the licences in section 9.
