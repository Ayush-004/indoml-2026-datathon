# Track 2: Noise Event Removal

Team sonicboom

This folder holds the code of our Track 2 systems for the IndoML 2026 Datathon
([Codabench competition 17835](https://www.codabench.org/competitions/17835/)).

We have two systems. They share one pipeline and one set of trained models:

- **T4** is the default. It is model-only: every clip takes the same learned path.
- **E3** is the setting of our Phase 2 leaderboard entry. It adds a route rule and a pass-through step.

## 1. Task and score

Each input clip is noisy speech (16 kHz, mono, PCM16). Some clips are synthetic: clean speech plus noise events,
with a hidden clean reference. The other clips are natural recordings.

For each clip, a system writes:

- one enhanced WAV: 16 kHz, mono, PCM16, with the same file name and the same number of samples as the input;
- a transcript of the enhanced WAV, made by the mandated ASR model `ARTPARK-IISc/SraVaani-1.0`.

Score = SI-SDR + ΔWER.

- **SI-SDR**: the mean SI-SDR (dB) of the enhanced synthetic clips against their clean references.
- **ΔWER**: the WER drop in percentage points (WER of the noisy clip minus WER of the enhanced clip):
  0.32 × the drop on the synthetic clips + 0.68 × the drop on the natural clips.

The Phase 2 test set has 3,178 clips.

## 2. Systems and results

Both systems run `inference/enhance.py` with the same three fine-tuned models.

### T4 (default, `--system t4`)

1. **ZipEnhancer** (fine-tuned, 2.0 M parameters) enhances the whole clip in 4 views: the clip, the clip delayed by
   50 samples, the clip with its sign flipped, and both. Each output is aligned back and its sign is restored.
   The 4 outputs are averaged. Result: `z`.
2. **TF-GridNet v8r-M** (fine-tuned, 8.5 M parameters) enhances the clip. Result: `t`. A least-squares gain
   `g = <z, t> / <t, t>` matches the level of `t` to `z`.
3. **Fusion head** (about 29k parameters). It predicts a weight `w` in [0, 1] for each time-frequency bin
   (STFT 512, hop 128, Hann). Its inputs per bin are log|X|, log|Z|, log|G| (each minus the clip mean of log|X|) and
   cos(∠Z − ∠G), where X, Z and G are the STFTs of the input, `z` and `g·t`. Output: the inverse STFT of
   `w·Z + (1 − w)·G`.
4. The last 64 samples of the output come from `z` only (the gain `g` also leaves them out).
5. If the output exceeds the int16 range, the whole clip is scaled down.
6. **SraVaani-1.0** transcribes the enhanced WAV.

Clips longer than 20 s run in 20-s windows with 1-s linear cross-fades. If a model runs out of GPU memory on a clip,
the clip runs again in 20-, 10- or 5-s windows with the same models. No rule chooses the processing per clip, and no
part of the input passes through unchanged. If one model fails on a clip, the output of the other model is used.
Any other failure stops the run.

### E3 (`--system e3`)

E3 is the same pipeline with these options:

- one ZipEnhancer view instead of 4;
- a **route rule**: if at least 20 % of the frames of the ZipEnhancer output are pass-through frames (next
  point), the clip takes the ZipEnhancer output alone; the other clips take the fusion path;
- a **pass-through step** on every output: where the model changes a 16-ms frame by less than 20 dB, the frame
  takes the input, scaled by a least-squares gain (5-ms cross-fades);
- on any other failure (after the model fallback), the input is written unchanged.

### Results

| System | Data | SI-SDR | WER | Score |
|---|---|---|---|---|
| T4 | VH695 | 15.62 dB | 32.64 % | - |
| T4 | VS695 | 18.78 dB | - | - |
| T4 | NAT1806 | - | 28.28 % | - |
| E3 | Phase 2 test, 3,178 clips (Codabench submission 961873) | 13.20 dB | ΔWER 2.11 | **15.31**, 10th of 20 on the final board |

- **VS695**: the 695 synthetic clips of the organizer validation set.
- **VH695**: the same 695 clips at a harder level: clean reference + 1.68 × (noisy − clean), so the noise is 4.5 dB
  louder. A mix with a peak above 32,440 is scaled down.
- **NAT1806**: the 1,806 natural clips of the organizer validation set.
- WER is the WER of the SraVaani-1.0 transcripts of the enhanced clips.

T4 has validation numbers only. It is not on the leaderboard.

## 3. Folder map

```
track2_noise_removal/
  main.py                            inference entry point: enhancement, then ASR
  requirements.txt
  inference/
    enhance.py                       enhancement code (T4 and E3 settings come from environment variables)
    sravaani_asr.py                  the mandated SraVaani-1.0 call -> transcripts.jsonl
    build_submission.py              Codabench ZIP (WAVs + transcripts.jsonl) with format checks
  training/
    train_zipenhancer_stage1.py      ZipEnhancer stage 1 (SI-SNR loss)
    train_zipenhancer_stage2_asr.py  ZipEnhancer stage 2 (SI-SNR + ASR-feature losses)
    train_tfgridnet.py               TF-GridNet fine-tuning (SI-SNR loss)
    dump_fusion_mixtures.py          fusion-head data: fresh mixtures through both frozen enhancers
    fusion_head.py                   fusion-head model and training
    pp_lib.py                        pass-through function (picks the fusion-head training crops)
    data/
      build_banks.py                 speech-bed and noise-event banks from the Vaani training release
      bank_vad.py                    Silero VAD speech share of each noise event
      build_clean_beds.py            clean speech beds (DeepFilterNet3 + Silero VAD mask)
      build_public_noise.py          public noise bank (DEMAND, ESC-50, MUSAN)
      build_valnewlib.py             val-newlib selection set (validation speech + held-out public noise)
  third_party/
    zipenhancer/                     ZipEnhancer code (ModelScope, Apache-2.0); base weights go here
    espnet_universal/                ESPnet code for TF-GridNet (Apache-2.0); base weights go here (CC BY 4.0)
    deepfilternet3/                  DeepFilterNet3 code (MIT OR Apache-2.0); training data preparation only
  checkpoints/                       our fine-tuned checkpoints go here (v1.0 release)
```

The entry points are `main.py` and the scripts in `inference/`, `training/` and `training/data/`. The files in
`third_party/` are libraries that these scripts import. The training scripts read their settings from `T2_*`
environment variables; each run writes its settings to `history.json`.

## 4. Setup

Python 3.12 (tested with 3.12.13).

```
python3.12 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

Then get the model files:

1. ZipEnhancer base weights from ModelScope: see `third_party/zipenhancer/README.md`.
2. TF-GridNet base weights and config from Hugging Face: see `third_party/espnet_universal/README.md`.
3. Our three fine-tuned checkpoints: `track2_checkpoints.zip` of the [v1.0 release](https://github.com/Ayush-004/indoml-2026-datathon/releases/tag/v1.0).
   Unzip it in this folder (`unzip track2_checkpoints.zip`). `checkpoints/README.md` gives the layout and the
   sha256 values.
4. SraVaani-1.0: the Hugging Face model needs an access request (the approval is automatic). Log in on the machine
   (`hf auth login`), or download a snapshot (commit `f5dd5358325a5208775b91dad98918e079ea2b27`) and give its
   folder with `--asr-model`.

The ZipEnhancer and ESPnet code is in `third_party/`. The `modelscope` and `espnet` packages are not needed.

## 5. How to run

```
python main.py INPUT_DIR OUTPUT_DIR                   # T4
python main.py INPUT_DIR OUTPUT_DIR --system e3       # E3
```

- `INPUT_DIR`: a folder tree with the input WAVs (16 kHz, mono, PCM16). The file names must be unique. WAVs in
  folders named `syntheticCleanRefAudio` are skipped.
- `OUTPUT_DIR`: one enhanced WAV per input WAV (same name, same length) and `transcripts.jsonl` with one
  `{"clip_id", "text"}` line per clip.

| Flag | Default | Meaning |
|---|---|---|
| `--system t4\|e3` | `t4` | system settings (section 2) |
| `--device auto\|cuda\|mps\|cpu` | `auto` | device of the enhancers (auto: CUDA, then MPS, then CPU) |
| `--chunk N` | `100` | clips per runner process (a fresh process for each chunk keeps the memory low) |
| `--no-asr` | off | write the WAVs only |
| `--asr-model ID_OR_DIR` | `ARTPARK-IISc/SraVaani-1.0` | Hugging Face id or a local snapshot folder |
| `--ckpt-dir DIR` | `checkpoints/` | folder with the three fine-tuned checkpoints |
| `--zipenhancer-dir DIR` | `third_party/zipenhancer` | ZipEnhancer code and the official weight file |
| `--espnet-dir DIR` | `third_party/espnet_universal` | ESPnet code and the TF-GridNet base files |

`main.py` links the model folders into a temporary work folder. It runs `inference/enhance.py` once per chunk of
clips, collects the WAVs, and then runs `inference/sravaani_asr.py` on the CPU. The system settings are these
environment variables of `inference/enhance.py`:

| Variable | T4 | E3 | Meaning |
|---|---|---|---|
| `T2_ZIP_TTA` | `s50f` | (empty) | ZipEnhancer views: `s50f` = the 4 views of section 2 |
| `T2_RAVG_THR` | `2.0` | `0.20` | route rule: ZipEnhancer alone if its pass-through share is at least this value (above 1 = never) |
| `T2_PP` | (empty) | `pt20` | pass-through step with a 20-dB threshold |
| `T2_NO_RAW_FALLBACK` | `1` | (empty) | `1` = stop on a failure instead of writing the input |

Both systems use the same values for the other settings: the fusion head (`T2_FUSE_HEAD`), the 64-sample end rule
(`T2_RAVG_EDGE`), the 20-s windows (`T2_WIN_MAX_S`, `T2_WIN_S`, `T2_WIN_OV_S`) and the int16 fit (`T2_FIT_PEAK`).

To make a Codabench ZIP from the outputs (the script checks the format and does not upload anything):

```
python inference/build_submission.py --input-dir TEST_WAV_DIR --enhanced-dir OUTPUT_DIR \
    --hyp OUTPUT_DIR/transcripts.jsonl --out submission.zip
```

It checks one WAV per test clip (PCM16, 16 kHz, mono, input length), one transcript per clip, and that the
transcripts are newer than the WAVs. All files go to the ZIP root.

## 6. How the models are trained

### 6.1 Data

We train on public data only: no test audio and no organizer validation audio. We use the organizer validation
set only to select checkpoints.

- **Clean targets (speech beds).** Event-free speech spans of the Vaani Noise-Event training release: all verified
  (Gold) clips and 14 h of unverified (Silver) clips. DeepFilterNet3 cleans each source clip (full suppression).
  A Silero VAD mask then sets the pauses to exact zeros (speech probability >= 0.35, ±128 ms, 40-ms raised-cosine
  edges), as in the organizer clean references. Result: 25,280 beds, 24.7 h.
- **Noise.** Each event comes from one of four sources:
  - Vaani environment events (paired tags with no words inside), probability 0.15;
  - Vaani point-tag events (breathing, cough, ...; at most 6,000), probability 0.25;
  - procedural noise (engine, machine, fan, rumble, hiss, tones, music), probability 0.20;
  - public noise, probability 0.40: ESC-50 folds 1-4 (0.50), MUSAN noise (0.25), DEMAND (0.15) and MUSAN music
    without vocals (0.10).

  Vaani events count only when their Silero VAD speech share is below 0.2.
- **Mixtures.** A crop of a speech bed plus 1-3 noise events. Event SNR ~ N(-8 dB, 6 dB), clipped to [-20, 12] dB,
  with 5-ms fades. 5 % of the mixtures have no noise. The mixtures are made on the fly.
- **Leakage rule.** Every Vaani clip of a recording session that occurs in the organizer validation set is
  dropped from the banks.
- **Held-out noise.** DEMAND TMETRO and NPARK and ESC-50 fold 5 are not in training. They make the val-newlib
  selection set.

### 6.2 Data preparation

Inputs:

- `VAANI`: a local copy of the Vaani Noise-Event training release (`ARTPARK-IISc/Vaani-Noise-Event-Dataset`,
  revision `be488e2ac12fd62bef46b9f83e3a5feded575333`, 90,630 clips): `manifest.jsonl` (one row per clip with
  `clip_id`, `audio_path`, `duration`, `quality`, `events`, `language`, `source.metadata.transcript`) and the audio
  files. The Vaani preparation step of the Track 1 folder writes this manifest.
- `VALIDATION`: the organizer validation set (`validationMetadata.json`, `syntheticNoiseAudio/`,
  `syntheticCleanRefAudio/`, `naturalNoisyAudio/`) from the competition page.
- `PUBLIC`: DEMAND, ESC-50 and MUSAN (section 7), laid out as `demand/<ENV>_16k/.../ch01.wav`,
  `esc50/<...>/meta/esc50.csv` + `audio/`, `musan/musan/{noise,music}/`.
- `silero_vad.jit`: Silero VAD v5 (section 7).
- DeepFilterNet3 weights in `third_party/deepfilternet3/weights/` (see its README).

```
python training/data/build_banks.py --manifest VAANI/manifest.jsonl \
    --val-meta VALIDATION/validationMetadata.json --out DATA_ROOT/noise_bank
python training/data/bank_vad.py --vaani-dir VAANI --bank DATA_ROOT/noise_bank/noise_bank.jsonl \
    --silero-jit silero_vad.jit --out DATA_ROOT/noise_bank_vad
for i in 0 1 2 3; do             # clean beds in 4 parallel parts, then one merge
  python training/data/build_clean_beds.py --part $i/4 --vaani-dir VAANI \
      --beds DATA_ROOT/noise_bank/speech_beds.jsonl --silero-jit silero_vad.jit --work beds_work &
done; wait
python training/data/build_clean_beds.py --merge --work beds_work --out DATA_ROOT/clean_beds
python training/data/build_public_noise.py --src PUBLIC --out DATA_ROOT/public_noise
python training/data/build_valnewlib.py --validation-dir VALIDATION --public-noise-src PUBLIC \
    --out DATA_ROOT/val_newlib
```

Silero VAD v5: `curl -L -o silero_vad.jit https://raw.githubusercontent.com/snakers4/silero-vad/1e261b036686cd0017d500ee96acd1c4ba572a9d/src/silero_vad/data/silero_vad.jit`
(sha256 `e1122837f4154c511485fe0b9c64455f7b929c96fbb8d79fbdb336383ebd3720`).

The training scripts find their inputs by folder name below `T2_INPUT` (the data root):

```
DATA_ROOT/
  noise_bank/           build_banks.py output
  noise_bank_vad/       bank_vad.py output
  clean_beds/           build_clean_beds.py --merge output
  public_noise/         build_public_noise.py output
  val_newlib/           build_valnewlib.py output
  validation/           the organizer validation set (a folder or validation.zip)
  helpers/val_order.json
  sravaani/             a snapshot of ARTPARK-IISc/SraVaani-1.0 (commit f5dd5358...)
  zipenhancer/          link to third_party/zipenhancer (with the official weight file)
  espnet_universal/     link to third_party/espnet_universal (with tfgridnet_urgent25/)
```

`helpers/val_order.json` lists the validation file names in sorted order:

```
python -c "import json, pathlib, sys; v = pathlib.Path(sys.argv[1]); n = lambda d: sorted(p.name for p in (v / d).glob('*.wav')); json.dump({'syn_noisy': n('syntheticNoiseAudio'), 'syn_clean': n('syntheticCleanRefAudio'), 'nat_noisy': n('naturalNoisyAudio')}, open(sys.argv[2], 'w'))" VALIDATION DATA_ROOT/helpers/val_order.json
```

### 6.3 Common training procedure

The three enhancer scripts share the data code, the selection and the final stage.

- **Loss and optimizer**: negative SI-SNR on the whole crop. Each batch runs as micro-batches (`T2_MICRO`), which
  gives the same gradient with less memory. AdamW (weight decay 1e-5), a linear warm-up over 150 steps, then a
  cosine decay to the last step; gradient clipping at norm 5.0.
- **Selection** every 500 steps on two sets:
  - val-hard: VH695, every 3rd clip (232 clips);
  - val-newlib: the 695 organizer clean references with held-out public noise at the organizer event spans and
    energies, at the val-hard level; every 6th clip.

  The score of each set is SI-SDR + 0.3203 × the WER drop in points (the synthetic part of the official ΔWER).
  It is measured on the output after the E3 pass-through step. Selection = (2 × val-hard + val-newlib) / 3. The
  best step is kept.
- **Stop check** at step 1000: training stops if the selection score has gained at most 0.2 and the hard clips
  (val-hard SI-SDR of the start model below 20 dB) have gained less than 0.3 dB.
- **Final stage**: a WiSE-FT candidate (0.5 × start weights + 0.5 × best weights) is scored in the same way.
  `final.pt` is the better one. The script also writes the outputs of the final model on val-hard, val-newlib and
  every 2nd natural clip.

The commands below are for an Apple M-series Mac (`T2_DEVICE=mps`). Common environment:

```
export T2_INPUT=DATA_ROOT T2_LOCAL_VAANI=VAANI T2_DEVICE=mps T2_WORKERS=0 T2_BUDGET_H=8
```

`T2_LOCAL_VAANI` points to the local Vaani copy (`manifest.jsonl` + audio). `T2_BUDGET_H` is a time limit in hours.

### 6.4 ZipEnhancer stage 1 (SI-SNR)

Start: the official ModelScope weights. 2,000 steps, batch 4, 3-s crops, learning rate 5e-5, seed 1234 (the
defaults at the top of the script). The best step is 2,000.

```
T2_OUT=runs/zip_stage1 T2_TMP=runs/tmp_zip_stage1 python training/train_zipenhancer_stage1.py
```

### 6.5 ZipEnhancer stage 2 (SI-SNR + ASR-feature losses): checkpoint `zipenhancer_g3asr`

Start: the stage-1 `final.pt`. 3,000 steps, batch 4, 2-s crops, learning rate 3e-5, seed 2234. Two more loss
terms use the frozen SraVaani-1.0 model:

- an L1 loss between the SraVaani front-end features of the output and of the target;
- an L1 loss between the outputs of the frozen SraVaani encoder on these features.

During the first 20 batches only the SI-SNR loss trains. Then each term gets the weight 0.25 × mean |SI-SNR loss| /
mean term of these batches (here 13.70 for the front end and 11.79 for the encoder). The best step is 3,000, and
`final.pt` is the best checkpoint.

```
mkdir -p DATA_ROOT/zip_stage1_init && cp runs/zip_stage1/final.pt DATA_ROOT/zip_stage1_init/best.pt
T2_OUT=runs/zip_stage2 T2_TMP=runs/tmp_zip_stage2 T2_INIT_KERNEL=zip_stage1_init \
T2_CROP=2.0 T2_LR=3e-5 T2_STEPS=3000 T2_SEED=2234 T2_W_FE=0.25 T2_W_ENC=0.25 \
    python training/train_zipenhancer_stage2_asr.py
cp runs/zip_stage2/final.pt checkpoints/zipenhancer_g3asr/final.pt
```

### 6.6 TF-GridNet: checkpoint `tfgridnet_v8rm`

Start: the URGENT 2025 weights `kohei0209/tfgridnet_urgent25`. 2,000 steps, batch 4, 2-s crops, learning rate 3e-5,
seed 1234. The selection scores the E3-style output: the route rule and an equal-weight average with fixed
stage-1 ZipEnhancer outputs, then the pass-through step. The best step is 1,500, and `final.pt` is the best
checkpoint.

```
mkdir -p DATA_ROOT/zipref
cp runs/zip_stage1/wav_hard_best.zip DATA_ROOT/zipref/zipref_hard.zip
cp runs/zip_stage1/wav_new_best.zip DATA_ROOT/zipref/zipref_new.zip
T2_OUT=runs/tfgridnet T2_TMP=runs/tmp_tfgridnet T2_CROP=2.0 T2_LR=3e-5 T2_STEPS=2000 T2_MICRO=1 \
    python training/train_tfgridnet.py
cp runs/tfgridnet/final.pt checkpoints/tfgridnet_v8rm/final.pt
```

### 6.7 Fusion head: checkpoint `fusion_head`

Data: 6,000 fresh training mixtures (3-s crops, seed 7234, the mixture settings of section 6.1) go through the frozen
stage-2 ZipEnhancer (one view) and the frozen TF-GridNet. `dump_fusion_mixtures.py` is the TF-GridNet script with
this dump option. It writes float16 shards of the input, target and both outputs, and then ends.

```
mkdir -p DATA_ROOT/zip_stage2_final DATA_ROOT/tfgridnet_init
cp runs/zip_stage2/final.pt DATA_ROOT/zip_stage2_final/final.pt
cp runs/tfgridnet/final.pt DATA_ROOT/tfgridnet_init/best.pt
T2_OUT=runs/fusion_dump T2_TMP=runs/tmp_fusion_dump T2_DUMP_MIX=6000 T2_SEED=7234 T2_CROP=3.0 \
T2_ENS_ZIP=zip_stage2_final T2_INIT_KERNEL=tfgridnet_init T2_PROBE=1 T2_MICRO=1 T2_SEL_RAVG=0 \
    python training/dump_fusion_mixtures.py
python training/fusion_head.py train runs/fusion_dump/dumpmix checkpoints/fusion_head/head.pt 20 1e-3
```

Training: only the crops on the fusion path of the route rule (ZipEnhancer pass-through share below 0.20): 4,746 of
6,000 crops, split 4,272 train / 474 check. 20 epochs, Adam, learning rate 1e-3, batch 16, negative SI-SNR loss,
gradient clipping 5.0. The last layer starts at weight 0 and bias logit(0.70), so the untrained head gives the
plain average 0.70 z + 0.30 g·t. The head of the last epoch is used. `fusion_head.py` runs on MPS when it is
available, else on the CPU.

## 7. Models and dependencies

| Item | Use | Source | Licence |
|---|---|---|---|
| ZipEnhancer | base of our ZipEnhancer | weights: ModelScope `iic/speech_zipenhancer_ans_multiloss_16k_base` (sha256 `b18896915e27a821...`); code: github.com/modelscope/modelscope @ `6cb35d56` (vendored) | Apache-2.0 |
| TF-GridNet URGENT 2025 | base of our TF-GridNet | Hugging Face `kohei0209/tfgridnet_urgent25` @ `daad0009` (sha256 `8350b6f84bb5de01...`) | CC BY 4.0, attribution Kohei Saijo |
| ESPnet code | TF-GridNet model code (vendored subset) | github.com/kohei0209/espnet, branch urgent2025 @ `6be80495` (= ESPnet v.202409 for these files) | Apache-2.0 |
| SraVaani-1.0 | the mandated ASR; selection; ASR-feature loss | Hugging Face `ARTPARK-IISc/SraVaani-1.0` @ `f5dd5358` (access request, automatic approval) | MIT (model card) |
| DeepFilterNet3 | cleans the training speech beds | weights: github.com/Rikorose/DeepFilterNet @ `d375b2d8`, `models/DeepFilterNet3.zip`; code: PyPI `deepfilternet` 0.5.6 (vendored) | MIT OR Apache-2.0 |
| Silero VAD v5 | bed masks; speech share of noise events | github.com/snakers4/silero-vad @ `1e261b036686cd0017d500ee96acd1c4ba572a9d`, `src/silero_vad/data/silero_vad.jit` (sha256 `e1122837f4154c51...`) | MIT |
| DEMAND | public noise (10 environments; TMETRO and NPARK held out) | Zenodo record 1227121, the `*_16k.zip` files | CC BY 4.0 |
| ESC-50 | public noise (folds 1-4; fold 5 held out) | github.com/karolpiczak/ESC-50 | CC BY-NC 3.0 |
| MUSAN | public noise (noise, and music without vocals, at most 5 h) | OpenSLR SLR17, `musan.tar.gz` | CC BY 4.0 (the music files carry their own CC licences) |
| Vaani Noise-Event Dataset | speech beds and noise events (training release) | Hugging Face `ARTPARK-IISc/Vaani-Noise-Event-Dataset` @ `be488e2a` (access request) | CC BY 4.0 |
| Organizer validation set | checkpoint selection, val-newlib | the competition page | competition terms |

Our fine-tuned checkpoints (sha256 in `checkpoints/README.md`) are derivatives of the ZipEnhancer weights
(Apache-2.0) and of the TF-GridNet weights (CC BY 4.0, Kohei Saijo). The training data includes ESC-50 clips
(CC BY-NC 3.0, non-commercial), so use our checkpoints for non-commercial purposes only.

## 8. Hardware notes

- Our numbers come from an Apple M-series Mac with the PyTorch MPS backend (inference and training).
- On CUDA the outputs differ slightly: on the 695 synthetic validation clips, the mean SI-SDR is 0.04 dB lower
  than on MPS.
- T4 takes about 13 s per test clip on the Mac (4 ZipEnhancer passes), plus the ASR on the CPU.
- Two runs on the same Mac can differ by at most 1 sample step (about -100 dB), because the MPS float order can
  change.
- `main.py` starts a fresh process for each chunk of clips, because MPS memory grows with variable-length inputs.
  On MPS it sets `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.6` and `PYTORCH_MPS_LOW_WATERMARK_RATIO=0.35` (unless they
  are set already).
- Fine-tuning needs a GPU (MPS or CUDA). `T2_MICRO=1` keeps the memory low. On a Mac, use `T2_WORKERS=0`.

## 9. Limitations

- **Release form.** The scripts here are the code that made our models and outputs. In this release, local paths
  and folder names are arguments and environment variables, and the comments are rewritten. The algorithms and
  constants are unchanged. The scripts have not been run again in this exact form.
- **Not bit-reproducible training.** The mixture generator seeds each data worker with the run seed + worker id ×
  7919 + the current time. A new run makes similar, but not identical, checkpoints.
- **Unused options.** `inference/enhance.py` keeps options that T4 and E3 do not use: FRCRN and MossFormerGAN
  enhancers, two other route modes, a residual post-filter, an input level change and a raw-output copy. They are
  off by default. FRCRN and MossFormerGAN need the ClearerVoice-Studio code, which this release does not include.
  The training scripts also keep options that our runs did not use (for example `T2_TARGET_G`, `T2_W_EVENT`,
  `T2_HOLDOUT`).
- **Model fallbacks.** In the fusion path, if the ZipEnhancer pass fails on a clip, the TF-GridNet output alone is
  used; if the TF-GridNet pass fails, the ZipEnhancer output is used.
- **Fusion head inputs.** The fusion head is trained on single-view ZipEnhancer outputs and on the clips of the
  fusion path of the route rule. T4 applies the same head to the 4-view average and to every clip.
- **MUSAN speech filter.** Our public noise bank drops 5 of 930 MUSAN noise files whose Silero VAD speech share is
  at least 0.2. `build_public_noise.py` reads the shares from `PUBLIC/musan_noise_vad.json` (a JSON object: file
  name -> speech share). The step that computes these shares is not part of this release. Without the file, the 5
  files stay in the bank.
- **Vaani manifest.** The local Vaani copy (`manifest.jsonl` + audio) comes from the Vaani preparation step of the
  Track 1 folder. This folder does not repeat that step.
- **Default paths.** The scripts keep generic defaults for `T2_INPUT`, `T2_OUT` and `T2_TMP` (`/kaggle/input`,
  `/kaggle/working`, `/tmp/t2`). Set the variables as shown above.
- **Long clips.** The out-of-memory fallback uses smaller windows, so the outputs of long clips can differ a little
  between machines. The transcripts always come from the WAVs that are written.
