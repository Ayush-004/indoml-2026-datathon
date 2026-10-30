# ZipEnhancer (vendored code)

ZipEnhancer is a 16 kHz single-channel speech enhancement model from Tongyi Lab, Alibaba. It is the ModelScope
model `iic/speech_zipenhancer_ans_multiloss_16k_base` and has 2,044,436 parameters.

- Paper: H. Wang and B. Tian, "ZipEnhancer: Dual-Path Down-Up Sampling-based Zipformer for Monaural Speech
  Enhancement", ICASSP 2025, doi:10.1109/ICASSP49660.2025.10888703.
- The model card says that the model is trained on DNS Challenge 2020 data (English, 16 kHz).

This folder runs the model with plain PyTorch. It does not need the ModelScope library.

## Files

| Path | Content | Origin |
|---|---|---|
| `zip_infer.py` | `ZipEnh(device)(x)`: inference class. `forward_train(model, x_batch)`: differentiable forward for fine-tuning. | new code (Apache-2.0) |
| `configuration.json` | official model configuration (unchanged, sha256 `877861f930f597b4...`) | ModelScope model repository |
| `code/zipenhancer_ms/zipenhancer_layers/*.py` | model layers (byte-identical) | ModelScope code |
| `code/zipenhancer_ms/zipenhancer.py` | model file. ModelScope imports and the registry decorator are removed; the changes are marked `[IndoML-T2 change]`. | ModelScope code, changed |
| `code/zipenhancer_ms/pipeline.py` | official pre- and post-processing as plain functions (from `ans_pipeline.py` and `audio_utils.py`) | ModelScope code, changed |
| `code/zipenhancer_ms/__init__.py` | package init | new code |
| `code/LICENSE` | Apache License 2.0 text of the ModelScope repository | ModelScope code |

Code source: https://github.com/modelscope/modelscope, commit `6cb35d56b5e45cf30154d85b56db3c67edb032ac`.
The headers of the changed files say what changed (Apache-2.0, section 4(b)). The file headers refer to a folder
`upstream_unmodified/` with the unchanged upstream files; this release does not include that folder (get the files
from the commit above).

## Weights (not included)

Download `pytorch_model.bin` (8,424,575 bytes, sha256
`b18896915e27a821585584221d0c0820f35e12145315ae3f1e73ccd5a68d195f`) from the official ModelScope repository and
save it in this folder as `zipenhancer_pytorch_model.bin`:

```
git lfs install
git clone https://www.modelscope.cn/iic/speech_zipenhancer_ans_multiloss_16k_base.git zipenhancer_ms_repo
cp zipenhancer_ms_repo/pytorch_model.bin third_party/zipenhancer/zipenhancer_pytorch_model.bin
shasum -a 256 third_party/zipenhancer/zipenhancer_pytorch_model.bin
```

The weight file comes from commit `bed374bb53e4a5ffc62093676bab408a56b0db48` of that repository. Licence of the
weights: Apache-2.0 (ModelScope model page and model card).

The Track 2 code loads our fine-tuned weights (`checkpoints/zipenhancer_g3asr/final.pt`) into this model. It still
needs the official file in this folder, because the inference and training scripts look for it by name.

## What `ZipEnh` does

The official ModelScope pipeline (`ANSZipEnhancerPipeline`) does these steps:

1. Read the WAV as float64 in [-1, 1].
2. `audio_norm`: scale the clip to -25 dBFS RMS (two steps), then convert to float32.
3. Length rule: clips shorter than 2 s get zeros up to 2 s; clips of 2-6 s take one pass; longer clips run in
   2-s windows with a 1.5-s hop.
4. Per pass: unit-RMS scaling, STFT (n_fft 400, hop 100, Hann 400), magnitude ** 0.3, network, inverse steps.
5. Output: int16 at the `audio_norm` level.

`ZipEnh` does the same steps with these differences. Each one has a switch.

| Item | Official pipeline | `ZipEnh` default | Switch |
|---|---|---|---|
| Output length, 2-6 s clips | 100 x floor(T / 100) samples | T samples | `fix_length=False` |
| Output tail, clips > 6 s | the last samples can be zero | the last window keeps its right edge | `fix_length=False` |
| Output level | `audio_norm` level | input level | `level='official'` |
| int16 | truncation | float output; the caller rounds and clips | - |
| Memory | full attention and convolution buffers | split into pieces (same numbers up to float rounding) | `attn_budget_mb=None, conv_budget_mb=None` |
| All-zero input | NaN | zeros | - |

`mode='whole'` runs one pass over the whole clip for any length. Track 2 uses `mode='whole'`.
With `fix_length=False`, `level='official'` and no memory budgets, `ZipEnh` gives int16 output that is bit-equal to the
unchanged ModelScope pipeline on our check excerpts (the check script is not part of this release).

`forward_train(model, x_batch)` takes a (B, T) float tensor and returns (B, T). It follows the same path without
`audio_norm` and without int16 rounding. Keep the model in `eval()` mode: the upstream
`ActivationDropoutAndLinear.forward` returns None in train mode, so `forward_train` stops with an error there.

## Use

```python
import sys
sys.path.insert(0, 'third_party/zipenhancer')
from zip_infer import ZipEnh
enh = ZipEnh('cpu', mode='whole')    # or 'cuda' / 'mps'; weights=<path> loads another state_dict (strict)
y = enh(x)                           # x: float numpy (T,) in [-1, 1], 16 kHz mono; y: float64 (T,), input level
```

`ZipEnh` finds the weights in this order: the `weights` argument, `$ZIPENH_WEIGHTS`, this folder (or `weights/`),
up to three parent folders, then below `/kaggle/input`. It warns when the file is not the official one (expected for
fine-tuned weights).

## Licence

Apache-2.0 for the weights and the code. `zipformer.py` and `scaling.py` also carry the Xiaomi Corp. (icefall)
Apache-2.0 header. Keep `code/LICENSE` with any copy.
