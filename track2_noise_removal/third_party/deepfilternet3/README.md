# DeepFilterNet3 (vendored code)

DeepFilterNet3 (H. Schroeter et al., Interspeech 2023) is a 48 kHz speech enhancement model with 2,135,484
parameters. Track 2 uses it only to prepare training data: `training/data/build_clean_beds.py` cleans the speech
beds with it (full suppression). The inference pipeline does not use it.

This folder runs the model without the `deepfilternet` and `DeepFilterLib` packages. It needs `torch`, `numpy` and
`soxr` (or `scipy`).

## Files

| Path | Content | Origin |
|---|---|---|
| `infer.py` | `DeepFilterNet3(device).enhance(x)`: 16 kHz in, 16 kHz out (48 kHz inside, soxr resampling) | new code |
| `code/df/*.py` | model files of PyPI `deepfilternet` 0.5.6, unchanged; only `df/__init__.py` is reduced (marked `[IndoML-T2 change]`) | DeepFilterNet |
| `code/libdf_np.py` | NumPy translation of the DeepFilterLib 0.5.6 functions that inference uses (STFT with Vorbis window, ERB features, normalisations) | new code, translated from the DeepFilterLib Rust source |
| `code/shims/loguru/__init__.py` | small logging stand-in, used only when `loguru` is not installed | new code |
| `code/licenses/` | the DeepFilterNet licence texts (MIT, Apache-2.0, dual-licence note) | DeepFilterNet |

Code source: PyPI `deepfilternet` 0.5.6 and `DeepFilterLib` 0.5.6 (https://github.com/Rikorose/DeepFilterNet).
The docstring of `libdf_np.py` names a file `parity_df.json`; this release does not include it.

## Weights (not included)

Download `models/DeepFilterNet3.zip` from https://github.com/Rikorose/DeepFilterNet at commit
`d375b2d8309e0935d165700c91da9de862a99c31` (7,986,207 bytes, sha256
`49c52edc8947ae1f9bf50d81530beaf3a2c3245aeaf34b6f31ff535cd22284d2`) and unzip it into `weights/`:

```
curl -L -o DeepFilterNet3.zip https://raw.githubusercontent.com/Rikorose/DeepFilterNet/d375b2d8309e0935d165700c91da9de862a99c31/models/DeepFilterNet3.zip
mkdir -p third_party/deepfilternet3/weights
unzip DeepFilterNet3.zip -d third_party/deepfilternet3/weights/
shasum -a 256 third_party/deepfilternet3/weights/DeepFilterNet3/checkpoints/model_120.ckpt.best
```

Then these files exist:

- `weights/DeepFilterNet3/config.ini`
- `weights/DeepFilterNet3/checkpoints/model_120.ckpt.best` (8,714,073 bytes, sha256
  `23b92884f63ccf54bb026014604625ab231657b6480df65db4095c4c171e6003`)

`infer.py` also accepts the folder in `$DFN3_DIR` or as the `weights` argument.

## Licence

MIT OR Apache-2.0 (code and weights). Keep `code/licenses/` with any copy.
