# ESPnet speech enhancement code for TF-GridNet (vendored code)

This folder runs the TF-GridNet "URGENT 2025" baseline model without the `espnet` package. It needs only `torch`,
`numpy` and `PyYAML`.

| Model | Source (Hugging Face, commit) | Parameters | STFT at 16 kHz |
|---|---|---|---|
| `tfgridnet_urgent25` | `kohei0209/tfgridnet_urgent25` @ `daad000927daa131e7692376a71c8144bcbfa6f8` (TF-GridNet v3, URGENT 2025 baseline, trained by Kohei Saijo) | 8,524,352 | n_fft 512, hop 256 |

Track 2 uses only this model. The code also supports a BSRNN model (`bsrnn_large_noncausal` in `infer.py`,
`espnet_min/bsrnn*.py`). Track 2 does not use it.

## Files

| Path | Content | Origin |
|---|---|---|
| `infer.py` | `UniEnh(name, device)(x)`: inference class; `forward_tensor(x)`: differentiable forward for fine-tuning | new code (Apache-2.0) |
| `espnet_min/` | a changed subset of the ESPnet code. Each file starts with a notice that gives its source and the changes. `VENDOR_MANIFEST.json` lists the files. | ESPnet (Apache-2.0) |
| `espnet_min/LICENSE` | ESPnet licence (Apache-2.0) | ESPnet |
| `LICENSES/ATTRIBUTION.md` | credits for the weights and the code | - |
| `LICENSES/Apache-2.0_ESPnet.txt`, `LICENSES/CC-BY-4.0.txt` | licence texts | - |

The code comes from the ESPnet fork https://github.com/kohei0209/espnet (branch `urgent2025`, commit
`6be80495bca76d8406ca11e5971a9954949ac9c1`). For these files the fork is byte-identical to ESPnet tag `v.202409`.
The docstring of `infer.py` names a file `SOURCES.json`; this release does not include it, and this README gives its
content (sources, commits, SHA-256 values).

## Weights (not included)

Download two files unchanged from Hugging Face (public, CC BY 4.0) and put them in `tfgridnet_urgent25/` in this
folder:

| File in the Hugging Face repository | Put it at | Bytes | sha256 |
|---|---|---|---|
| `exp/enh_train_enh_tfgridnet_dm_raw/config.yaml` | `tfgridnet_urgent25/config.yaml` | 6,426 | `9591c84a91349834fb7a31a021fa4d0858b46c001c6aa7ab096f0ee502684a61` |
| `exp/enh_train_enh_tfgridnet_dm_raw/valid.loss.ave_5best.pth` | `tfgridnet_urgent25/valid.loss.ave_5best.pth` | 34,192,512 | `8350b6f84bb5de01646b7cebe9d19d5b1fc4318cd85c31d242caccc2442d276e` |

```
B=https://huggingface.co/kohei0209/tfgridnet_urgent25/resolve/daad000927daa131e7692376a71c8144bcbfa6f8/exp/enh_train_enh_tfgridnet_dm_raw
mkdir -p third_party/espnet_universal/tfgridnet_urgent25
curl -L -o third_party/espnet_universal/tfgridnet_urgent25/config.yaml "$B/config.yaml"
curl -L -o third_party/espnet_universal/tfgridnet_urgent25/valid.loss.ave_5best.pth "$B/valid.loss.ave_5best.pth"
shasum -a 256 third_party/espnet_universal/tfgridnet_urgent25/*
```

`UniEnh` finds the two files in this order: the `weights` / `config` arguments, `$UNIENH_TFGRIDNET_URGENT25_WEIGHTS`
and `$UNIENH_TFGRIDNET_URGENT25_CONFIG`, `tfgridnet_urgent25/` next to `infer.py`, any folder below it, up to three
parent folders, then below `/kaggle/input`.

The Track 2 code loads our fine-tuned TF-GridNet weights (`checkpoints/tfgridnet_v8rm/final.pt`, a state_dict of
`UniEnh('tfgridnet_urgent25').model`) over these base weights.

## What `UniEnh` does

1. The official ESPnet path (`espnet2.bin.enh_inference.SeparateSpeech`, whole clip): x / std(x), STFT, separator,
   inverse STFT with the input length, times std(x).
2. One scalar gain per clip (`output_level`). The raw output level of the model is arbitrary, because it is trained
   with scale-invariant losses. Default `'input'`: least-squares gain onto the noisy input. Other values: `'peak'`
   (official `normalize_output_wav=True`), `'raw'`, `'rms'`. The gain does not change SI-SDR.

On our check clips the raw output is equal to the output of ESPnet's own `SeparateSpeech` (max abs difference
0.0). Track 2 calls `UniEnh('tfgridnet_urgent25', device, lstm_chunk=128)`. `lstm_chunk` runs each LSTM on at most
N sequences at a time: same math, less memory.

## Licences

- Weights and configs: CC BY 4.0 (`LICENSES/CC-BY-4.0.txt`), attribution Kohei Saijo; see `LICENSES/ATTRIBUTION.md`.
  Our fine-tuned TF-GridNet weights are a derivative of these weights.
- Code in `espnet_min/`: Apache-2.0 (ESPnet, Copyright 2017 Johns Hopkins University (Shinji Watanabe) and the
  ESPnet contributors). Each file lists its changes.
