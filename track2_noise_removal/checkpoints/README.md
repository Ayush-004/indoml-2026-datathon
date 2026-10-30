# Fine-tuned checkpoints

The three fine-tuned checkpoints of Track 2 are in `track2_checkpoints.zip` of the [v1.0 release](https://github.com/Ayush-004/indoml-2026-datathon/releases/tag/v1.0).
Unzip it in `track2_noise_removal/` (`unzip track2_checkpoints.zip`). It gives this layout (or give another
folder with `main.py --ckpt-dir`):

```
checkpoints/
  zipenhancer_g3asr/final.pt     ZipEnhancer, stage 2 (G3-ASR)
  tfgridnet_v8rm/final.pt        TF-GridNet (v8r-M)
  fusion_head/head.pt            time-frequency fusion head
```

| File | sha256 | Bytes | Content |
|---|---|---|---|
| `zipenhancer_g3asr/final.pt` | `1ec3186c205f136d60725ebb30fee7f72ad5eb694cba6ec339c0eae81965d79d` | 8,363,247 | state_dict of the ZipEnhancer model (2,044,436 parameters) |
| `tfgridnet_v8rm/final.pt` | `75538ba75f1125835412990bace7281df346f06e10b27c11dbb6af62fec9262e` | 34,175,419 | state_dict of `UniEnh('tfgridnet_urgent25').model` (8,524,352 parameters) |
| `fusion_head/head.pt` | `3f67d819099bdc0da25256c2149bf18bc86c76b5442c27eb61ac40a2b8e39987` | 119,509 | `{'state_dict', 'w0', 'nfft', 'hop'}` of `training/fusion_head.py` (about 29k parameters) |

Check the files with:

```
shasum -a 256 checkpoints/*/*.pt
```

How each checkpoint is made: see "How the models are trained" in `../README.md`.

Licences of the base weights: the ZipEnhancer checkpoint is a derivative of the Apache-2.0 ModelScope weights. The
TF-GridNet checkpoint is a derivative of the CC BY 4.0 weights `kohei0209/tfgridnet_urgent25` by Kohei Saijo.
The training data includes ESC-50 clips (CC BY-NC 3.0), so use these checkpoints for non-commercial purposes only.
