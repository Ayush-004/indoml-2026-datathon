# Attribution and licences

Written in ASD-STE100 Simplified Technical English.

## 1. Model weights and configs (CC BY 4.0)

The licence text is in `CC-BY-4.0.txt` (from https://creativecommons.org/licenses/by/4.0/legalcode.txt).
This release does not include the weight files or the config files. Download them unchanged from the sources
below (the SHA-256 of the TF-GridNet weights is in `../README.md`). Only the TF-GridNet model is used in Track 2.

1. `tfgridnet_urgent25/`: "kohei0209/tfgridnet_urgent25", trained by Kohei Saijo (Hugging Face user
   kohei0209) with the urgent25 recipe of ESPnet (fork https://github.com/kohei0209/espnet, branch
   urgent2025). Pre-trained baseline of the URGENT 2025 Challenge (https://urgent-challenge.github.io/urgent2025).
   Source: https://huggingface.co/kohei0209/tfgridnet_urgent25, commit
   daad000927daa131e7692376a71c8144bcbfa6f8. Licence: CC BY 4.0.
   Files: `exp/enh_train_enh_tfgridnet_dm_raw/valid.loss.ave_5best.pth`, `exp/enh_train_enh_tfgridnet_dm_raw/config.yaml`.
2. `bsrnn_large_noncausal/`: "wyz/vctk_dns2020_whamr_bsrnn_large_noncausal", trained by wyz (Wangyou Zhang)
   with the universal_se_v1 recipe of ESPnet (configs: https://github.com/Emrys365/se-scaling).
   Source: https://huggingface.co/wyz/vctk_dns2020_whamr_bsrnn_large_noncausal, commit
   05f45f26b889822091410d9f1f2c1025f84d93bb. Licence: CC BY 4.0.
   Files: `exp_vctk_dns20_whamr/enh_train_enh_bsrnn_large_noncausal_raw/97epoch.pth`, `.../config.yaml`.

The original model cards are on the Hugging Face pages above (this release does not copy the weight files, the
config files or the model cards). They ask users to cite ESPnet (Watanabe et al.,
Interspeech 2018) and ESPnet-SE (Li et al., IEEE SLT 2021).

Model references (as given in the ESPnet source files and the se-scaling README):
- TF-GridNet: Z.-Q. Wang, S. Cornell, S. Choi, Y. Lee, B.-Y. Kim, S. Watanabe, "TF-GridNet: Integrating
  Full- and Sub-Band Modeling for Speech Separation", TASLP 2023; and "TF-GridNet: Making Time-Frequency
  Domain Models Great Again for Monaural Speaker Separation", ICASSP 2023.
- BSRNN: J. Yu, H. Chen, Y. Luo, R. Gu, C. Weng, "High fidelity speech enhancement with band-split RNN",
  Interspeech 2023; J. Yu, Y. Luo, "Efficient monaural speech enhancement with universal sample rate
  band-split RNN", ICASSP 2023.
- Scaling study of the BSRNN model: "Beyond Performance Plateaus: A Comprehensive Study on Scalability in
  Speech Enhancement", arXiv:2406.04269.

## 2. Code (Apache License 2.0)

`espnet_min/` is a modified subset of ESPnet (https://github.com/espnet/espnet), Copyright 2017 Johns
Hopkins University (Shinji Watanabe) and the ESPnet contributors, licensed under the Apache License 2.0
(`Apache-2.0_ESPnet.txt`, the same text as `espnet_min/LICENSE`). The files come from the fork
https://github.com/kohei0209/espnet (branch urgent2025, commit 6be80495bca76d8406ca11e5971a9954949ac9c1);
for these files the fork is byte-identical to ESPnet tag v.202409. Each file starts with a notice that
gives its source and the changes (Apache-2.0 section 4(b)); `espnet_min/VENDOR_MANIFEST.json` lists them.
`espnet_min/tcn_norm.py` comes from ESPnet `espnet2/enh/layers/tcn.py`, which states that it is based on
https://github.com/kaituoxu/Conv-TasNet (MIT licence).

`infer.py` (UniEnh wrapper) is new code of the IndoML 2026 Track 2 project; use it under the same
Apache-2.0 terms.
