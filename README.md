# IndoML 2026 Datathon: noise event detection and noise removal on Vaani speech

**Team sonicboom** · Codabench user ayushm

This repository holds the code of team sonicboom for both tracks of the IndoML 2026 Datathon:
[Track 1, noise event detection](https://www.codabench.org/competitions/17825/) and
[Track 2, noise removal](https://www.codabench.org/competitions/17835/).
The 3-page technical report is [`report/IndoML2026_Technical_Report.pdf`](report/IndoML2026_Technical_Report.pdf).
Our fine-tuned model weights are in the [v1.0 release](https://github.com/Ayush-004/indoml-2026-datathon/releases/tag/v1.0).

## Results

| Track | System | What it is | Score | Final rank |
| --- | --- | --- | --- | --- |
| 1 Noise event detection | 950232, leaderboard entry | ATST-Frame ensemble + template matching with the noise library | 1.6112 (event F1 0.7911 + Dice 0.8201) | 4th of 82 |
| 1 Noise event detection | C8, model-only | 4 ATST-Frame members × 4 time shifts, one vote | 1.5925 (event F1 0.7795 + Dice 0.8130) | — |
| 2 Noise removal | E3, leaderboard entry | ZipEnhancer + TF-GridNet + fusion head, with a route rule and a pass-through step | 15.31 (SI-SDR 13.20 dB + ΔWER 2.11) | 10th of 20 |
| 2 Noise removal | T4, model-only | 4-view ZipEnhancer + TF-GridNet + fusion head | Validation: 15.62 dB SI-SDR on VH695 | — |

A model-only system runs one learned pipeline on every clip. At test time it reads no file name, no peak level
and no noise library. Scores are from the Codabench servers; ranks are from the final public leaderboards.

- Track 1 score = event F1 (onset and offset within 20 % of the reference length, at least 50 ms) + segment Dice
  (10-ms frames). The maximum is 2.0.
- Track 2 score = SI-SDR on the synthetic clips (dB) + ΔWER, the WER drop with the required SraVaani-1.0 ASR
  (0.32 × synthetic + 0.68 × natural).

## Repository layout

```
report/                    the 3-page technical report (PDF)
track1_event_detection/    Track 1: members, training, LIBSYN, the C8 vote, the 950232 additions
track2_noise_removal/      Track 2: T4 and E3 inference, enhancer fine-tuning, fusion head
```

Each track folder has its own README. It gives the setup, the data preparation, the training and inference steps,
and a table of the pre-trained dependencies with their sources and licences.

## What is not in this repository

- **Audio, labels and predictions.** The training data is the
  [Vaani Noise-Event Dataset](https://huggingface.co/datasets/ARTPARK-IISc/Vaani-Noise-Event-Dataset) on Hugging
  Face. The validation and test releases come from the competition pages.
- **The noise library.** The Track 1 code builds it from the official validation release.
## Model weights

The [v1.0 release](https://github.com/Ayush-004/indoml-2026-datathon/releases/tag/v1.0) holds our fine-tuned weights; its `SHA256SUMS.txt` lists their sha256 values. The
track READMEs say where each file goes. The pre-trained base models come from their official sources (see the
track READMEs).

| Track | System | Files |
| --- | --- | --- |
| 1 | C8 | `P2L3.pt`, `S1L3.pt`, `R10L2.pt`, `PSR10BL2.pt` (the 4 members); `parent-native-e5.pt` (architecture) |
| 1 | 950232 | `parent-p2.pt`, `S1-004000-student.pt`, `R10-u15500-fp32.pt`, `PSR10B-u15000-fp32.pt` (members); `R10S1-final-015500.pt`, `SOUP2.pt` (fallback donors); `e0-head.zip` (E0 head); `parent-native-e5.pt` |
| 2 | T4 and E3 | `track2_checkpoints.zip`: ZipEnhancer (stage 2), TF-GridNet v8r-M and the fusion head |

Track 1 weights build on ATST-Frame (MIT); the E0 head runs on frozen WavLM-Large (see its model card). Track 2
weights build on ZipEnhancer (Apache-2.0) and TF-GridNet `kohei0209/tfgridnet_urgent25` (CC BY 4.0, Kohei
Saijo). The Track 2 training data includes ESC-50 clips (CC BY-NC 3.0), so use the Track 2 weights for
non-commercial purposes only.

## Licence

Our code is under the MIT licence (see [`LICENSE`](LICENSE)). Third-party code and models keep their own licences,
which the track READMEs list.
