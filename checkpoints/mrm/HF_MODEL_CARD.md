---
license: mit
tags:
- audio
- speech
- anti-spoofing
- partial-spoof
- deepfake-detection
---

# MRM (Multi-Resolution Model): Partial Spoof Detector Checkpoint

Checkpoint accompanying the paper *"How Trustworthy Are Partial Spoof Detectors? A Cross-Domain Operational Audit"*,
accepted at IJCB 2026, Special Session 1: Trustworthy and Secure AI for Behavioural
and Biometric Recognition. Yash Sukhdeve, Ajan Ahmed, and Masudul H. Imtiaz, Department of
Electrical and Computer Engineering, Clarkson University.

Analysis code and verification suite: https://github.com/AVHBAC/partial-spoof-cross-domain-audit-reproducibility

## What this is
A multi-resolution partial-spoof detector: the model of Zhang et al. (the PartialSpoof
multi-resolution countermeasure), via the open reimplementation
**`MultiResoModel-Simple`** (Luong et al.), **trained by us** on the PartialSpoof
training set. Front-end: wav2vec 2.0 Large; back-end: losses supervised jointly at
frame (20 ms), segment, and utterance scales.

This is **not** an authors'-released checkpoint of the original model, and **not** the
public `MultiResoModel-Simple` checkpoint. It is our own training run.

Note that authors' released multi-resolution checkpoints do exist: the PartialSpoof
repository ships `03multireso/01_download_pretrained_models.sh`, which retrieves
`multi-reso.tar.gz` from Zenodo record 6674660. This checkpoint is an independent
retraining, not a substitute for those artifacts, and results obtained with it should
not be read as reproducing them.

## File
| File | SHA256 |
|---|---|
| `55.pth` | `5b753752f7c25370c6abf973f69f58e100dad4b5d3ea035872335358a876fdd1` |

## Reported performance (PartialSpoof eval, ours)
- Utterance-level EER: **0.94%**
- Segment-level (20 ms frame) EER: **13.91%**

For reference, the public reimplementation checkpoint reports ~1.48% / ~13.67%, and the
original Zhang et al. model reports 0.49% utterance-level EER. Cross-domain behaviour
(LlamaPartialSpoof, PartialEdit, HQ-MPSD) is the subject of the paper.

These thresholds and scores are derived from the PartialSpoof **evaluation** split; no
separate development split was held out. In-domain figures are therefore optimistic.

## Training recipe
- Multi-resolution units {0.02, 0.04, 0.08, 0.16, 0.32, 0.64} s; segment duration 9.6 s;
  `random_seek = true`, `use_mask = true`.
- Optimizer Adam, lr 1e-5; scheduler StepLR, step 10, gamma 0.5; loss P2SGradLoss.
- Batch size 8, 6 dataloader workers. Trained to epoch 55.
- Implementation: `github.com/hieuthi/MultiResoModel-Simple` @
  `0f69db3a2d654de47822d951fe6ad256bbaac9ba`.

### Random seeding

**A fixed seed was used.** `train.py` defaults to `--seed 1234` and calls
`reproducibility(seed)`, which seeds `torch`, `random`, `numpy`, `PYTHONHASHSEED` and
CUDA, and sets `cudnn.deterministic = True`, `cudnn.benchmark = False`. The launch
script passes no override, so the default applies. The `random_seek` crop draws from
Python's `random`, which PyTorch seeds deterministically per dataloader worker.

> Earlier revisions of this card stated that no seed was fixed. That was incorrect and
> is corrected here.

Bit-identical reproduction is still not guaranteed, for two remaining reasons:
`torch.use_deterministic_algorithms(True)` is not set, so some CUDA kernels used during
wav2vec 2.0 fine-tuning may vary run to run; and the original training log was not
retained, so the seed in force can be established from the committed script but not from
a run record. The published weights, not the recipe, are the authoritative artifact
behind every MRM number in the paper.

## Intended use
Research and reproducibility only. This checkpoint supports an audit of an existing
detector design under cross-domain partial-spoof attacks. It is not a deployable
forensic tool, and no claim is made about its reliability outside the corpora studied.

## Related artifacts
- Frame-level and utterance-level detector score outputs:
  `huggingface.co/datasets/sukhdeveyash/partial-spoof-cross-domain-audit-data`

## License & attribution
Released under MIT, following the `MultiResoModel-Simple` reimplementation (MIT). If you
use this checkpoint, please cite the original multi-resolution model (Zhang et al.,
IEEE/ACM TASLP 2023, *The PartialSpoof Database and Countermeasures for the Detection of
Short Fake Speech Segments Embedded in an Utterance*) and the reimplementation
(Luong et al., ICASSP 2025, *LlamaPartialSpoof*), together with the paper above.
