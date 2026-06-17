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

Checkpoint for the IJCB 2026 submission
*"How Trustworthy Are Partial Spoof Detectors? A Cross-Domain Operational Audit."*

## What this is
A multi-resolution partial-spoof detector: the model of Zhang et al. (the
PartialSpoof multi-resolution countermeasure), via the open reimplementation
**`MultiResoModel-Simple`** (Luong et al.), **trained by us** on the PartialSpoof
training set. Front-end: wav2vec 2.0 Large; back-end: losses supervised jointly at
frame (20 ms), segment, and utterance scales.

This is **not** an authors'-released checkpoint of the original model and **not**
the public `MultiResoModel-Simple` checkpoint; it is our own training run.

## File
| File | SHA256 |
|---|---|
| `55.pth` | `5b753752f7c25370c6abf973f69f58e100dad4b5d3ea035872335358a876fdd1` |

## Reported performance (PartialSpoof eval, ours)
- Utterance-level EER: **0.94%**
- Segment-level (20 ms frame) EER: **13.91%**

For reference, the public reimplementation checkpoint reports ~1.48% / ~13.67%;
the original Zhang et al. model reports 0.49% utterance-level EER. Cross-domain
behaviour (LlamaPartialSpoof, PartialEdit, HQ-MPSD) is the subject of the paper.

## Training recipe
- Config: multi-resolution units {0.02, 0.04, 0.08, 0.16, 0.32, 0.64} s; segment
  duration 9.6 s; `random_seek`, `use_mask`; lr 1e-5; scheduler step 10, decay 0.5.
- Trained to epoch 55. **Not bit-reproducible** (`random_seek`, no fixed seed),
  which is why the trained weights are released directly.

## Intended use
Research / reproducibility only. Audits an existing detector design under
cross-domain partial-spoof attacks; not a deployable forensic tool.

## License & attribution
Released under MIT, following the `MultiResoModel-Simple` reimplementation (MIT).
If you use this checkpoint, cite the original multi-resolution model (Zhang et al.,
IEEE/ACM TASLP 2023, *The PartialSpoof Database and Countermeasures...*) and the
reimplementation (Luong et al., ICASSP 2025, *LlamaPartialSpoof*), plus the IJCB
2026 paper.
