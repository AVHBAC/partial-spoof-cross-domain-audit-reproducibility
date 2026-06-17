# File Registry

What each required file is, where it lives, and which part of the paper it reproduces.

## Where things live
| Channel | Contents |
|---|---|
| This repo | analysis + verification code, derived result CSVs, category maps, docs |
| HF dataset `sukhdeveyash/partial-spoof-cross-domain-audit-data` | raw per-utterance/-frame score `.npy` (download into `data/`) |
| HF model `sukhdeveyash/partial-spoof-mrm-audit` | the trained MRM checkpoint `55.pth` (4 GB) |
| External (see `CHECKPOINTS.md`) | BAM / CFPRF checkpoints, SSL front-ends, evaluation datasets |

## Required files by category
| Category | Location | In git? | Purpose |
|---|---|---|---|
| Analysis scripts `code/analysis/*.py` | repo | yes | numbers + figures from score outputs |
| Inference runners `code/inference/*.py` | repo | yes | how detector scores were produced (`run_e1`, `run_e5`) |
| MRM training recipe `code/mrm_training/` | repo | yes | `baseline.toml` + `train_baseline.sh` |
| Library `lib/xps_forensic/utils/*.py` | repo | yes | metrics/stats imported by `c11` + `compute_cross_domain` |
| Verifiers `verification/*.py` | repo | yes | `run_all.py` + 6 active verifiers (+ `verify_metadata`, needs datasets) |
| Result CSVs `data/*.csv` | repo | yes | the published numbers (Tables 1-5) |
| Category maps `data/*_utt_categories.csv` | repo | yes | partial / full / bonafide labels for partial-only scoring |
| Raw score outputs `data/raw_e{1,5}_*/*.npy` | HF dataset | no | per-utterance/-frame scores; inputs to every analysis script |
| MRM checkpoint `55.pth` (4 GB) | HF model | no | the locally-trained MRM weights; SHA256 in `CHECKPOINTS.md` |

## Script to paper element
| Script | Reproduces |
|---|---|
| `c10_llamaps_partial_only.py` | Table 1 LlamaPS EER, ensemble/recalibration/fusion, Jaccard |
| `c11_llamaps_partial_frame_metrics.py` | Table 2 LlamaPS Seg-EER, Table 5 LlamaPS IoU |
| `c6_hqmpsd_composition.py` | Table 1 HQ-MPSD EER, Jaccard, composition |
| `c4_far_operating_points.py` / `c4b_ensemble_far.py` | Table 4 (per-detector + ensemble FAR=1%) |
| `c5_mcnemar_tests.py` | McNemar significance |
| `c12_partial_discordance.py` | BAM-vs-CFPRF discordance (96 / 53 / 81) |
| `compute_cross_domain_localization_iou.py` | Table 5 localization-conditional IoU |
| `f3_error_correlation.py` | Fig. 3 (Jaccard heatmap) |
| `c2_score_distributions.py` / `f2_iou_distributions.py` | Fig. 2 / Fig. 1 |
| verification suite | recomputes EER from raw `.npy`; checks raw -> JSON -> CSV |

## Reproduce
See `README.md`: download the score data, then run the analysis scripts. Each prints
a reproduction-guard line confirming the regenerated values match the paper.
