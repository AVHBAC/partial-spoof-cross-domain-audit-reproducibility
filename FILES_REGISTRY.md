# Reproducibility File Registry

Every file required to reproduce *"How Trustworthy Are Partial Spoof Detectors?
A Cross-Domain Operational Audit"* (IJCB 2026), where it lives, and what it is for.

> **Test status:** end-to-end re-tested **2026-06-12** from a clean extract of
> `partial-spoof-audit-reproducibility-2026-06-10.zip`, all 6 analysis scripts +
> `compute_cross_domain_localization_iou` run, and all 6 numbers-verifiers PASS.

## Three distribution channels
| Channel | Contains | Notes |
|---|---|---|
| **GitHub repo** | code + CSVs + docs (everything *not* gitignored) | reproduces tables/figures from the bundled CSVs/scores' derived data; raw `.npy`/checkpoints are NOT here (`*.npy`/`*.pth`/`*.zip` are gitignored, and > GitHub's 100 MB limit) |
| **Package zip** (4.1 GB) | the GitHub content **+** raw `.npy` score outputs **+** MRM checkpoint | fully self-sufficient; SHA256 `276a5a3a...` |
| **External** | datasets, BAM/CFPRF checkpoints, SSL front-ends, MRM anon mirror | download links + SHA256 in `CHECKPOINTS.md` |

## Required files by category

| Category | Count | Location(s) | In git? | Purpose |
|---|---|---|---|---|
| Analysis scripts `code/analysis/*.py` | 15 | package, repo (`paper/scripts`) | yes | numbers + figures from score outputs |
| Inference runners `code/inference/*.py` | 2 | package, repo | yes | how detector scores were produced (`run_e1`, `run_e5`) |
| MRM training recipe `code/mrm_training/` | 2 | package, repo | yes | `baseline.toml` + `train_baseline.sh` (the locally-trained model) |
| Library `lib/xps_forensic/utils/*.py` | 5 | package, repo | yes | metrics/stats imported by `c11` + `compute_cross_domain` |
| Verifiers `verification/*.py` | 8 | package, repo | yes | `run_all.py` + 6 active verifiers (+ `verify_metadata`, needs datasets) |
| Result CSVs `data/*.csv` | 26 | package, repo | yes | the published numbers (Tables 1–5) |
| Category maps `data/*_utt_categories.csv` | 2 | package, repo | yes | partial/full/bonafide labels for partial-only scoring |
| Raw score outputs `data/raw_e{1,5}_*/*.npy` | 60 | **package only** | **no** (gitignored) | per-utterance/-frame scores; inputs to every analysis script |
| IoU distributions `data/iou_dist_*.npy` | 9 | **package only** | **no** (gitignored) | per-utterance IoU for `c11` / Fig. 1 |
| **MRM checkpoint** `checkpoints/mrm/55.pth` | 1 (4 GB) | **package + anon HF** | **no** (gitignored) | our locally-trained MRM weights; SHA256 `5b753752...` |
| Docs | 6 | package, repo | yes | `README`, `CHECKPOINTS`, `TRAINING`, `HF_MODEL_CARD`, `FILES_REGISTRY`, `docs/number_audit` |
| Env | 2 | package, repo | yes | `ENVIRONMENT.txt`, `SHA256SUMS-bundled.txt` |

## Not bundled (external, see `CHECKPOINTS.md`)
| Item | Where | Why not bundled |
|---|---|---|
| BAM checkpoint `model.ckpt` | authors' Google Drive | no license → cannot redistribute |
| CFPRF checkpoint `1FDN_PS.pth` | `github.com/ItzJuny/CFPRF` | authors host it (MIT) |
| SSL front-ends (wav2vec2 / WavLM / XLS-R) | fairseq / Microsoft / HF | downloadable; large |
| Datasets (PartialSpoof / LlamaPS / PartialEdit / HQ-MPSD-English) | dataset sources | licensed / large |

## Script → paper element
| Script | Reproduces |
|---|---|
| `c10_llamaps_partial_only.py` | Table 1 LlamaPS EER, ensemble/recal/fusion (§5.2–5.3), Jaccard (§6) |
| `c11_llamaps_partial_frame_metrics.py` | Table 2 LlamaPS Seg-EER, Table 5 LlamaPS IoU |
| `c6_hqmpsd_composition.py` | Table 1 HQ-MPSD EER, Jaccard (§6), composition disclosure |
| `c4_far_operating_points.py` / `c4b_ensemble_far.py` | Table 4 (individual + ensemble FAR=1%) |
| `c5_mcnemar_tests.py` | §5.2 McNemar significance |
| `compute_cross_domain_localization_iou.py` | Table 5 localisation-conditional IoU |
| `c1`/`f3_error_correlation.py` | Fig. 3 (Jaccard heatmap) |
| `c2_score_distributions.py` / `f2_iou_distributions.py` | Fig. 2 / Fig. 1 |
| verification suite | recomputes EER from raw `.npy`; checks raw→JSON→CSV |

## Reproduce
See `README.md` (path A: tables/figures from scores, no GPU; path B: scores from
audio, GPU + datasets + checkpoints). All numbers independently reproduced from
raw `.npy`; full audit in `docs/number_audit_2026-06-09.md`.
