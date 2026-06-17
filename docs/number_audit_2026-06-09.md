# Paper-v2 Number Audit & Reproducibility Report

**Date:** 2026-06-09
**Auditor pass:** thorough number check + reproduction from raw `.npy`, per project rules
R-01…R-08, R5 (reproducibility), R6 (citations), and the no-fabrication red line.
**Scope:** every number printed in `paper-v2` prose + tables (Abstract is not yet
written, so no abstract numbers audited).
**Verdict:** **Every headline number in the paper reproduces from raw data.** No
fabricated or untraceable result was found. One apparent discrepancy (McNemar
96/53/81 vs. a memory note of 96/34/73) was investigated and resolved in the
paper's favour (the paper uses the correct partial-only values; the note was
stale full-set values). Shortcomings below are about **process/traceability**, not
wrong numbers, except where explicitly marked.

---

## 1. What was run (commands)

| Step | Command | Result |
|---|---|---|
| Suite | `deliverables/verification/run_all.py` (8 verifiers) | all 8 exit 0; EER recomputed from raw `.npy` via `roc_curve` MATCHES JSON+CSV |
| LlamaPS partial-only | `paper/scripts/c10_llamaps_partial_only.py` | guards passed; values match paper |
| LlamaPS partial frame/IoU | `paper/scripts/c11_llamaps_partial_frame_metrics.py` (CSV) | matches paper |
| HQ-MPSD partial-only | `paper/scripts/c6_hqmpsd_composition.py` (registry C6) | matches paper |
| FAR individual + ensemble | `paper/scripts/c4_far_operating_points.py`, `c4b_ensemble_far.py` | guards passed; matches paper |
| McNemar discordance | independent recompute from raw `.npy` (partial-only, in-domain EER thresholds) | matches paper |
| Data-tree integrity | `sha256sum` across both data trees | raw `.npy` symlinked + byte-identical |

Environment: Python 3.14.0, NumPy 2.3.5, SciPy 1.16.3, scikit-learn 1.7.2 (matches
registry §2.1).

---

## 2. Reproduced numbers ledger (paper value ← reproduced value ← source)

### 2.1 In-domain detection (PartialSpoof) - `run_all` / e1
| Paper | Value | Reproduced (raw `.npy`) | Source |
|---|---|---|---|
| BAM Utt-EER | 0.54% | roc_curve 0.005328 ≈ JSON 0.005354 (Δ<0.001) | `e1_baseline_partialspoof.csv` |
| CFPRF Utt-EER | 5.88% | roc_curve 0.058698 ≈ JSON 0.058796 | same |
| MRM Utt-EER | 0.94% | roc_curve 0.009379 ≈ JSON 0.009381 | same |
| Seg-EER 4.65 / 7.70 / 13.91 | - | registry §4.1, `run_all` MATCH | same |
| Seg-F1 0.796 / 0.896 / 0.818 | - | MATCH | same |

### 2.2 Cross-domain detection (partial-only) - c10 / c6 / c11
| Paper | Value | Reproduced | Source |
|---|---|---|---|
| LlamaPS Utt-EER | 12.26 / 27.65 / 24.76 | 12.26 / 27.65 / 24.76 | c10 (guards passed) |
| LlamaPS fully-fake EER (BAM 27.0 vs 12.3) | - | 27.04 / 12.26 | c10 |
| HQ-MPSD Utt-EER | 8.25 / 12.52 / 10.31 | registry C6 | c6 |
| LlamaPS Seg-EER | 24.52 / 42.97 / 43.82 | 24.52 / 42.97 / 43.82 | c11 `llamaps_partial_frame_metrics.csv` |
| PartialEdit Seg-EER | 45.15 / 32.71 / 44.07 | `run_all` verify_partialedit MATCH | `partialedit_results.csv` |

### 2.3 Ensemble / fusion / recalibration (partial-only) - c10 / c4b
| Paper | Value | Reproduced | Source |
|---|---|---|---|
| Ensemble in-domain BPCER/ACER | 0.4% / 0.6% | `run_all` verify_voting; c4b 0.33/0.57 | `voting_detection.csv` |
| Ensemble LlamaPS ACER | 34.4% | c10 34.4; c4b 34.44 | c10 / c4b |
| Recalibration BPCER/APCER/ACER | 17.7 / 9.7 / 13.7 | c10 17.7 / 9.7 / 13.7 | c10 |
| Fusion soft-vote / best / OR / AND | 13.55 / 12.26 / 46.8 / 10.2 | c10 identical | c10 |
| Ensemble FAR=1% BPCER LlamaPS / HQ | 62.0% / 25.4% | c4b 62.02 / 25.36 | c4b (guards passed) |

### 2.4 FAR individual table (Table 4) - c4
All cells reproduced (e.g. MRM LlamaPS FAR=1% BPCER 89.0, CFPRF 53.4, BAM 28.7;
HQ-MPSD 41.1 / 26.0 / 18.7). Guards passed.

### 2.5 Correlated failures (Jaccard, partial-only) - c10 / c6
| Paper | Value | Reproduced | Source |
|---|---|---|---|
| LlamaPS CFPRF/MRM | 65.5% | c10 0.6545 | c10 |
| LlamaPS BAM/CFPRF, BAM/MRM | 13.1 / 14.2% | c10 0.1307 / 0.1418 | c10 |
| HQ-MPSD CFPRF/MRM | 45.5% | registry C6 0.455 | c6 |
| PartialSpoof (in-domain) 5.2 / 17.5 / 8.2% | - | registry §4.6 | `error_correlation.csv` |

### 2.6 McNemar discordance (BAM correct on BAM/CFPRF disagreements)
| Dataset | Paper | Reproduced (independent, partial-only) |
|---|---|---|
| PartialSpoof | 96% | b=3960 c=154 → 96.3% |
| LlamaPS | 53% | b=6094 c=5435 → 52.9% (N=107,155) |
| HQ-MPSD | 81% | b=2146 c=491 → 81.4% (N=15,485) |

### 2.7 IoU table (Table 5)
| Condition / dataset | Paper | Reproduced | Source |
|---|---|---|---|
| Operational PartialSpoof | 0.83 / 0.64 / 0.74 | 0.828 / 0.640 / 0.743 | `iou_metrics.csv` |
| Operational PartialEdit | 0.22 / 0.27 / 0.21 | 0.2176 / 0.2714 / 0.2142 | `iou_metrics.csv` |
| Operational LlamaPS | 0.53 / 0.38 / 0.47 | c11 0.53 / 0.38 / 0.47 | c11 |
| Loc-conditional LlamaPS | 0.49 / 0.53 / 0.53 | c11 0.49 / 0.53 / 0.53 | c11 |
| Loc-conditional PartialSpoof / PartialEdit | 0.59/0.78/0.67, 0.19/0.23/0.20 | registry §4.5 | `cross_domain_localization_iou.csv` |

---

## 3. Shortcomings (prioritised) - NONE are wrong numbers unless marked

**S1 (medium) - `run_all.py` PASS/FAIL is misleading.** It reports PASS = "script
exited 0", but the verifier scripts print `ERRORS: N` without `sys.exit(1)`. The run
showed `ERRORS: 4` (e5) and `ERRORS: 3` (partialedit) yet "ALL PASSED". Both error
sets are the two already-documented, **zero-numerical-impact** items: MRM drops 9
utterances on LlamaPS (paper uses 140,607) and PartialEdit E1/E2 ID collision
(44,165 unique of 88,330; arrays row-aligned, count 88,330 correct). *Fix:* make
verifiers exit non-zero on unexpected errors, or explicitly whitelist these two.

**S2 (medium) - the paper's actual numbers are not in the one-command suite.** The
headline cross-domain values are **partial-only** and come from `c6/c10/c11/c4/c4b`
+ the McNemar recompute, none of which are in `run_all.py`. Their CSVs live in
`partial-deepfake-analysis/data/`, not the `deliverables/data/` the suite checks.
They DO carry internal reproduction guards and all reproduced here, but R5
("one-command verification") does not currently cover the numbers actually printed.
*Fix:* add `c6/c10/c11/c4/c4b/c5` to `run_all.py` and write a partial-only verifier.

**S3 (medium) - registry staleness (violates R-06).** `results_registry.md` still
headlines **full-set** values the paper no longer uses: §4.6 Jaccard LlamaPS
CFPRF/MRM = 47.4% (paper uses partial-only 65.5%); McNemar BAM-correct 96/34/73
(paper uses partial-only 96/53/81). The paper values are correct and reproduce; the
registry was not updated when the paper moved to partial-only. *Fix:* update the
registry partial-only rows and mark the full-set ones superseded.

**S4 (low-medium) - datasets.tex LlamaPS composition off by 9.** The sentence gives
96,582 partial / 33,461 fully / 10,573 bonafide (sums to **140,616**, the release)
but says "we score every detector on the common subset of **140,607**". The scored
partial/fully counts are 96,576 / 33,458 (c10). *Fix:* either use 96,576 / 33,458,
or reword so the 140,616 breakdown is clearly the release and 140,607 the scored set.

**S5 (low) - EER estimator unspecified + two near-equal prints.** Methods does not
state the EER estimator. Nearest-point (`roc_curve` in verifiers) vs. interpolated
`brentq` (c10/c8) differ at 0.01–0.02 pp. Table 1 prints CFPRF in-domain **5.88**
(Utt-EER) while Table 4 prints **5.86** (BPCER at the EER threshold) - different
quantities that look like an inconsistency. *Fix:* state the EER computation method;
optionally add a footnote that Table 4's "EER" column is BPCER at the EER threshold.

**S6 (low) - registry checksum table empty.** `results_registry.md` §2.5 "Data
Checksums" was never populated. *Fix:* run `sha256sum` on the source CSVs and record.

**S7 (OPEN, Category-B - could NOT verify locally).** Three external comparison
numbers in `models.tex` require the primary sources (R6): CFPRF Seg-EER **7.61%**
(Luong et al. 2025), MRM Utt-EER **0.49%** (Zhang et al. 2023), BAM 160 ms Seg-EER
**3.58%** (Zhong et al. 2024). These are not in our data and were not verified in
this pass. *Action:* fetch each paper and confirm the exact value, or soften.

**S8 (low - not independently traced).** `metrics.tex` "4 and 28 fully synthesised
utterances" (PartialSpoof / PartialEdit, under 0.1%) was not traced to a CSV in this
audit. Low impact (both < 0.1%), but it is a printed number without a located source.

**S9 (low) - derived CSV duplication.** Raw `.npy` is single-sourced (symlinked,
verified identical), but derived CSVs are copied across the two data dirs and the
two FAR scripts write to different dirs (`c4`→deliverables, `c4b`→partial-deepfake).
Risk of future divergence. *Fix:* pick one canonical `data/` for derived CSVs.

---

## 4. Bottom line

- **No hallucinated or unsupported number** entered the paper from what I can trace.
- **All headline numbers reproduce from raw `.npy`.**
- The apparent McNemar discrepancy resolved as **paper-correct** (partial-only).
- The real gaps are **traceability/process** (S1–S3, S6, S9), one **wording
  inconsistency** (S4), one **documentation gap** (S5), and the **three Category-B
  external numbers** (S7) that still need primary-source confirmation before
  submission.
