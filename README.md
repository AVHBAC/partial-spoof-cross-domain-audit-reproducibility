# Reproducibility: "How Trustworthy Are Partial Spoof Detectors? A Cross-Domain Operational Audit"

**Paper:** accepted at IJCB 2026, Special Session 1: Trustworthy and Secure AI for Behavioural and Biometric Recognition.

**Author and maintainer of this repository:** Yash Sukhdeve, Department of Electrical and
Computer Engineering, Clarkson University.

Analysis code, a verification suite, and the derived result tables for a
cross-domain operational audit of three partial-spoof detectors (MRM, BAM, CFPRF)
on PartialSpoof, LlamaPartialSpoof, PartialEdit, and HQ-MPSD.

The large binaries are hosted separately on the Hugging Face Hub:

- **Score outputs** (~2.4 GB, required to reproduce the numbers):
  https://huggingface.co/datasets/sukhdeveyash/partial-spoof-cross-domain-audit-data
- **Trained MRM checkpoint** (4 GB):
  https://huggingface.co/sukhdeveyash/partial-spoof-mrm-audit

No GPU is needed to reproduce the reported numbers.

## 1. Get the score data (required)
```bash
pip install huggingface_hub
hf download sukhdeveyash/partial-spoof-cross-domain-audit-data --repo-type dataset \
    --local-dir data --exclude ".gitattributes"
```
This fills `data/raw_e1_baseline/` and `data/raw_e5_cross_dataset/` with the raw
per-utterance and per-frame score `.npy`.
(The `--exclude` keeps the Hub's own `.gitattributes` out of `data/`; without it git
reports the tracked `.npy` files there as modified, although their contents are unchanged.)

## 2. Reproduce the paper numbers (no GPU)
```bash
export PYTHONPATH="$PWD/lib"
python code/analysis/c10_llamaps_partial_only.py          # LlamaPS partial: EER, fusion, Jaccard
python code/analysis/c11_llamaps_partial_frame_metrics.py # LlamaPS partial: Seg-EER + IoU
python code/analysis/c6_hqmpsd_composition.py             # HQ-MPSD partial: EER + Jaccard
python code/analysis/c12_partial_discordance.py           # partial-only discordance (96/53/81)
python code/analysis/c4b_ensemble_far.py                  # ensemble FAR=1% operating points
python code/analysis/c4_far_operating_points.py           # per-detector FAR=1% operating points
```
Each script prints a "Reproduction guards passed" line confirming the regenerated
values match the paper. The expected outputs are also committed under `data/` as CSVs.

The remaining scripts in `code/analysis/` (c1, c2, c5, c7, c8, c9, f2, f3,
`compute_cross_domain_localization_iou.py`) run the same way; `FILES_REGISTRY.md` maps
each one to the table or figure it produces. Figures are written to `code/figures/`.

## 3. Verify (integrity: recompute EER from raw .npy)
```bash
cd verification && python run_all.py
```
The verifiers recompute EER from the raw `.npy` and check the raw -> JSON -> CSV
chain. (`verify_metadata.py` ships too but is excluded from the default run; it
checks raw-audio file counts and needs the external datasets.)

## Layout
```
code/analysis/     analysis + figure scripts (c1..c12, f2/f3, compute_*)
code/inference/    run_e1_baseline.py, run_e5_cross_dataset.py (how the scores were produced)
code/mrm_training/ MRM training recipe
lib/xps_forensic/  utilities imported by some scripts
data/              category maps + derived result CSVs + nine small per-utterance
                   operational-IoU arrays (raw score .npy fetched in step 1)
verification/      run_all.py + verifiers
checkpoints/mrm/   model card (the 4 GB 55.pth is on Hugging Face, link above)
CHECKPOINTS.md     provenance, SHA256, and sources for all detectors + front-ends
```

## Checkpoints and datasets
- **MRM** (ours): https://huggingface.co/sukhdeveyash/partial-spoof-mrm-audit (SHA256 in `CHECKPOINTS.md`).
- **BAM / CFPRF**: from their authors' releases (BAM has no license; CFPRF's authors host it). See `CHECKPOINTS.md`.
- **Datasets** (PartialSpoof / LlamaPS / PartialEdit / HQ-MPSD English subset): external; sources in `CHECKPOINTS.md`.

## Environment
`ENVIRONMENT.txt` (Python 3.14, numpy 2.3.5, scipy 1.16.3, scikit-learn 1.7.2,
pandas 2.3.3, matplotlib 3.11.2). Install with `pip install -r ENVIRONMENT.txt`.
Detector inference uses each detector repo's own requirements (torch, fairseq, etc.).

## Notes
- MRM was trained with the upstream default seed (1234). Retraining is still not
  guaranteed to be bit-identical, so the trained checkpoint is hosted. See `TRAINING.md`.
- "FAR=1%" in the scripts and CSVs means a 1% false-alarm rate on PartialSpoof bona fide
  speech: the threshold is the 99th percentile of in-domain bona fide scores, which is a
  BPCER of about 1% in-domain. It is not a false-acceptance rate.
- The fixed thresholds (EER point and FAR=1%) are computed on the PartialSpoof
  evaluation set; no separate development set was held out, so in-domain figures at
  those thresholds are optimistic.
- CFPRF results use its frame-detection stage (`1FDN_PS.pth`) only. See `CHECKPOINTS.md`.
- Cite the paper and the three detectors per their licenses (BAM has none; contact its
  authors for redistribution).

## License
The code and derived tables in this repository are released under the MIT License (see
`LICENSE`). The MRM checkpoint on Hugging Face is MIT as well. BAM and CFPRF checkpoints and
the evaluation datasets keep their own licenses.
