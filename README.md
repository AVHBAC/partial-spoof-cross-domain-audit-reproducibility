# Reproducibility Package
## "How Trustworthy Are Partial Spoof Detectors? A Cross-Domain Operational Audit" (IJCB 2026)

Reproduces every number in the paper from the raw detector score outputs, and
documents the exact detector checkpoints used. Built 2026-06-10; self-contained
for the analysis/reproduction path (no GPU, no datasets needed).

## Contents
```
README.md            this file
CHECKPOINTS.md       provenance, SHA256, licenses, downloads for all 3 detectors + front-ends
TRAINING.md          recipe for MRM (the one model we trained locally)
ENVIRONMENT.txt      analysis/verification Python environment
code/
  analysis/          our scripts: numbers + figures from score outputs (c1..c11, f2/f3, compute_*, ...)
  inference/         run_e1_baseline.py, run_e5_cross_dataset.py (how detector scores were produced)
  mrm_training/      baseline.toml + train_baseline.sh (MRM training recipe)
lib/xps_forensic/    minimal library (utils/metrics, utils/stats, ...) imported by c11 / compute_cross_domain_localization_iou
data/
  raw_e1_baseline/        raw per-utterance/-frame .npy + results.json (PartialSpoof in-domain)
  raw_e5_cross_dataset/   raw .npy for LlamaPS / PartialEdit / HQ-MPSD
  llamaps_utt_categories.csv, hqmpsd_utt_categories.csv   partial/full/bonafide maps
  *.csv                   24 verified result CSVs (the published numbers)
verification/        run_all.py + 7 verifiers (raw .npy -> JSON -> CSV checks)
checkpoints/mrm/     55.pth (our locally-trained MRM, 4 GB) + HF_MODEL_CARD.md + upload.py
docs/                number_audit_2026-06-09.md  (full number-reproduction audit)
```

## Reproduce the paper numbers (no GPU)
From the package root:
```bash
export PYTHONPATH="$PWD/lib"          # needed by c11 / compute_cross_domain_localization_iou
python code/analysis/c10_llamaps_partial_only.py        # LlamaPS partial: EER, fusion, recal, Jaccard
python code/analysis/c11_llamaps_partial_frame_metrics.py   # LlamaPS partial: Seg-EER + IoU
python code/analysis/c6_hqmpsd_composition.py           # HQ-MPSD partial: EER + Jaccard + composition
python code/analysis/c12_partial_discordance.py         # partial-only McNemar discordance (BAM-correct % = 96/53/81)
python code/analysis/c4b_ensemble_far.py                # ensemble FAR=1% operating points
python code/analysis/c4_far_operating_points.py         # individual FAR=1% operating points
```
Each script prints a "Reproduction guards passed" line confirming the headline
values match the paper. Outputs are written under `data/`.

## Verify (integrity: recompute EER from raw .npy)
```bash
cd verification && python run_all.py
```
The 6 numbers verifiers resolve paths relative to the package (no setup needed);
they recompute EER from the raw `.npy` and check raw -> JSON -> CSV.
(`verify_metadata.py` ships too but is excluded from the default run: it checks
raw-audio file counts and needs the external datasets.) All headline numbers were
independently reproduced from raw `.npy` (2026-06-10); see
`docs/number_audit_2026-06-09.md`.

## Detector checkpoints
- **MRM** (ours, locally trained): bundled at `checkpoints/mrm/55.pth`; also
  uploadable to an anonymous Hugging Face repo via `checkpoints/mrm/upload.py`
  (card `HF_MODEL_CARD.md`), with the anonymized link recorded in `CHECKPOINTS.md`.
- **BAM / CFPRF**: not bundled (BAM has no license; CFPRF's authors host it).
  Download + SHA256 + commit in `CHECKPOINTS.md`.
- **Datasets** (PartialSpoof / LlamaPS / PartialEdit / HQ-MPSD English subset):
  external; sources in `CHECKPOINTS.md`.

## Environment
- Analysis/verification: `ENVIRONMENT.txt` (Python 3.14, numpy 2.3.5, scipy 1.16.3,
  scikit-learn 1.7.2).
- Detector **inference** (path B) uses each detector repo's own `requirements.txt`
  (torch, fairseq, etc.).

## Notes
- MRM is not bit-reproducible from scratch (`random_seek`, no fixed seed); the
  trained checkpoint is therefore shipped. See `TRAINING.md`.
- Cite the paper + the three detectors per their licenses (BAM has none; contact
  its authors for redistribution).
