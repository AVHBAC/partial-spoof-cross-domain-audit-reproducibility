# Model Checkpoints: Provenance, Hashes, Licenses

Three detectors are audited. **Only the MRM checkpoint is our own artifact and is
bundled in this package** (`checkpoints/mrm/55.pth`). BAM and CFPRF are the
**original authors' released** checkpoints; we do **not** re-host them (BAM has no
license; CFPRF's authors host their own). Download them from the sources below and
verify the SHA256 to obtain the exact weights we used.

> Integrity note: MRM is **not** the original Zhang et al. model. It is the open
> reimplementation `MultiResoModel-Simple` (Hieu-Thi Luong), which we **trained
> ourselves** on PartialSpoof (see `TRAINING.md`). The paper attributes it
> accordingly.

## Detector checkpoints

| Detector | File (our path) | Size | SHA256 | Source repo @ commit | License | In this package? |
|---|---|---|---|---|---|---|
| **MRM** (ours, locally trained) | `MRM/checkpoints/exp/baseline/55.pth` | 4.0 GB | `5b753752f7c25370c6abf973f69f58e100dad4b5d3ea035872335358a876fdd1` | reimpl: `github.com/hieuthi/MultiResoModel-Simple` @ `0f69db3a2d654de47822d951fe6ad256bbaac9ba` | MIT | **YES** → `checkpoints/mrm/55.pth` |
| **BAM** | `BAM/bam_checkpoint/model.ckpt` | 1.35 GB | `5c7379d23028606d3beeb8b1aa5c340395fcc403ce70a2ca559ea322ab2994d1` | `github.com/media-sec-lab/BAM` @ `55f3fb9e3b4dd6281597b86d7712fb23454179f6` | **none (all rights reserved)** | NO, download below |
| **CFPRF** (FDN stage) | `CFPRF/checkpoints/1FDN_PS.pth` | 1.28 GB | `5fcbbc725761f99f7ca22a6bd095242b7d4fcbb2b285a766047941766d496267` | `github.com/ItzJuny/CFPRF` @ `358a901ead8a7d84dac979c3d626e34ef82c2854` | MIT | NO, download below |

> CFPRF note: only the FDN (frame-detection) checkpoint for PartialSpoof
> (`1FDN_PS.pth`) was available/used; there is no `2PRN_PS` proposal-refinement
> checkpoint in the repo for PartialSpoof.

## SSL front-ends (downloaded, not re-hosted)

| Front-end | Used by | File | Size | SHA256 | Download |
|---|---|---|---|---|---|
| wav2vec 2.0 Large (LV-60+Fisher+SWBD+CV), fixed | MRM | `w2v_large_lv_fsh_swbd_cv_fixed.pt` | 3.8 GB | `d74f71fe2b664688c1b1ae03c8231eea0ecbfcd1ff77d0d92ee9d4e276a2b270` | `https://dl.fbaipublicfiles.com/fairseq/wav2vec/w2v_large_lv_fsh_swbd_cv.pt` then run `MRM/fix_ssl.py` |
| WavLM Large | BAM | `wavlm_large.pt` | 1.26 GB | `6fb4b3c3e6aa567f0a997b30855859cb81528ee8078802af439f7b2da0bf100f` | Microsoft WavLM release (see BAM repo README) |
| XLS-R 300M | CFPRF | (fetched by CFPRF code) | - | - | fairseq/HuggingFace `facebook/wav2vec2-xls-r-300m` (see CFPRF repo) |

## Download instructions

- **BAM checkpoint** (`model.ckpt`): Google Drive link in the BAM README:
  `https://drive.google.com/file/d/1eL3Ca27hEruI20lkoqkQEnZlb2GzTyHT/view`.
  Place at `BAM/bam_checkpoint/model.ckpt`; verify SHA256 above.
- **CFPRF checkpoint** (`1FDN_PS.pth`): from the CFPRF repo at the commit above.
- **MRM**: bundled here. Luong also hosts *different-run* checkpoints at
  `huggingface.co/hieuthi/MultiResoModel-Simple-ckpts` (their `baseline-ps-e55`
  reports 1.48% utt-EER / 13.67% frame-EER; **ours** reports 0.94% / 13.91%, a
  separate training run, see `TRAINING.md`).

## Anonymized hosting (double-blind review)
The bundled `55.pth` is also uploaded to an **anonymous Hugging Face model repo**
so reviewers can fetch it without the 4 GB package:

> Anonymized URL: **`<ANON-HF-URL, set after running checkpoints/mrm/upload.py>`**

The checkpoint was scanned and contains no identifying paths/usernames. Upload
helper: `checkpoints/mrm/upload.py`; model card: `checkpoints/mrm/HF_MODEL_CARD.md`.
Verify SHA256 `5b753752f7c25370c6abf973f69f58e100dad4b5d3ea035872335358a876fdd1`
after download. Only the MRM checkpoint is hosted by us; BAM/CFPRF stay at their
authors' sources above. Replace with an identified link at camera-ready.

## Evaluation datasets (external, not bundled)

| Dataset | Use | Source |
|---|---|---|
| PartialSpoof | in-domain train/threshold/eval | Zhang et al. (Zenodo) |
| LlamaPartialSpoof | cross-domain (LLM-TTS) | Luong et al. (ICASSP 2025) |
| PartialEdit | cross-domain (neural editing), E1+E2 subsets | Zhang You et al. (Interspeech 2025) |
| HQ-MPSD | cross-domain (TTS), **English subset only** | Li et al. (arXiv 2512.13012) |
