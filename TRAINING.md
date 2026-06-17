# MRM Training Recipe (the one locally-trained model)

BAM and CFPRF use the authors' released checkpoints (see `CHECKPOINTS.md`).
**MRM was trained by us** on PartialSpoof, so its checkpoint
(`checkpoints/mrm/55.pth`) is included in this package and documented here.

## Code
- Reimplementation: `MultiResoModel-Simple`, `github.com/hieuthi/MultiResoModel-Simple`
  @ commit `0f69db3a2d654de47822d951fe6ad256bbaac9ba` (MIT).
- Training script: `code/mrm_training/train_baseline.sh`
- Config: `code/mrm_training/baseline.toml`

## Command
```bash
# 1. obtain PartialSpoof (train/dev/eval)            ./download_ps.sh
# 2. obtain front-end: wav2vec 2.0 Large + fix_ssl   (see CHECKPOINTS.md)
python -u train.py --config configs/baseline.toml --batch_size 8 --num_workers 6
# trained to epoch 55  ->  checkpoints/exp/baseline/55.pth
```

## Key hyper-parameters (`baseline.toml`)
- Multi-resolution units: 0.02, 0.04, 0.08, 0.16, 0.32, 0.64 s; `include_utt = true`
- Front-end: wav2vec 2.0 Large (`ssl_dim = 1024`, `ssl_tuning = true`)
- Segment duration 9.6 s, `random_seek = true`, `use_mask = true`
- Optimiser: lr 1e-5; scheduler step 10, decay 0.5
- Hardware used: NVIDIA RTX 4080 (16 GB), 64 GB RAM

## Reproducibility caveat (important)
The config sets `random_seek = true` and **no fixed random seed**, so training is
**stochastic and not bit-reproducible**: re-running `train_baseline.sh` yields a
*similar* but not identical checkpoint. **This is why `55.pth` is shipped** rather
than only the recipe, the shipped weights are the authoritative artifact behind
every MRM number in the paper.

Training logs from the original run were not retained; the config + command +
commit above fully specify the recipe.

## Our result vs. the public reimplementation checkpoint
| | utt-EER (ps-eval) | seg/frame-EER (ps-eval) |
|---|---|---|
| Our `55.pth` (this package) | **0.94%** | **13.91%** |
| Luong's released `baseline-ps-e55` | 1.48% | 13.67% |
| Original Zhang et al. MultiResoModel | 0.49% | - |

Our run differs from Luong's released checkpoint (different training run) and from
the original (a different implementation), which is exactly why the paper reports
MRM as a re-trained reimplementation, not a published model.
