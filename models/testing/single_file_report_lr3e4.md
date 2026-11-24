# Single-file training report — `plate6_CVCL_1125.h5ad`

- cells 34262, genes 50266, drug wells 88 (train 80 / held out 8)
- dose(s) [0.05, 5.0], chem tokens `all`, dropout 0.1
- dim 512, heads 8, layers 1, conc_dim 32, lr 0.0003
- 60 steps in 9.8 min, 8.44 s/step, peak VRAM 6.58 GB (reserved 8.30 GB of 12.88)

## Is it learning?

| reference | loss |
|---|---|
| predict 0 | 23.4044 |
| predict control mean | 17.5078 |
| untrained model | 25.5948 |
| after 15 steps | 22.6516 |
| after 60 steps | **18.6391** |

Eval loss dropped 4.0125 over the run (17.7%), and is -1.1313 relative to the control-mean baseline.

**It never beat 'predict the control mean'.** Everything it learned is the cell line's baseline expression; nothing drug-specific is being captured yet.

## Eval curve

| step | train (last window) | eval | total |g| | s/step |
|---|---|---|---|---|
| 15 | 22.9883 | 22.6516 | 2.696e+00 | 8.50 |
| 30 | 22.2418 | 21.9779 | 1.881e+00 | 8.57 |
| 45 | 21.1926 | 19.9878 | 1.767e+01 | 8.37 |
| 60 | 19.0549 | 18.6391 | 8.948e+00 | 8.33 |

## Gradient flow (final step, before clipping)

| module | ‖grad‖ | share of total |
|---|---|---|
| coordi_layer | 8.438e+00 | 88.9% |
| conc | 2.559e+00 | 8.2% |
| merger | 1.277e+00 | 2.0% |
| layer0 | 7.781e-01 | 0.8% |
| chem_embedding | 2.851e-01 | 0.1% |
| gene_embedding | 7.326e-02 | 0.0% |

### Vanishing check — 4 stacked layers, same batch

| layer | ‖grad‖ | vs layer0 |
|---|---|---|
| layer0 | 1.385e+00 | 1.000x |
| layer1 | 1.374e+00 | 0.992x |
| layer2 | 1.379e+00 | 0.995x |
| layer3 | 1.383e+00 | 0.998x |

Layer ratio (last/first): **0.998x** — no vanishing.

## Input gradient (the silent-gene bottleneck)

- ‖dL/dcontrol‖ = 2.824e+10
- per-gene: max 6.391e+08, median 4.935e+07 (**max/median 1.29e+01**)
- genes with exactly zero gradient: 0.0%

## Is the chemical path alive? (drug ablation)

- swap the drug, keep control + coords: output moves 2.2% (relative L2)
- mask the chemical entirely: output moves 1.6%
