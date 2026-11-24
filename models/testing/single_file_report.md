# Single-file training report — `plate6_CVCL_1125.h5ad`

- cells 34262, genes 50266, drug wells 88 (train 80 / held out 8)
- dose(s) [0.05, 5.0], chem tokens `all`, dropout 0.1
- dim 512, heads 8, layers 1, conc_dim 32, lr 0.0001
- 120 steps in 18.6 min, 8.25 s/step, peak VRAM 6.58 GB (reserved 8.30 GB of 12.88)

## Is it learning?

| reference | loss |
|---|---|
| predict 0 | 23.4044 |
| predict control mean | 17.5078 |
| untrained model | 25.5948 |
| after 20 steps | 22.9710 |
| after 120 steps | **19.3247** |

Eval loss dropped 3.6464 over the run (15.9%), and is -1.8169 relative to the control-mean baseline.

**It never beat 'predict the control mean'.** Everything it learned is the cell line's baseline expression; nothing drug-specific is being captured yet.

## Eval curve

| step | train (last window) | eval | total |g| | s/step |
|---|---|---|---|---|
| 20 | 23.4772 | 22.9710 | 1.487e+01 | 8.40 |
| 40 | 22.5417 | 22.6542 | 1.303e+00 | 8.35 |
| 60 | 22.3206 | 22.2440 | 3.503e+00 | 8.27 |
| 80 | 21.7850 | 21.5091 | 7.448e+00 | 8.17 |
| 100 | 20.7703 | 20.0977 | 7.172e+00 | 8.17 |
| 120 | 19.5485 | 19.3247 | 2.564e+00 | 8.14 |

## Gradient flow (final step, before clipping)

| module | ‖grad‖ | share of total |
|---|---|---|
| coordi_layer | 8.887e-01 | 79.0% |
| conc | 3.256e-01 | 10.6% |
| merger | 3.209e-01 | 10.3% |
| layer0 | 2.379e-02 | 0.1% |
| gene_embedding | 2.326e-02 | 0.1% |
| chem_embedding | 3.835e-03 | 0.0% |

### Vanishing check — 4 stacked layers, same batch

| layer | ‖grad‖ | vs layer0 |
|---|---|---|
| layer0 | 1.385e+00 | 1.000x |
| layer1 | 1.374e+00 | 0.992x |
| layer2 | 1.379e+00 | 0.995x |
| layer3 | 1.383e+00 | 0.998x |

Layer ratio (last/first): **0.998x** — no vanishing.

## Input gradient (the silent-gene bottleneck)

- ‖dL/dcontrol‖ = 1.917e+10
- per-gene: max 4.163e+08, median 3.395e+07 (**max/median 1.23e+01**)
- genes with exactly zero gradient: 0.0%

## Is the chemical path alive? (drug ablation)

- swap the drug, keep control + coords: output moves 0.6% (relative L2)
- mask the chemical entirely: output moves 0.9%

**The drug barely affects the prediction** — the chemical branch is being ignored.
