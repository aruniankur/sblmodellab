# Gradient-flow report — `models/model.py`

- device `cuda` · float32 · seed 0
- genes **20,000** · cells 32 · batch 1 · chemicals 10 · dim 256 · conc_dim 32 · heads 4 · layers 2
- data `structured` (zero-inflation 0.7) · dropout off during analysis

## 1. Headline

| quantity | value |
|---|---|
| loss | 8.3332e+01 |
| ‖∂L/∂control‖ — Q/K path (all genes) | 8.8451e+06 |
| ‖∂L/∂concise_emb‖ — V path | 7.0139e-04 |
| **Q/K ÷ V, all genes** | **1.2611e+10** |
| **Q/K ÷ V, silent genes excluded** | **5.6045e-01** |
| share of Q/K grad coming from silent genes | 100.00% |
| silent genes | 450 / 20000 |
| per-gene input-grad max ÷ median | 7.3962e+11 |
| gradient oneness max\|g\|/‖g‖ | 0.9095 |
| largest gradient coordinate | `coordi_layer.4.bias` |

## 2. Part-1 cosine kernel

| metric | value |
|---|---|
| cosine mean / std | +0.188 / 0.165 |
| cosine p50 / p90 / p99 | +0.168 / +0.420 / +0.614 |
| softmax entropy | 11.825 bits (uniform 14.29) |
| mean max prob | 0.0340 |
| mean self prob (diagonal) | 0.0340 |
| effective support | 4168.4 genes ( 20.84%) |
| rebuilt attn@V vs real SDPA — max abs err | 3.8743e-07 |
| rebuilt attn@V vs real SDPA — rel err | 6.9538e-07 |

## 3. Part-2 attention

| SDPA call | entropy bits | uniform | logit std | max prob | self prob | eff. support | support % |
|---|---|---|---|---|---|---|---|
| genes → chemicals | 2.800 | 3.32 | 0.1093 | 0.1643 | – | 7.0 |  69.66% |
| genes ↔ genes | 14.286 | 14.29 | 0.0504 | 0.0001 | 0.0000 | 19975.1 |  99.88% |

## 3b. Attention gradient split (RMS of ∂L/∂q, ∂L/∂k, ∂L/∂v)

A uniform softmax has a near-zero Jacobian, so it starves q and k.

| layer.stage | q RMS | k RMS | v RMS | q/v |
|---|---|---|---|---|
| `L0.stage1` | 5.3698e-09 | 3.5295e-06 | 3.2353e-04 | 1.660e-05 |
| `L0.stage2` | 1.2617e-08 | 1.4972e-08 | 1.9703e-06 | 6.404e-03 |
| `L1.stage1` | 5.3591e-09 | 4.2411e-06 | 3.1522e-04 | 1.700e-05 |
| `L1.stage2` | 1.1782e-08 | 1.4362e-08 | 1.7962e-06 | 6.560e-03 |

## 4. Stage-by-stage gradient norms

| stage | activation RMS | ‖∂L/∂stage‖ | grad RMS |
|---|---|---|---|
| `gene_embedding` | 9.9979e-01 | 1.9346e-02 | 8.5500e-06 |
| `chem_embedding` | 5.7468e-01 | 1.2954e-02 | 2.5602e-04 |
| `conc_emb` | 5.7988e-01 | 7.0139e-04 | 8.7674e-07 |
| `part1_sdpa_out` | 4.5225e-02 | 9.9565e-04 | 1.2446e-06 |
| `conc_ffn` | 3.2942e-02 | 6.2078e-03 | 7.7598e-06 |
| `merger` | 5.4466e-01 | 3.7318e-02 | 1.6493e-05 |
| `L0.q` | 3.1479e-01 | 1.2150e-05 | 5.3698e-09 |
| `L0.kv` | 3.3218e-01 | 1.6370e-02 | 2.2878e-04 |
| `L0.chem_ffn` | 3.2468e-02 | 1.8707e-03 | 8.2673e-07 |
| `L0.kqv` | 2.2437e-01 | 4.4585e-03 | 1.1376e-06 |
| `L0.gene_ffn` | 2.0820e-02 | 3.7219e-02 | 1.6449e-05 |
| `L0.out` | 5.4509e-01 | 3.7219e-02 | 1.6449e-05 |
| `L1.q` | 3.1649e-01 | 1.2126e-05 | 5.3591e-09 |
| `L1.kv` | 3.3256e-01 | 1.5950e-02 | 2.2291e-04 |
| `L1.chem_ffn` | 3.1038e-02 | 1.6794e-03 | 7.4221e-07 |
| `L1.kqv` | 2.2351e-01 | 4.0645e-03 | 1.0371e-06 |
| `L1.gene_ffn` | 2.1144e-02 | 3.7249e-02 | 1.6462e-05 |
| `L1.out` | 5.4568e-01 | 3.7249e-02 | 1.6462e-05 |

## 5. Local transmission ratios (‖grad_in‖ / ‖grad_out‖)

Ratio ≪ 1 = attenuating block; ratio > 1 = amplifying block.

| block | ‖grad_in‖ | ‖grad_out‖ | ratio |
|---|---|---|---|
| `coordi_layer_last` | 1.7166e-01 | 2.7140e-01 | 0.6325 |
| `coordi_layer_first` | 1.5767e-02 | 2.6994e-02 | 0.5841 |
| `L1.gene_ffn` | 5.8496e-03 | 3.7249e-02 | 0.1570 |
| `L1.kqv` | 2.3581e-03 | 4.0645e-03 | 0.5802 |
| `L1.chem_ffn` | 2.9839e-04 | 1.6794e-03 | 0.1777 |
| `L1.q_linear` | 6.7330e-06 | 1.2126e-05 | 0.5552 |
| `L1.kv_linear` | 9.4879e-03 | 1.5950e-02 | 0.5948 |
| `L1.block` | 3.7219e-02 | 3.7249e-02 | 0.9992 |
| `L0.gene_ffn` | 6.3188e-03 | 3.7219e-02 | 0.1698 |
| `L0.kqv` | 2.5707e-03 | 4.4585e-03 | 0.5766 |
| `L0.chem_ffn` | 3.0626e-04 | 1.8707e-03 | 0.1637 |
| `L0.q_linear` | 6.6805e-06 | 1.2150e-05 | 0.5498 |
| `L0.kv_linear` | 9.0583e-03 | 1.6370e-02 | 0.5533 |
| `L0.block` | 3.7318e-02 | 3.7219e-02 | 1.0027 |
| `merger` | 2.0294e-02 | 3.7318e-02 | 0.5438 |
| `conc_ffn` | 9.9565e-04 | 6.2078e-03 | 0.1604 |
| `conc_emb` | 4.2203e-04 | 7.0139e-04 | 0.6017 |

## 6. Parameter gradient norms

| group | ‖g‖ | ‖p‖ | ‖g‖/‖p‖ | zero-grad |
|---|---|---|---|---|
| `gene_embedding` | 1.9346e-02 | 2.2623e+03 | 8.552e-06 | 0/1 |
| `chem_embedding` | 1.5642e-01 | 9.2197e+00 | 1.697e-02 | 0/2 |
| `layers` | 6.5861e+00 | 5.2332e+01 | 1.259e-01 | 0/28 |
| `conc_emb` | 9.4717e-02 | 3.2819e+00 | 2.886e-02 | 0/2 |
| `conc_ffn` | 9.4342e-01 | 1.3663e+01 | 6.905e-02 | 0/4 |
| `merger` | 4.2309e+00 | 9.2392e+00 | 4.579e-01 | 0/2 |
| `coordi_layer` | 7.8364e+01 | 8.0536e+00 | 9.730e+00 | 0/6 |

## 7. Gene / unit coverage

| metric | value |
|---|---|
| genes with zero ∂L/∂input | 0 / 20000 |
| embedding rows with zero grad | 0 |
| per-gene input grad min / median / max | 6.1524e-08 / 1.5182e-06 / 1.1229e+06 |

| GELU site | fraction of negative inputs |
|---|---|
| `conc_ffn.1` |  49.33% |
| `layers.0.chem_ffn.1` |  50.18% |
| `layers.0.gene_ffn.1` |  49.99% |
| `layers.1.chem_ffn.1` |  51.09% |
| `layers.1.gene_ffn.1` |  52.46% |
| `coordi_layer.1` |  49.20% |
| `coordi_layer.3` |  48.03% |

## 8. Training dynamics

### adam (lr 0.001)

- held-out well loss 117.285416 → 101.183441 (drop 16.101974)
- total grad norm 1.0428e+02 → 1.4044e+01
- **diverged: False**
- cross-batch grad cosine -0.8695 (1 = consistent, 0 = noise, <0 = opposite)
- gradient oneness max|g|/‖g‖ = 0.6214 (≈1 ⇒ the gradient is one scalar, not a direction)

| parameter group | cross-batch grad cosine |
|---|---|
| `chem_embedding` | -0.0255 |
| `conc_emb` | -0.8921 |
| `conc_ffn` | -0.9743 |
| `coordi_layer` | -0.8850 |
| `gene_embedding` | -0.0229 |
| `layers` | -0.4586 |
| `merger` | -0.2790 |

| step | loss | ‖g‖ | oneness | ‖∂L/∂control‖ | ‖∂L/∂control‖ (expressed genes) |
|---|---|---|---|---|---|
| 0 | 117.285316 | 1.0428e+02 | 0.909 | 1.0569e+07 | 4.1774e-04 |
| 1 | 81.440971 | 5.9673e+01 | 0.892 | 1.0646e+07 | 4.7256e-04 |
| 2 | 88.290192 | 4.3747e+01 | 0.829 | 1.6737e+07 | 6.6487e-04 |
| 3 | 84.579803 | 2.4560e+01 | 0.573 | 3.8186e+07 | 6.0850e-04 |
| 4 | 80.890923 | 7.5001e+01 | 0.547 | 6.0913e+07 | 1.6299e-03 |
| 5 | 79.397079 | 9.2458e+00 | 0.620 | 1.9976e+07 | 3.8152e-04 |
| 6 | 94.014946 | 2.7209e+01 | 0.773 | 1.3013e+07 | 4.4000e-04 |
| 7 | 77.078171 | 1.9958e+01 | 0.803 | 1.1280e+07 | 3.4605e-04 |
| 8 | 102.424294 | 3.5313e+01 | 0.713 | 1.2683e+07 | 3.4629e-04 |
| 9 | 76.393158 | 5.6481e+00 | 0.645 | 1.1230e+07 | 2.1084e-04 |
| 10 | 86.388908 | 3.1272e+01 | 0.580 | 1.2314e+07 | 1.9769e-04 |
| 11 | 88.611465 | 6.1888e+01 | 0.837 | 1.2086e+07 | 4.9749e-04 |
| 12 | 79.694443 | 4.1557e+01 | 0.828 | 9.9481e+06 | 3.8619e-04 |
| 13 | 79.277428 | 9.5071e+00 | 0.785 | 9.1115e+06 | 2.2524e-04 |
| 14 | 95.481705 | 4.4041e+01 | 0.828 | 1.2157e+07 | 4.9650e-04 |
| 15 | 76.885216 | 1.4438e+01 | 0.799 | 1.1219e+07 | 3.3391e-04 |
| 16 | 104.146561 | 5.0642e+01 | 0.787 | 2.0715e+07 | 7.8674e-04 |
| 17 | 77.381119 | 3.5599e+01 | 0.661 | 2.7978e+07 | 6.3665e-04 |
| 18 | 86.370476 | 2.6521e+01 | 0.653 | 2.5565e+07 | 4.0510e-04 |
| 19 | 83.928993 | 1.0051e+01 | 0.677 | 2.1984e+07 | 2.9010e-04 |
| 20 | 77.606094 | 3.5761e+00 | 0.789 | 1.7395e+07 | 3.4002e-04 |
| 21 | 79.179428 | 1.9042e+00 | 0.598 | 1.6004e+07 | 3.0950e-04 |
| 22 | 93.454643 | 1.8166e+01 | 0.780 | 1.3280e+07 | 3.4139e-04 |
| 23 | 76.682571 | 2.7885e+00 | 0.574 | 1.6137e+07 | 2.7114e-04 |
| 24 | 102.920723 | 3.7350e+01 | 0.804 | 1.4538e+07 | 4.6995e-04 |
| 25 | 76.403740 | 8.2495e+00 | 0.721 | 1.8392e+07 | 2.8482e-04 |
| 26 | 85.831604 | 1.5937e+00 | 0.341 | 1.8861e+07 | 3.1148e-04 |
| 27 | 83.844551 | 8.5630e+00 | 0.686 | 2.2196e+07 | 2.9085e-04 |
| 28 | 77.842216 | 1.7786e+01 | 0.695 | 1.9805e+07 | 3.0397e-04 |
| 29 | 79.253685 | 1.0282e+01 | 0.725 | 1.8848e+07 | 3.0495e-04 |
| 30 | 93.028526 | 2.3135e+00 | 0.275 | 1.5660e+07 | 2.7102e-04 |
| 31 | 76.907478 | 1.5008e+01 | 0.730 | 1.7677e+07 | 2.5876e-04 |
| 32 | 101.307785 | 8.1172e+00 | 0.681 | 1.1446e+07 | 2.3598e-04 |
| 33 | 76.260727 | 1.0202e+00 | 0.136 | 1.2966e+07 | 2.1531e-04 |
| 34 | 86.166412 | 1.8527e+01 | 0.803 | 1.1343e+07 | 2.6931e-04 |
| 35 | 83.860039 | 1.3078e+01 | 0.781 | 1.2867e+07 | 2.3723e-04 |
| 36 | 78.390961 | 2.7454e+01 | 0.814 | 1.1418e+07 | 3.7249e-04 |
| 37 | 79.107964 | 5.2068e+00 | 0.816 | 1.2403e+07 | 2.6982e-04 |
| 38 | 92.933556 | 5.8041e+00 | 0.731 | 1.2786e+07 | 2.5008e-04 |
| 39 | 76.846169 | 1.4044e+01 | 0.756 | 1.7199e+07 | 2.6093e-04 |

| parameter | ‖Δp‖/‖p‖ (step 0) |
|---|---|
| `gene_embedding.weight` | 9.915e-04 |
| `chem_embedding.weight` | 1.979e-02 |
| `chem_embedding.bias` | 1.931e-02 |
| `layers.0.kv_linear.weight` | 2.758e-02 |
| `layers.0.kv_linear.bias` | 1.980e-02 |
| `layers.0.q_linear.weight` | 2.739e-02 |
| `layers.0.q_linear.bias` | 2.843e-02 |
| `layers.0.chem_ffn.0.weight` | 2.771e-02 |
| `layers.0.chem_ffn.0.bias` | 2.747e-02 |
| `layers.0.chem_ffn.2.weight` | 5.534e-02 |
| `layers.0.chem_ffn.2.bias` | 5.428e-02 |
| `layers.0.kqv.weight` | 3.775e-02 |
| `layers.0.kqv.bias` | 3.125e-02 |
| `layers.0.gene_ffn.0.weight` | 2.771e-02 |
| `layers.0.gene_ffn.0.bias` | 2.842e-02 |
| `layers.0.gene_ffn.2.weight` | 5.541e-02 |
| `layers.0.gene_ffn.2.bias` | 5.626e-02 |
| `layers.1.kv_linear.weight` | 2.753e-02 |
| `layers.1.kv_linear.bias` | 1.962e-02 |
| `layers.1.q_linear.weight` | 2.741e-02 |
| `layers.1.q_linear.bias` | 2.747e-02 |
| `layers.1.chem_ffn.0.weight` | 2.772e-02 |
| `layers.1.chem_ffn.0.bias` | 2.811e-02 |
| `layers.1.chem_ffn.2.weight` | 5.532e-02 |
| `layers.1.chem_ffn.2.bias` | 5.858e-02 |
| `layers.1.kqv.weight` | 3.778e-02 |
| `layers.1.kqv.bias` | 3.185e-02 |
| `layers.1.gene_ffn.0.weight` | 2.770e-02 |
| `layers.1.gene_ffn.0.bias` | 2.731e-02 |
| `layers.1.gene_ffn.2.weight` | 5.533e-02 |
| `layers.1.gene_ffn.2.bias` | 5.744e-02 |
| `conc_emb.weight` | 2.760e-02 |
| `conc_emb.bias` | 2.857e-02 |
| `conc_ffn.0.weight` | 9.790e-03 |
| `conc_ffn.0.bias` | 9.772e-03 |
| `conc_ffn.2.weight` | 3.929e-02 |
| `conc_ffn.2.bias` | 3.946e-02 |
| `merger.weight` | 2.944e-02 |
| `merger.bias` | 3.060e-02 |
| `coordi_layer.0.weight` | 2.777e-02 |
| `coordi_layer.0.bias` | 2.850e-02 |
| `coordi_layer.2.weight` | 1.960e-02 |
| `coordi_layer.2.bias` | 1.969e-02 |
| `coordi_layer.4.weight` | 1.265e-02 |
| `coordi_layer.4.bias` | 2.241e-02 |

### sgd (lr 0.001)

- held-out well loss 117.285416 → 101.869804 (drop 15.415611)
- total grad norm 1.0428e+02 → 8.2835e+00
- **diverged: False**
- cross-batch grad cosine -0.9647 (1 = consistent, 0 = noise, <0 = opposite)
- gradient oneness max|g|/‖g‖ = 0.9069 (≈1 ⇒ the gradient is one scalar, not a direction)

| parameter group | cross-batch grad cosine |
|---|---|
| `chem_embedding` | -0.0383 |
| `conc_emb` | -0.9503 |
| `conc_ffn` | -0.9872 |
| `coordi_layer` | -0.9692 |
| `gene_embedding` | -0.0114 |
| `layers` | -0.9707 |
| `merger` | -0.2954 |

| step | loss | ‖g‖ | oneness | ‖∂L/∂control‖ | ‖∂L/∂control‖ (expressed genes) |
|---|---|---|---|---|---|
| 0 | 117.285316 | 1.0428e+02 | 0.909 | 1.0569e+07 | 4.1774e-04 |
| 1 | 79.268150 | 4.4287e+01 | 0.908 | 5.7946e+06 | 2.5232e-04 |
| 2 | 88.186249 | 3.9791e+01 | 0.906 | 5.7843e+06 | 2.2159e-04 |
| 3 | 84.632095 | 2.2699e+01 | 0.905 | 6.4908e+06 | 1.6658e-04 |
| 4 | 77.766403 | 9.3425e+00 | 0.903 | 6.1142e+06 | 1.2960e-04 |
| 5 | 79.306076 | 5.9620e+00 | 0.886 | 5.4529e+06 | 1.2416e-04 |
| 6 | 93.678009 | 1.8535e+01 | 0.903 | 5.6691e+06 | 1.4705e-04 |
| 7 | 76.717293 | 4.1068e+00 | 0.836 | 6.2315e+06 | 1.0461e-04 |
| 8 | 102.082466 | 2.1894e+01 | 0.904 | 5.6442e+06 | 1.5296e-04 |
| 9 | 76.563606 | 1.0449e+01 | 0.882 | 5.3700e+06 | 1.0595e-04 |
| 10 | 85.908150 | 4.2259e+00 | 0.875 | 5.4576e+06 | 1.1140e-04 |
| 11 | 83.878418 | 1.9973e+00 | 0.321 | 7.4562e+06 | 1.0557e-04 |
| 12 | 77.685127 | 6.3059e+00 | 0.860 | 6.7215e+06 | 1.0174e-04 |
| 13 | 79.270927 | 4.5713e+00 | 0.822 | 5.7712e+06 | 1.0674e-04 |
| 14 | 93.373756 | 1.2042e+01 | 0.899 | 5.7120e+06 | 1.2585e-04 |
| 15 | 76.785233 | 8.2135e+00 | 0.886 | 6.3428e+06 | 1.0006e-04 |
| 16 | 101.905785 | 1.9077e+01 | 0.904 | 5.5135e+06 | 1.4174e-04 |
| 17 | 76.609154 | 1.2187e+01 | 0.888 | 5.3561e+06 | 1.0488e-04 |
| 18 | 85.889008 | 3.0696e+00 | 0.847 | 5.4142e+06 | 1.0648e-04 |
| 19 | 83.866837 | 2.3526e+00 | 0.561 | 7.3414e+06 | 1.0222e-04 |
| 20 | 77.683449 | 6.7791e+00 | 0.868 | 6.5969e+06 | 9.9166e-05 |
| 21 | 79.262711 | 4.8673e+00 | 0.837 | 5.6425e+06 | 1.0409e-04 |
| 22 | 93.348480 | 1.1779e+01 | 0.900 | 5.6185e+06 | 1.2247e-04 |
| 23 | 76.784019 | 8.3247e+00 | 0.888 | 6.2127e+06 | 9.7824e-05 |
| 24 | 101.889282 | 1.8939e+01 | 0.905 | 5.3831e+06 | 1.3840e-04 |
| 25 | 76.599457 | 1.2176e+01 | 0.890 | 5.2400e+06 | 1.0264e-04 |
| 26 | 85.882317 | 2.9884e+00 | 0.848 | 5.3371e+06 | 1.0404e-04 |
| 27 | 83.854225 | 2.3012e+00 | 0.576 | 7.1872e+06 | 1.0002e-04 |
| 28 | 77.674133 | 6.7780e+00 | 0.872 | 6.4547e+06 | 9.7013e-05 |
| 29 | 79.251816 | 4.8674e+00 | 0.843 | 5.5081e+06 | 1.0184e-04 |
| 30 | 93.332016 | 1.1724e+01 | 0.901 | 5.5282e+06 | 1.1986e-04 |
| 31 | 76.780411 | 8.3052e+00 | 0.890 | 6.0849e+06 | 9.5782e-05 |
| 32 | 101.878906 | 1.8892e+01 | 0.906 | 5.2643e+06 | 1.3557e-04 |
| 33 | 76.589066 | 1.2114e+01 | 0.892 | 5.1326e+06 | 1.0053e-04 |
| 34 | 85.876831 | 2.9461e+00 | 0.850 | 5.2658e+06 | 1.0188e-04 |
| 35 | 83.842804 | 2.2396e+00 | 0.584 | 7.0434e+06 | 9.8025e-05 |
| 36 | 77.665604 | 6.7633e+00 | 0.875 | 6.3239e+06 | 9.5044e-05 |
| 37 | 79.241920 | 4.8596e+00 | 0.848 | 5.3847e+06 | 9.9790e-05 |
| 38 | 93.317337 | 1.1680e+01 | 0.902 | 5.4458e+06 | 1.1749e-04 |
| 39 | 76.777328 | 8.2835e+00 | 0.892 | 5.9665e+06 | 9.3907e-05 |

| parameter | ‖Δp‖/‖p‖ (step 0) |
|---|---|
| `gene_embedding.weight` | 8.795e-09 |
| `chem_embedding.weight` | 3.508e-05 |
| `chem_embedding.bias` | 5.600e-05 |
| `layers.0.kv_linear.weight` | 2.426e-05 |
| `layers.0.kv_linear.bias` | 7.360e-05 |
| `layers.0.q_linear.weight` | 2.138e-07 |
| `layers.0.q_linear.bias` | 6.411e-06 |
| `layers.0.chem_ffn.0.weight` | 1.824e-05 |
| `layers.0.chem_ffn.0.bias` | 9.069e-05 |
| `layers.0.chem_ffn.2.weight` | 7.663e-05 |
| `layers.0.chem_ffn.2.bias` | 1.187e-03 |
| `layers.0.kqv.weight` | 4.326e-05 |
| `layers.0.kqv.bias` | 1.154e-03 |
| `layers.0.gene_ffn.0.weight` | 4.066e-05 |
| `layers.0.gene_ffn.0.bias` | 1.272e-03 |
| `layers.0.gene_ffn.2.weight` | 3.389e-04 |
| `layers.0.gene_ffn.2.bias` | 1.726e-02 |
| `layers.1.kv_linear.weight` | 2.275e-05 |
| `layers.1.kv_linear.bias` | 6.852e-05 |
| `layers.1.q_linear.weight` | 2.781e-07 |
| `layers.1.q_linear.bias` | 6.772e-06 |
| `layers.1.chem_ffn.0.weight` | 1.529e-05 |
| `layers.1.chem_ffn.0.bias` | 7.995e-05 |
| `layers.1.chem_ffn.2.weight` | 6.478e-05 |
| `layers.1.chem_ffn.2.bias` | 1.150e-03 |
| `layers.1.kqv.weight` | 4.278e-05 |
| `layers.1.kqv.bias` | 1.071e-03 |
| `layers.1.gene_ffn.0.weight` | 3.871e-05 |
| `layers.1.gene_ffn.0.bias` | 1.168e-03 |
| `layers.1.gene_ffn.2.weight` | 3.473e-04 |
| `layers.1.gene_ffn.2.bias` | 1.765e-02 |
| `conc_emb.weight` | 4.522e-06 |
| `conc_emb.bias` | 6.276e-04 |
| `conc_ffn.0.weight` | 3.502e-06 |
| `conc_ffn.0.bias` | 9.816e-05 |
| `conc_ffn.2.weight` | 2.873e-04 |
| `conc_ffn.2.bias` | 5.498e-03 |
| `merger.weight` | 2.838e-04 |
| `merger.bias` | 9.438e-03 |
| `coordi_layer.0.weight` | 1.132e-03 |
| `coordi_layer.0.bias` | 2.132e-02 |
| `coordi_layer.2.weight` | 3.537e-03 |
| `coordi_layer.2.bias` | 7.484e-02 |
| `coordi_layer.4.weight` | 3.372e-02 |
| `coordi_layer.4.bias` | 2.125e+00 |

## 9. Depth sweep

| layers | loss | ‖g‖ | ‖∂L/∂control‖ | expressed-gene ‖∂L/∂control‖ | ‖∂L/∂merger.W‖ | newest layer gene_ffn.W |
|---|---|---|---|---|---|---|
| 1 | 87.2628 | 9.2990e+01 | 8.9937e+06 | 3.9766e-04 | 1.9407e+00 | 4.9092e-01 |
| 2 | 83.3322 | 7.8760e+01 | 8.8451e+06 | 3.9309e-04 | 2.0017e+00 | 5.2584e-01 |
| 4 | 80.0056 | 6.2268e+01 | 1.0196e+07 | 4.4591e-04 | 1.5731e+00 | 4.5717e-01 |
| 8 | 91.9886 | 1.1008e+02 | 1.4341e+07 | 6.3401e-04 | 2.5471e+00 | 8.9760e-01 |
