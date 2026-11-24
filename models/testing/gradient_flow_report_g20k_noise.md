# Gradient-flow report — `models/model.py`

- device `cuda` · float32 · seed 0
- genes **20,000** · cells 32 · batch 1 · chemicals 10 · dim 256 · conc_dim 32 · heads 4 · layers 2
- data `noise` (zero-inflation 0.7) · dropout off during analysis

## 1. Headline

| quantity | value |
|---|---|
| loss | 3.6871e+01 |
| ‖∂L/∂control‖ — Q/K path (all genes) | 1.0606e+06 |
| ‖∂L/∂concise_emb‖ — V path | 1.1284e-03 |
| **Q/K ÷ V, all genes** | **9.3997e+08** |
| **Q/K ÷ V, silent genes excluded** | **7.0762e-01** |
| share of Q/K grad coming from silent genes | 100.00% |
| silent genes | 418 / 20000 |
| per-gene input-grad max ÷ median | 2.6547e+10 |
| gradient oneness max\|g\|/‖g‖ | 0.9093 |
| largest gradient coordinate | `coordi_layer.4.bias` |

## 2. Part-1 cosine kernel

| metric | value |
|---|---|
| cosine mean / std | +0.721 / 0.161 |
| cosine p50 / p90 / p99 | +0.753 / +0.821 / +0.867 |
| softmax entropy | 14.091 bits (uniform 14.29) |
| mean max prob | 0.0005 |
| mean self prob (diagonal) | 0.0005 |
| effective support | 17461.2 genes ( 87.31%) |
| rebuilt attn@V vs real SDPA — max abs err | 8.9407e-08 |
| rebuilt attn@V vs real SDPA — rel err | 3.3047e-07 |

## 3. Part-2 attention

| SDPA call | entropy bits | uniform | logit std | max prob | self prob | eff. support | support % |
|---|---|---|---|---|---|---|---|
| genes → chemicals | 3.315 | 3.32 | 0.1066 | 0.1173 | – | 9.9 |  99.49% |
| genes ↔ genes | 14.286 | 14.29 | 0.0503 | 0.0001 | 0.0000 | 19975.2 |  99.88% |

## 3b. Attention gradient split (RMS of ∂L/∂q, ∂L/∂k, ∂L/∂v)

A uniform softmax has a near-zero Jacobian, so it starves q and k.

| layer.stage | q RMS | k RMS | v RMS | q/v |
|---|---|---|---|---|
| `L0.stage1` | 6.9434e-09 | 5.2085e-06 | 4.4521e-04 | 1.560e-05 |
| `L0.stage2` | 1.6551e-08 | 2.3467e-08 | 3.2360e-06 | 5.115e-03 |
| `L1.stage1` | 8.6815e-09 | 6.6823e-06 | 4.3367e-04 | 2.002e-05 |
| `L1.stage2` | 1.5495e-08 | 2.2638e-08 | 2.9550e-06 | 5.244e-03 |

## 4. Stage-by-stage gradient norms

| stage | activation RMS | ‖∂L/∂stage‖ | grad RMS |
|---|---|---|---|
| `gene_embedding` | 9.9979e-01 | 2.5390e-02 | 1.1221e-05 |
| `chem_embedding` | 5.6853e-01 | 1.7616e-02 | 3.4816e-04 |
| `conc_emb` | 5.7988e-01 | 1.1284e-03 | 1.4105e-06 |
| `part1_sdpa_out` | 3.5889e-02 | 1.2941e-03 | 1.6176e-06 |
| `conc_ffn` | 3.2628e-02 | 8.1451e-03 | 1.0181e-05 |
| `merger` | 5.4466e-01 | 4.9061e-02 | 2.1682e-05 |
| `L0.q` | 3.1479e-01 | 1.5711e-05 | 6.9434e-09 |
| `L0.kv` | 3.2994e-01 | 2.2528e-02 | 3.1483e-04 |
| `L0.chem_ffn` | 2.5625e-02 | 3.0702e-03 | 1.3568e-06 |
| `L0.kqv` | 2.2416e-01 | 7.3225e-03 | 1.8684e-06 |
| `L0.gene_ffn` | 2.0763e-02 | 4.8854e-02 | 2.1591e-05 |
| `L0.out` | 5.4509e-01 | 4.8854e-02 | 2.1591e-05 |
| `L1.q` | 3.1650e-01 | 1.9644e-05 | 8.6815e-09 |
| `L1.kv` | 3.3354e-01 | 2.1945e-02 | 3.0669e-04 |
| `L1.chem_ffn` | 2.6875e-02 | 2.7598e-03 | 1.2196e-06 |
| `L1.kqv` | 2.2333e-01 | 6.6868e-03 | 1.7062e-06 |
| `L1.gene_ffn` | 2.1165e-02 | 4.8919e-02 | 2.1619e-05 |
| `L1.out` | 5.4568e-01 | 4.8919e-02 | 2.1619e-05 |

## 5. Local transmission ratios (‖grad_in‖ / ‖grad_out‖)

Ratio ≪ 1 = attenuating block; ratio > 1 = amplifying block.

| block | ‖grad_in‖ | ‖grad_out‖ | ratio |
|---|---|---|---|
| `coordi_layer_last` | 1.1419e-01 | 1.8053e-01 | 0.6325 |
| `coordi_layer_first` | 1.0492e-02 | 1.7959e-02 | 0.5842 |
| `L1.gene_ffn` | 7.6926e-03 | 4.8919e-02 | 0.1573 |
| `L1.kqv` | 3.8776e-03 | 6.6868e-03 | 0.5799 |
| `L1.chem_ffn` | 4.9065e-04 | 2.7598e-03 | 0.1778 |
| `L1.q_linear` | 1.0193e-05 | 1.9644e-05 | 0.5189 |
| `L1.kv_linear` | 1.3084e-02 | 2.1945e-02 | 0.5962 |
| `L1.block` | 4.8854e-02 | 4.8919e-02 | 0.9987 |
| `L0.gene_ffn` | 8.2904e-03 | 4.8854e-02 | 0.1697 |
| `L0.kqv` | 4.2217e-03 | 7.3225e-03 | 0.5765 |
| `L0.chem_ffn` | 5.0371e-04 | 3.0702e-03 | 0.1641 |
| `L0.q_linear` | 9.5517e-06 | 1.5711e-05 | 0.6080 |
| `L0.kv_linear` | 1.2424e-02 | 2.2528e-02 | 0.5515 |
| `L0.block` | 4.9061e-02 | 4.8854e-02 | 1.0042 |
| `merger` | 2.6620e-02 | 4.9061e-02 | 0.5426 |
| `conc_ffn` | 1.2941e-03 | 8.1451e-03 | 0.1589 |
| `conc_emb` | 6.7831e-04 | 1.1284e-03 | 0.6011 |

## 6. Parameter gradient norms

| group | ‖g‖ | ‖p‖ | ‖g‖/‖p‖ | zero-grad |
|---|---|---|---|---|
| `gene_embedding` | 2.5390e-02 | 2.2623e+03 | 1.122e-05 | 0/1 |
| `chem_embedding` | 2.0447e-01 | 9.2197e+00 | 2.218e-02 | 0/2 |
| `layers` | 1.0756e+01 | 5.2332e+01 | 2.055e-01 | 0/28 |
| `conc_emb` | 1.5520e-01 | 3.2819e+00 | 4.729e-02 | 0/2 |
| `conc_ffn` | 1.5466e+00 | 1.3663e+01 | 1.132e-01 | 0/4 |
| `merger` | 6.9033e+00 | 9.2392e+00 | 7.472e-01 | 0/2 |
| `coordi_layer` | 1.2846e+02 | 8.0536e+00 | 1.595e+01 | 0/6 |

## 7. Gene / unit coverage

| metric | value |
|---|---|
| genes with zero ∂L/∂input | 0 / 20000 |
| embedding rows with zero grad | 0 |
| per-gene input grad min / median / max | 4.7453e-08 / 3.8795e-06 / 1.0299e+05 |

| GELU site | fraction of negative inputs |
|---|---|
| `conc_ffn.1` |  48.95% |
| `layers.0.chem_ffn.1` |  51.21% |
| `layers.0.gene_ffn.1` |  49.57% |
| `layers.1.chem_ffn.1` |  49.07% |
| `layers.1.gene_ffn.1` |  52.45% |
| `coordi_layer.1` |  49.11% |
| `coordi_layer.3` |  48.14% |

## 8. Training dynamics
