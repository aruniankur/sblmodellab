# Gradient-flow diagnostics

Test-only harness for `models/model.py`. **The model is not modified** — every
number here comes from autograd hooks, retained gradients and chunked
re-implementation of the attention probabilities from the model's own q/k
tensors. `F.scaled_dot_product_attention` is never patched.

## Run

```bash
# headline run: 20k genes, 20k x 20k part-1 kernel
python models/testing/gradient_flow.py --genes 20000 --cells 32 --steps 40

# contrast run: i.i.d. noise instead of co-expressed genes
python models/testing/gradient_flow.py --genes 20000 --cells 32 --data noise --quick

# fast structural-only pass
python models/testing/gradient_flow.py --genes 2000 --quick

# full option list
python models/testing/gradient_flow.py --help
```

Useful flags: `--genes`, `--cells`, `--batch`, `--dim`, `--conc-dim`, `--heads`,
`--layers`, `--steps`, `--depth-sweep`, `--sparsity`, `--data structured|noise|both`,
`--quick`, `--no-sgd`, `--tag`, `--out-dir`.

`--cells 32` with `--conc-dim 32` is deliberate: `control_cell_line` and
`concise_emb` then have the same shape, so their gradient norms are directly
comparable.

## What each phase measures

| phase | question it answers |
|---|---|
| `structure` | one eval-mode fwd/bwd with hooks on every block: activation RMS, ‖∂L/∂stage‖, and **local transmission ratio** ‖grad_in‖/‖grad_out‖ per module |
| `attention` | exact (chunked) softmax statistics for all three SDPA calls: entropy vs uniform, max prob, **self prob**, effective support, **logit std**, plus the q/k/v split of the gradient |
| `train` | Adam and SGD on a fixed batch pool: held-out loss, grad norm, **gradient oneness** max\|g\|/‖g‖, per-group cross-batch grad cosine |
| `depth` | `num_layers` sweep: does the stack attenuate the gradient reaching the input? |

## Outputs

- `gradient_flow_report<tag>.md` — human-readable report
- `gradient_flow_report<tag>.json` — every number
- `grad_by_stage<tag>.png`, `grad_by_param_group<tag>.png`,
  `training_curves_<opt><tag>.png`, `depth_sweep<tag>.png`

## Data

`structured` builds **sparse Poisson counts with co-expression modules**
(~78% zeros, like sparse scRNA-seq) instead of dense non-negative noise, which
is important: dense non-negative data pushes every gene-gene cosine to ~0.9 and
makes the part-1 kernel degenerate. ~2% of genes are silenced (all-zero rows) in
both modes so the model's `clamp_min(1e-12)` branch is exercised.

## Summary of the 20k-gene run

See `gradient_flow_report_g20k.md` for the full tables.

1. **Part-2 gene↔gene attention is uniform at init** — logit std 0.05, entropy
   14.286 of 14.29 bits. ∥∂L/∂q∥ and ∥∂L/∂k∥ are ~150× smaller than ∥∂L/∂v∥,
   so the block cannot learn to sharpen.
2. **Silent genes dominate ‖∂L/∂input‖** — 450/20000 all-zero genes carry 100%
   of the Q/K-path gradient (`x / clamp_min(‖x‖, 1e-12)` gives them a 1e12
   backward gain). Affects input gradients only, not parameter gradients.
3. **The gradient is rank-1 at init** — oneness 0.91, owned by
   `coordi_layer.4.bias`. Cross-batch cosine is therefore just the sign of that
   one scalar and is meaningless on its own.
4. **No vanishing with depth** — expressed-gene ‖∂L/∂control‖ is flat from 1 to
   8 layers. Residual connections hold the block transmission at ~1.0 even
   though each FFN only passes ~0.17.
5. **Part-1 cosine kernel is healthy on sparse data** — self prob 0.034,
   effective support 20.8% of the panel. On dense data it goes nearly uniform
   (self prob 0.0005), i.e. the high-cosine failure mode is under-selectivity,
   not self-locking.
