# Training pipeline

Virtual-cell training on the Tahoe-100M `data_by_plate_cell_line/` grid.

```
models/
├── train.py                 # entry point: epoch plan, train/val loop, checkpoints
├── tahoe_data.py            # h5ad indexing, CSR row reader, normalisation, batch build
├── molformer_atoms.py       # MolFormer token cache (build once, then read-only)
├── cache/atom_embeddings.pt # 380 drugs x (n_tokens, 768) + atom rows [70 MB, built]
├── runs/<name>/             # log.csv, config.json, last.pt, best.pt
└── testing/check_chunk_parity.py
```

## Moving this to a server

Nothing in `models/` has a hard-coded path any more; everything is resolved
relative to the repo root, and the two data locations can be overridden.

```bash
git clone <repo> && cd <repo>

# 1. torch first, matching the server's CUDA:
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements-training.txt

# 2. the h5ad grid (593 GB, not in git) -> either put it at the default place...
#      data_by_plate_cell_line/plate{N}_{CELL_LINE}.h5ad
#    ...or point at it:
export TAHOE_DATA_ROOT=/scratch/tahoe/data_by_plate_cell_line

# 3. the drug table is in git; the token cache is not (67 MB, regenerable):
python models/molformer_atoms.py          # ~2 min, needs network once

# 4. sanity, then train
python models/train.py --dry-run
python models/train.py --epochs 60 --out runs/v1
```

What is **not** in the repo, by design (see `.gitignore`): the h5ad data, the
MolFormer token cache, checkpoints, TensorBoard events, `metadata/`,
`data_by_sample_embedded/`, notebooks and the analysis folders. So a fresh clone
is a few MB.

Two server-specific notes:

* **Chunk sizes are tuned for a 12 GB card.** On a big GPU raise `--readout-chunk`
  and `--query-chunk`, or drop them to `0` to disable chunking and let the
  kernels allocate freely — they only exist to bound peak memory.
* **`--file-cache` / `--load-group` control h5ad I/O.** With fast local storage
  and many cores you can raise them; the data reader is single-process, so the
  GPU may idle on slow network storage.

## What one training example is

| tensor | shape | source |
|---|---|---|
| `control` | `(1, 50266, 256)` | 256 cells from a **DMSO well** of the file, `log1p(CP10K)` |
| `chem` | `(1, N, 769)` | MolFormer tokens: **every real token** of the SMILES (atoms + `(`, `=`, `1`, `.` …) + `log10(dose + 1e-3)` |
| `chem_mask` | `(1, 1, 1, N)` | `True` = real token (padded to the batch max length) |
| `coords` | `(1, 128, 2)` | the 128 predicted cells' `coord_1/coord_2` |
| `target` | `(1, 128, 50266)` | those 128 cells' `log1p(CP10K)` expression |

`N` is the number of tokens per drug: 7 … 212, **median 57** (median 30 of those
are atoms). Lengths differ per drug, which is why `chem_mask` exists. The model's
stage-1 cross-attention already accepts an arbitrary key/value set, so *no
architecture change* was needed to go from one pooled vector to a token set.

`--chem-tokens atoms` restricts the keys to the atom rows (median 30, RDKit
aligned). The default is `all` because atom-only is a lossy filter: ethanol `CCO`
and acetaldehyde `CC=O` differ only by a `=` token and their atom vectors are
cosine **0.9998** apart, so atom-only cannot distinguish them. Everything else is
identical, and the extra keys cost ~2× a stage-1 term that is already ~870×
cheaper than the gene self-attention.

Dose is `log10(x + 1e-3)`, so the four Tahoe doses map to
`0 → -3.000`, `0.05 → -1.292`, `0.5 → -0.300`, `5 → 0.699`.

## Epoch layout

* Plates **13 and 14 are never opened** (`--exclude-plates 13,14`, enforced by
  omitting them from the file index).
* Every cell line visits `--passes-per-cell-line 5` **distinct plates** per epoch.
* Each `(cell line, plate)` unit = `--batches-per-unit 2` micro-batches, one per
  DMSO control sample, each with its own random drug; the unit shares one
  optimizer step.
* Files never mix across plates; they are opened lazily and kept in a small cache
  (`--load-group 5`, `--file-cache 6`), which is what "load 5 files at a time"
  amounts to.
* 48 train cell lines × 5 plates = **240 units = 480 micro-batches per epoch**
  (2 of the 50 cell lines are held out by `--val-cell-lines 2`).

## Run it

```bash
# 1. once: build the token cache (needs transformers + network, ~2 min on GPU)
python models/molformer_atoms.py

# 2. data path only, no model
python models/train.py --dry-run

# 3. smoke test
python models/train.py --limit-files 10 --epochs 1 --passes-per-cell-line 1

# 4. real run
python models/train.py --epochs 60 --out models/runs/v1
```

Resume with `--resume models/runs/v1/last.pt`. `last.pt` stores
`{model, optimizer, epoch, epoch_done, step, config}`; `best.pt` is written when
validation improves.

## Precision: use bf16

Measured with `models/testing/amp_benchmark.py` (identical weights, identical
real batches, `dim=512`, `layers=1`):

| mode | s/step | epoch (480 micro-batches) | peak GPU | steps applied | loss vs fp32 |
|---|---|---|---|---|---|
| `--amp off --no-tf32` | 18.6 | ~2.5 h | 8.75 GB | 6/6 | reference |
| `--amp off` (TF32) | 12.6 | ~1.7 h | 8.75 GB | 6/6 | reference |
| **`--amp bf16` (default)** | **7.2** | **~58 min** | **6.31 GB** | **6/6** | Δ ≈ 1e-4 rel. |
| `--amp fp16` | 7.1 | — | 6.04 GB | **0/6** | — |

bf16 tracks fp32 to four significant figures over the same batches
(`|g|` 51.88 vs 51.79). It needs no `GradScaler` because its 8-bit exponent has
fp32's range.

**fp16 does not work on this model.** Every step overflows (`|g| = inf`, 1–3
parameters non-finite); GradScaler halves `32768 → 16384 → … → 1024` and **skips
all 6 optimizer steps**. The cause is range, not precision: it overflows even
after the scaler's own rescaling, so it never recovers within a run. The flag is
kept only so the failure stays reproducible.

## Measured timings (RTX A2000 12 GB, dim=512, heads=8, G=50,266)

| setting | s / unit (2 micro-batches) | peak GPU | epoch (240 units) |
|---|---|---|---|
| `--amp bf16 --layers 1` (default) | 12.1 | 5.8–6.1 GB | ~58 min |
| `--amp bf16 --layers 2` | 21.8 | 7.1 GB | ~87 min |
| `--amp off --layers 1` | ~25 | 8.2 GB | ~1.7 h |

## `benchmark.py` — find this machine's limit

Trains **one** h5ad file at a shape you choose, so the numbers are real
(forward + backward + AdamW step) rather than a synthetic matmul. Genes always
come from the file; it asks for the five things that shape memory:

```
dim (width) | heads | layers | conc_dim | batch (wells stacked per forward)
```

```bash
python models/benchmark.py                       # pick the file, answer 5 prompts
python models/benchmark.py --file plate6_CVCL_1125.h5ad \
    --dim 1024 --heads 8 --layers 1 --batch 1
```

It reports, per step: data time (h5ad reads) and compute time separately, loss,
peak GPU **allocated** and peak **reserved** (what the allocator actually holds),
then a verdict against the card's total memory and a projected epoch time.
On CUDA OOM it stops and names the knob to lower. If one step exceeds
`--time-budget` (90 s) it gives up, because a box that has spilled into system
RAM produces a meaningless number, not a result.

Chunk sizes (`--query-chunk`, `--readout-chunk`) default to **auto**: each chunk
is sized to ~1 GB, so a bigger `dim` fails on the architecture rather than on the
probe's own bookkeeping. `0` disables a chunk (only safe for small shapes).

`dim / heads > 128` is rejected up front — above that the gene self-attention
cannot use a fused kernel, which is a kernel limit rather than a property of the
machine.

Reference points measured on this box (RTX A2000 12 GB, `plate6_CVCL_1125`):

| dim | heads | layers | batch | s/step | peak alloc | peak reserved | verdict |
|---|---|---|---|---|---|---|---|
| 512 | 8 | 1 | 1 | 6.6 | 4.29 GB | 6.31 GB (49 %) | fits |
| 1024 | 8 | 1 | 1 | 12.7 | 6.26 GB | 7.66 GB (59 %) | fits |
| 512 | 8 | 1 | 4 | 33.8 | 9.35 GB | 10.92 GB (85 %) | fits, 8.4 s/well |
| 1024 | 8 | 2 | 2 | 180.6 | 14.29 GB | — | spills to system RAM |

Two readings from that table: widening (`dim` 512 → 1024) is cheaper than
stacking layers, and `batch > 1` costs more *per well* (8.4 s vs 6.6 s) while only
reducing the number of optimizer steps — so batch is a throughput knob, not a
memory win.

## `train_small.py` — overfit one file

The cheap sanity test before committing GPU-days: can the architecture fit a
single perturbation experiment at all?

```bash
python models/train_small.py
```

```
700 files in data_by_plate_cell_line (showing the first 25)
    1. plate1_CVCL_0023.h5ad
    ...
file (number, or part of a name) [1]: CVCL_1125

dataset (genes always come from the file):
  cells         : 34262
  genes         : 50266
  drug wells    : 88  (>= 128 cells)
  control wells : 2 DMSO
  doses         : [0.05, 5.0]

model shape:
  dim (model width) [512]:
  heads [8]:
  layers [1]:
  conc_dim [32]:
  batch (wells per step) [1]:
  epochs [10]:
```

Genes are never asked for — they come from the file. Everything else is
overridable by flag (`--file --dim --heads --layers --conc-dim --batch --epochs`),
so the same run can be scripted.

* There is **no train/held-out split**: every drug well of the file is trained on,
  so the curves measure optimisation and memorisation, not generalisation.
* One **epoch** is one pass over the file's drug wells —
  `steps_per_epoch = ceil(n_drug_wells / batch)` (88 here), reshuffled each epoch.
* A fixed probe set (4 wells, fixed cells) is scored at the end of every epoch, so
  epoch-to-epoch numbers are comparable.
* Defaults chosen for fitting rather than generalising: `--dropout 0`,
  `--lr 3e-4`, `--grad-clip 10` (clipping at 1.0 truncates most steps, measured).

```
models/runtime/model_20261124_202259/
    info.json      config + per-epoch metrics, rewritten every epoch
    model.pt       weights, atomically overwritten every epoch
    log.csv        one row per step (epoch, step, loss, seconds, drugs)
    tb/            TensorBoard for this run
```

`info.json` is written **before** training starts, so a crashed run still
documents its configuration:

```json
{"run": "model_20261124_202259", "file": "plate6_CVCL_1125.h5ad",
 "model": {"dim": 256, "heads": 4, "layers": 1, "params": 14848065},
 "training": {"batch": 1, "epochs": 1, "steps_per_epoch": 88, "lr": 0.0003},
 "epochs": [{"epoch": 1, "train_loss": 21.62, "probe_loss": 18.85,
             "seconds": 338.0, "peak_vram_gb": 3.358}]}
```

Measured, `dim=256 / heads=4 / layers=1`, one epoch of `plate6_CVCL_1125`:
88 steps in 5.6 min (3.43 s/step), probe loss **23.68 → 18.85**, peak 3.36 GB.
At `dim=512` the same file runs ~8 s/step and ~6.5 GB.

## TensorBoard

```bash
python -m pip install tensorboard          # once; torch does not bundle it

python models/train.py --epochs 60 --out models/runs/v1
tensorboard --logdir models/runs           # then open http://localhost:6006
```

Logs go to `<out>/tb`, so every run under `models/runs` appears as its own set of
curves (`--no-tb` disables, `--tb-dir` relocates).

| tag | what |
|---|---|
| `loss/train` | loss per optimizer step |
| `loss/val` | loss on the 2 held-out cell lines, per epoch |
| `loss/epoch_train` | epoch mean |
| `loss/by_plate/plate06` | the same step value grouped by plate (12 series) |
| `loss/by_cell_line/CVCL_x` | grouped by cell line (48 series) |
| `loss/by_drug/<drug>` | only with `--tb-per-drug` (380 series) |
| `file/current` | **Text tab** — `plate6_CVCL_1125.h5ad \| drugs: ...` per step |
| `vram/step_peak_gb` | **peak VRAM during that unit** |
| `vram/step_peak_reserved_gb` | peak reserved (what the allocator really holds) |
| `vram/run_peak_gb` | running max over the run |
| `vram/total_gb` | the card's capacity, so the chart has a ceiling |
| `perf/seconds_per_unit`, `perf/wells_per_second` | throughput |

The VRAM series are per-step high-water marks: the trainer calls
`torch.cuda.reset_peak_memory_stats()` before each unit. (Logging
`memory_allocated()` after the optimizer step instead gives a flat ~0.55 GB line,
because by then the activations are freed — that mistake is why the reset is
there.)

TensorBoard is **on by default** for the benchmark: every run writes to
`models/runs/benchmark/<file>/<shape>/`, so two files can never collide and
re-running the same shape *replaces* the previous measurement instead of
appending to it.

```powershell
python models/benchmark.py --file plate6_CVCL_1125.h5ad --dim 512 --steps 3
python models/benchmark.py --file plate6_CVCL_1125.h5ad --dim 1024 --steps 3
python models/benchmark.py --file plate10_CVCL_0023.h5ad --dim 1024 --steps 3

python -m tensorboard.main --logdir models\runs            # everything
python -m tensorboard.main --logdir models\runs\benchmark  # benchmarks only
```

The runs panel then lists, e.g.:

```
benchmark/plate6_CVCL_1125/dim512_h8_l1_c32_b1
benchmark/plate6_CVCL_1125/dim1024_h8_l1_c32_b1
benchmark/plate10_CVCL_0023/dim1024_h8_l1_c32_b1
```

Tick several and overlay `loss/benchmark` against `vram/step_peak_gb` to see what
a bigger shape actually costs. `--no-tb` turns it off; if `tensorboard` is not
installed the benchmark warns and continues rather than aborting.

## Two memory-only knobs (the reason the model can train at all)

`G = 50,266` makes two tensors explode; both are handled *outside* the maths:

* `--query-chunk 4096` — part 1 attends `(1, G, C)` queries against `(1, G, C)`
  keys, i.e. the attention head dim is `C = 256`. No fused kernel supports that,
  so the math backend materialises `(1, 1, G, G)` fp32 ≈ **40.7 GB**. Splitting
  the query axis bounds it to ~1.8 GB.
* `--readout-chunk 16` — the readout builds `(1, 128, G, dim+2)` = **13.4 GB** at
  `dim=512`. Splitting the cell axis bounds it to ~0.9 GB.

Both are exact: `models/testing/check_chunk_parity.py` compares the default and
chunked paths on identical weights and reports forward `4.5e-08`, parameter
gradients `9.5e-07`, loss `0.0` (float32 round-off from kernel tiling only).
`model.py`'s default behaviour (`query_chunk=None, readout_chunk=None`) is the
original code path, byte for byte.

`--heads` must satisfy `dim / heads <= 128`, otherwise the gene self-attention
falls back to the math backend and OOMs. This is asserted at startup.

## Defaults chosen for the open questions

| question | default | flag |
|---|---|---|
| "wells with >32 cells" vs 128 outputs | eligibility `--min-cells 128`; below `--n-target` the cells are drawn **with replacement**, so `--min-cells 32` reproduces the original rule | `--min-cells`, `--n-target` |
| how many drugs per file | 2 (your spec). Raising it costs exactly one micro-batch per drug and the control tensors are already in RAM: `--batches-per-unit 8` sees 40 drugs/cell-line/epoch instead of 10 | `--batches-per-unit` |
| chemical keys | every MolFormer token (median 57/drug), not just atoms | `--chem-tokens all\|atoms` |
| 48 vs 50 cell lines | 2 held out for validation; they are never trained on | `--val-cell-lines` |
| 5 passes | 5 random *distinct plates* per cell line per epoch | `--passes-per-cell-line` |

## Things to watch

* **Coordinate shortcut.** The query coords are the *treated* cells' own atlas
  positions and the target is those same cells' expression, so a
  `coord → expression` map is learnable and the chemical branch can be ignored.
  The loss will still look fine. Diagnostic: keep the control and coords fixed
  and swap the drug — the prediction must move substantially. If it doesn't, the
  chemical path is dead.
* **Silent genes.** `mean(model.py:277) control / clamp_min(||x||, 1e-12)` gives a
  1e12 Jacobian for the ~450 all-zero genes in a well; they absorb ~100 % of
  `∂L/∂control`. Parameter gradients are unaffected, so this shows up as a
  training signal only through the input. See `testing/gradient_flow_report_g20k.md`.
* **`loss` scale** is unchanged from the reference: sum over genes, mean over
  cells and batch, divided by `n_genes ** 0.5`.
* A well with fewer than `--min-cells` cells is skipped (plate 1 has 8 wells with
  1–49 cells). Skipped units are counted in the epoch summary. Measured over the
  600 usable files:

  | threshold | files failing | units skipped / epoch |
  |---|---|---|
  | `--min-cells 32` | 24 (4 %) | ~10 of 240 |
  | `--min-cells 128` (default) | 44 (7 %) | ~18 of 240 |
  | `--min-cells 256` | 65 (11 %) | ~26 of 240 |

  The 24 worst files are 1,000–2,500 cells across 96 wells (10–27 cells/well) and
  are unusable at any threshold; they belong to a few consistently low-cell lines.
