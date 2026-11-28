# Tahoe model training

Start from the repository root:

```powershell
python run.py
```

The implementation lives entirely under `models/`. The root `run.py` is only a
launcher; `python models/run.py` and `cd models; python run.py` also work.

## Interactive workflow

1. Check imports for torch, numpy, scipy, pandas, anndata, h5py and tqdm.
2. Report the CUDA GPU and available VRAM. Use BF16 on supported GPUs, FP32
   otherwise; CPU fallback is reported explicitly.
3. Choose a new run or an existing saved run.
4. For a new run, choose `full` or `control`, then all files or one file.
5. Enter `heads`, `head_dim`, `layers`, forward batch size `B`, and `epochs`. Total width is
   `dim = heads * head_dim`; the co-expression bottleneck is `conc_dim = head_dim`.
   Gene count and chemical feature width come from the data and MolFormer cache.
6. For an existing run, choose the run and enter the number of additional epochs.
   Select the forward batch size B; each batch updates weights once.
   Saved architecture, optimizer and data selection are restored. Clipping uses
   the current default of `3.0`, including when resuming an older run.

For example, `heads=8` and `head_dim=128` creates `dim=1024` and `conc_dim=128`.
The control model also uses these derived widths, but has no attention stack,
so `layers` and `dropout` do not affect its computation. Existing checkpoints
retain their original widths and bottleneck, even if they predate this sizing rule.

New full-model runs enable a chemical FFN residual in every attention block:
`chem_context = chem_ffn_norm(chem_context + chem_ffn(chem_context))`.
The setting is saved as `chem_ffn_residual`; checkpoints from schema 5 or older
resume with their original behavior. Use `--no-chem-ffn-residual` for a new-run
ablation. The residual adds no parameters, but changes the forward computation.

## Two models, two training modes

| Model | Context | Target | Extra input |
|---|---|---|---|
| `full` | DMSO cells | Drug-treated cells | MolFormer tokens + dose, treated-cell coordinates |
| `control` | DMSO cells | DMSO cells | DMSO target-cell coordinates; no chemicals |

For both models, context and target always belong to the **same plate and cell
line**. Different DMSO wells and overlapping cells are allowed. All training is
restricted to plates **1-12**, including single-file training.

- **Full all-file mode (new runs):** visit every eligible training plate/cell-line
  file once per epoch. Select one batch of up to B treatment wells, rotating
  among the least-exposed available MolFormer classes, then chemicals and wells.
  B=2 uses different classes whenever at least two are available. Each item uses
  a DMSO well from that same file; DMSO assignment is shuffled without replacement
  before cycling. One forward/backward batch produces one optimizer update.
- **Legacy/control all-file mode:** select five distinct available plates per
  training cell line and up to `max(batch_size, batches_per_unit)` target wells
  per file. `--sampling-strategy legacy` selects this schedule for a new full run.
- **Single-file mode:** visit every eligible target well in the chosen file once
  per epoch, in random order, grouped into B-well batches. Each forward batch
  has one backward pass and one optimizer update.
- **Control:** train every eligible cell line with usable coordinates (48 in the
  intended dataset). No cell lines are reserved. Fixed DMSO reconstruction probes
  measure fitting; they are not held-out generalization measurements.
- **Full:** reserve two eligible cell lines by a fixed seeded split. Score them
  on their first two available plates with fixed-seed sampling. Single-file runs
  instead use fixed reconstruction probes from the same training file.

The planner checks finite-coordinate wells and matching gene identity/order.
Optional `_coord_verify.csv` bookkeeping skips files known to have no coordinates.
The actual usable files, training lines and validation lines are saved in JSON.
No cell-line IDs or gene counts are hard-coded. A file without DMSO context or an
eligible target well is omitted. The summary reports the actual number of lines.

## Data and defaults

`data_by_plate_cell_line/plate{N}_{CELL_LINE}.h5ad` must contain CSR raw counts,
`var` gene IDs in a consistent order, and `obs` columns `sample`, `drug`,
`canonical_smiles`, `concentration`, `coord_1`, and `coord_2`.

Only sampled rows are densified and converted to `log1p(CP10K)`. Defaults are
512 context cells and 128 target cells, with a target-well eligibility threshold
of 128 cells. Sampling uses replacement when a context or target pool is smaller
than its requested sample count. Duplicate sampled rows are read once and then
expanded back into the requested order.

The full model reads `models/cache/atom_embeddings.pt`. Build it once with:

```powershell
python models/molformer_atoms.py
```

The control path never loads that cache. Both models use the targeted hidden-state normalization described below. The objective is squared error summed over genes, averaged over
cells and examples, and divided by `n_genes ** 0.5`.

Full-grid defaults: AdamW, learning rate `1e-4`, weight decay `0.01`, gradient
clipping `3.0`, dropout `0.1`, eight heads of width `64` (total width `512`,
bottleneck `64`) and one perturbation block. Single-file fitting uses learning
rate `3e-4` and dropout `0`, also with clipping `3.0`. Query chunk `4096` and readout chunk `16` bound activation memory.

Install torch with the appropriate CUDA build, then:

```powershell
python -m pip install -r requirements-training.txt
```

`transformers` and `rdkit` are needed only to build the chemical cache; they are
not required to train with an existing cache or to train the control model.
Set `TAHOE_DATA_ROOT` before launching, or use `--root`, for an external data path.

## Saved runs and resuming

```text
models/trained_model/
  model_YYYYMMDD_HHMMSS_microseconds_control/  # or _full
    info.json       # architecture, data plan, parameter counts, epoch scores, status
    config.json     # resolved training settings
    log.csv         # one row for every training backward pass
    last.pt         # resumable model + optimizer + progress + random states
    best.pt         # best held-out score, or reconstruction probe for fitting runs
```

Every new run creates a unique directory, including single-file runs. Resuming
continues in the same directory and appends to its CSV. The checkpoint restores
best score, gene vocabulary, selected experiments, position within an epoch,
optimizer and NumPy/PyTorch CPU/CUDA random states. The normalization setting is
also saved. Pre-normalization checkpoints resume with `normalization=none`;
start a new run to use LayerNorm rather than silently changing an old experiment.

`last.pt` is atomically replaced every 50 optimizer steps and at epoch boundaries.
A newly initialized model also gets an initial checkpoint. An interrupted run
replays work after the last saved boundary; CSV rows newer than that checkpoint
are removed before continuing, preventing duplicate backward-pass records.
If an epoch is incomplete, finishing it counts as the first additional epoch.
Older checkpoints from the previous pipeline are not accepted as resumable runs.

`log.csv` contains epoch, unit, optimizer step, backward-pass index, micro-batch,
accumulation count, file, plate, cell line, input/target sample, drug, dose, loss,
backward loss, gradient norm by module, total gradient norm, elapsed time and peak
allocated VRAM. No separate gradient CSV is created.

`loss` is the unaveraged example objective; `backward_loss` is `loss / number of
micro-batches in the unit`. Gradient columns measure that example's contribution
before clipping, with accumulation averaging undone. Prior accumulated gradients
are subtracted so each row describes its own backward pass. Gradient CSV fields
use eight digits after the decimal point (text formatting, not FP16/BF16 tensor storage); loss
and training/checkpoint precision are unchanged. For accumulation,
this requires a temporary copy of the preceding gradients.

## Non-interactive commands

```powershell
python models/train.py --model control --epochs 60
python models/train.py --model full --heads 8 --head-dim 128 --epochs 60
python models/train_small.py --model control --file plate6_CVCL_1125.h5ad --epochs 10
python models/train_small.py --model full --file plate6_CVCL_1125.h5ad --epochs 10
python models/train.py --resume models/trained_model/<run> --epochs 10
python models/train.py --model control --dry-run
```

On resume, `--epochs` means additional epochs. Clipping defaults to `3.0`;
`--grad-clip` can override it explicitly. Architecture overrides are rejected. Use `--help` for script-only settings, including sampling counts,
precision, memory chunks, save frequency and debug limits. `--max-steps` limits
optimizer updates per invocation; `--max-units-per-epoch` caps sampled units.
`--dry-run` builds a data plan and one example without creating a training run.

## Code organization and future changes

| File | Responsibility |
|---|---|
| `run.py` | Interactive menus and environment checks |
| `zipfolder.py` | Select and ZIP a complete run for transfer |
| `train.py`, `train_small.py` | Thin command-line entry points |
| `model.py` | Chemical perturbation architecture |
| `control_model.py` | Provided DMSO reconstruction architecture |
| `expression_features.py` | Shared five-feature summaries of pre-L2 log-expression |
| `tahoe_data.py` | Backed H5AD reading, well grouping, normalization, row sampling |
| `molformer_atoms.py` | Offline drug embedding cache builder/reader |
| `training/config.py` | Defaults, validation, CLI configuration |
| `training/data.py` | Eligible files, splits, epoch schedules, target selection |
| `training/modeling.py` | Model factory, chemical loading, forward/loss dispatch |
| `training/engine.py` | Shared optimization loop and evaluation |
| `training/metrics.py` | Module gradient metrics and one CSV writer |
| `training/checkpoints.py` | Run naming, atomic saves, metadata, RNG restoration |
| `benchmark.py`, `training/capacity.py` | Range sweep and isolated training-capacity worker |
| `model_flow.mmd` | Current full/control architecture and training flow |
| `tests/test_training.py` | Synthetic-data sampling, training and resume tests |

Change sampling in `training/data.py`, defaults in `training/config.py`, metrics
in `training/metrics.py`, or model integration in `training/modeling.py`. Both
entry points use the same engine. When changing persisted state or CSV schema,
bump the checkpoint schema and explicitly handle old-run compatibility.

Run checks without touching real data:

```powershell
python -m unittest discover -s models/tests -v
```

The supplied coordinates are expression-derived inputs. A reconstruction probe
checks fitting at known coordinates; it does not demonstrate prediction at
unknown coordinates or prove that the chemical branch is being used.

## Training capacity benchmark

```powershell
python models/benchmark.py
```

Enter the number of heads, per-head dimension start/end/step, and layer start/end,
then select a real H5AD file. Total width and bottleneck are derived for every shape.
For a scriptable sweep:

```powershell
python models/benchmark.py --file plate6_CVCL_1125.h5ad --heads 8 --head-dim-start 32 --head-dim-end 128 --head-dim-step 32 --layers-start 1 --layers-end 4
```

Both ranges use inclusive bounds (the end is tested if aligned with the step).
`--layers-step` defaults to 1. Heads default to 8 when prompted. For every
candidate, `dim = heads * head_dim` and `conc_dim = head_dim`; neither total dim
nor conc_dim is requested. There is no fixed cap on `head_dim`; candidates
are checked against actual runtime and GPU memory capacity.

Each shape runs in a fresh subprocess using actual cells, all genes in the file,
chemical tokens, the same chunk sizes and precision as training, and the full
forward/backward/gradient-metrics/clipping/AdamW path. By default one warmup unit
allocates optimizer state and two timed units are measured, each with up to two
micro-batches. This measures a training unit rather than inference. Weights are
discarded; it never saves a trained model or alters an existing run.

Reports go to `models/benchmark_results/benchmark_<timestamp>/results.csv` and
`summary.json`, updated after each completed shape. `--out` selects a new report
folder. The report includes parameters, loss, gradients, compute time, peak
allocated/reserved GPU memory, and a result for every pair:

- `fits`: completed with the requested free-VRAM headroom (default 10%).
- `tight`: completed but exceeds the memory budget; leave room before a long run.
- `oom`: CUDA allocation failed.
- `timeout`: the isolated process exceeded `--timeout` seconds (default 180),
  including startup and data reading; this does not prove memory failure.
- `nonfinite`, `invalid`, `error`: numerical, shape-validation or execution failure.
- `cpu_completed`: training worked on CPU; this is not a GPU-capacity verdict.

The summary reports the largest fitting per-head and total dimensions for each
layer count. Result rows include heads, head_dim, dim and conc_dim.
Use `--steps`, `--warmup`, `--timeout`, `--max-vram-fraction`, `--amp`,
`--query-chunk`, `--readout-chunk` and sampling flags to match your intended run.
A short benchmark on one file measures that workload; it is not a guarantee for
all drug lengths, background GPU usage or a complete long training run.
`--model control` is also supported, but layer counts have no effect there, so it
runs one trial per head dimension rather than repeating identical architectures.

## Targeted hidden-state normalization

New runs default to `normalization=layernorm` in both training modes and the
capacity benchmark. Current full models use these placements:

| Model | Placement | Normalized feature width |
|---|---|---|
| Both | After `merger`, before attention or readout | `dim` |
| Full | After chemical projection, once before the stack | `dim` |
| Full | Before each block's `q_linear` | `dim` |
| Full, optional | Before each block's `kqv`; disabled by default (`joint_normalization=False`) | `2 * dim` |
| Full | Separate projected Q/K/V norms before head splitting, both attention stages | `dim` each |
| Full | After chemical attention output + chemical FFN output | `dim` |
| Full | After gene FFN output + incoming gene residual | `dim` |
| Full | After the complete attention stack, before readout | `dim` |

LayerNorm uses learnable scale/bias and epsilon `1e-5`. It normalizes each token's
hidden features; it never normalizes across genes or cells. The chemical states
are normalized once because they are reused unchanged across attention blocks.
The chemical FFN residual is enabled for new full-model runs, as described above.
The cosine kernel, raw coordinates, and output expression scale are unchanged.
No extra `conc_ffn` normalization or coordinate encoder is introduced. Magnitude
statistics enter the merger as described below.

For an explicit new-run ablation, pass `--normalization none` to `train.py`,
`train_small.py` or `benchmark.py`. Saved metadata and benchmark results record
which architecture was used. Gradient columns include each trainable norm group.

Checkpoint schema 2 introduced the normalization setting; schema 3 also records
the gene-statistics setting. Schema-1 checkpoints
without that setting are interpreted as `none`, preserving their weights,
optimizer parameter groups and CSV schema on resume. Normalized checkpoints
resume with their own LayerNorm parameters and optimizer state. Normalization
cannot be changed during resume; use a fresh run for a different architecture.

## Five per-gene log-expression features

New full and control models append five features directly to `self.merger`:

1. Mean `log1p(CP10K)` expression across the sampled DMSO context cells.
2. Population standard deviation of those same log values (`correction=0`).
3. Detected fraction: number of positive log values divided by context-cell count.
4. Median of strictly positive log-expression values.
5. 90th percentile of strictly positive log-expression values.

The shared implementation is `expression_features.py`. It operates on the
existing `(B, G, C)` log-expression tensor **before** gene-wise L2 normalization.
It does not reconstruct counts, apply another logarithm, normalize the summaries
or fit a negative-binomial distribution. Reductions and quantile calculations
run in FP32 even inside mixed precision. Positive quantiles use linear
interpolation and one shared sort. For an all-zero gene, all five features are
zero; with one positive value, both positive quantiles equal that value.

The concatenation order is `[gene_embedding, coexpression_context, statistics]`,
where statistics follow the list above. The merger is now
`Linear(dim + conc_dim + 5, dim)`, followed by the existing merger LayerNorm.
No additional MLP is introduced. All summaries are invariant to control-cell
order and describe the same sampled DMSO population used by the cosine pathway.

The full training, single-file training and capacity benchmark use the same
feature implementation. Config/metadata record `gene_statistics` and feature
names. Use `--no-gene-stats` for an explicit new-run ablation.

Schema-1 and schema-2 checkpoints without the statistics setting resume with
`gene_statistics=False`; their merger weights, optimizer groups and original
architecture are preserved. New schema-3 checkpoints save the setting explicitly.
Start a new run to use five-feature fusion with a formerly statistics-free model.

## Bring a server run back to your PC

From the repository root:

```powershell
python zipfolder.py
```

Select a folder listed from `models/trained_model/`. The helper creates a ZIP in
`models/trained_model_zips/` containing that run's top-level folder, finalized
weights (including `last.pt` and `best.pt` when present), JSON metadata, CSV logs
and other finalized run files. It preserves the source folder, supports ZIP64,
checks ZIP integrity, and uses unique names so earlier exports are not replaced.
Temporary `.tmp`/`.partial` files are omitted. Changes to the selected run during
export cause the export to fail rather than publish an inconsistent archive.

`python models/zipfolder.py` also works. No torch, CUDA or third-party libraries
are required. For non-interactive use:

```powershell
python zipfolder.py --folder model_<timestamp>_control
python zipfolder.py --folder model_<timestamp>_full --out /scratch/exports
```

Transfer the resulting ZIP to your PC, extract it into `models/trained_model/`,
to keep its weights and metadata together. The ZIP contains the run only; the PC
still needs the code and any dataset/drug cache required to train. Continuing
training also requires the saved data paths to be available on the destination.

### Gene embedding residual

New runs in both architectures use `LayerNorm(merger(features) + gene_embeddings)`. This adds an identity gradient path without changing parameter counts. `--no-gene-residual` disables it for controlled comparisons. Schema 1–3 checkpoints keep the residual disabled when the setting is absent; resuming preserves their saved architecture. Schema 4 introduced this setting; current checkpoints use schema 10.


### Training mean-loss comparisons

Every backward-pass CSV row and tqdm postfix includes `loss`, `t_mean_loss`,
and `i_mean_loss`. The target baseline repeats the sampled target cells' mean
expression vector; the input baseline repeats the sampled DMSO cells' mean
expression vector. Means are computed per gene from log1p(CP10K) expression,
before the model's L2 normalization. Both use the model's squared-error loss
and `scale_exponent`, without accumulation averaging, and do not affect gradients
or checkpoint selection. The target mean is a target-derived constant reference.
Schema 7 adds these CSV columns. On resume, old committed rows receive blank
baseline fields; values cannot be reconstructed from historical scalar losses.

Current full models use `ffn_multiplier=2` for both FFNs and disable joint normalization before gene KQV by default. Schema 8 records both settings; older checkpoints infer them from saved tensor shapes and parameter keys. FFN width can be set with `--ffn-multiplier 4`. Query and joint normalization are bypassed in new runs; their settings are retained only for legacy checkpoints.


New full runs also use a direct chemical skip in every attention block:
`gene_ffn_norm(x + gene_ffn(gene_context) + sigmoid(chem_skip_gate_logit) * chem_context)`.
Each block learns its own scalar gate, initialized to 0.1 and bounded to [0, 1]
by sigmoid. Chemical context is taken after the chemical FFN residual and norm.
`--chem-skip-init` changes the initial value (strictly between 0 and 1);
`--no-chem-skip-gate` disables this new-run ablation. Schema 9 records both
settings and saves the learned logits as model parameters. Gate gradient norms
are included automatically in CSV parameter groups; effective gate values are
recorded per completed epoch as `chem_skip_gates` in `info.json`.
Checkpoints from schema 8 or earlier default to gates disabled, preserving
strict weight/optimizer loading and original forward behavior. Resuming such a
run does not enable gates: start a new gated run to use this architecture.


Schema 10 records `batch_size`, `checkpoint_layers`, and legacy-only
`accumulate_batches`. New runs default to B=2 and no accumulation; each item
has its own 512 control cells, 128 targets, chemistry, dose and coordinates.
Chemical tokens are padded with per-item masks; different drugs are preferred
within a batch when the selected file permits. Short final batches are allowed.
The interactive launcher asks B for both new and continued runs. CLI examples:
`python models/train.py --batch-size 4`, or
`python models/train.py --resume <run> --epochs 2 --batch-size 4`.
An explicit resumed B disables legacy accumulation and splits pending units if
needed; a pending old unit may be smaller than the requested B until the next
epoch. CLI resume without a B override preserves the saved update policy.
Older checkpoints default to B=1 with their original accumulation policy.

`log.csv` has one row per actual backward batch. `batch_size` is its actual
number of wells; `batch_samples` is JSON containing each well's drug, dose,
control/target sample, individual loss and both mean baselines. Top-level losses
are batch averages; gradients are those of the batch mean before clipping.
`backward_pass` counts actual backward calls. Epoch losses are weighted by
number of wells. New CSV columns are added on resume; old rows are B=1.
The dashboard retains batch metadata as text.

Full-model layer activation checkpointing defaults on to reduce B>1 VRAM:
layer activations are recomputed during backward with dropout RNG preserved.
`--no-checkpoint-layers` disables it for new runs. Old checkpoints retain their
saved behavior. The capacity benchmark uses the configured real forward B;
use `python models/benchmark.py --batch-size 2` to measure your exact GPU and
model configuration. Passing a small local test does not certify 48-GB fit.


### Attention normalization in new runs (schema 11)

New full-model blocks apply LayerNorm only to the chemical FFN update and
post-residual gene FFN update. Projected chemical/gene Q, K and V are not
normalized; the query-input norm is also removed. Parent-model normalization
is unchanged. Start a new run to use this architecture. Resuming schema 1–10
checkpoints preserves their original normalization and optimizer parameter order
through the compatibility-only `legacy_attention_norms` setting.

New full-model runs also pass the final block output directly to the coordinate
readout, without an additional final LayerNorm. Legacy checkpoint resumes retain
the historical final norm and its saved affine parameters.


### Eight-class sampling (schema 12)

New full all-file runs default to `sampling_strategy=class_rotation`.
The frozen mapping is `Chemcial_info/drug_cluster_labels_k8.json`; override with
`--chemical-classes PATH` for an explicitly prepared mapping. Unknown training
drugs fail indexing rather than being silently assigned a class. The mapping and
its hash are stored in the plan. No clustering runs during training or resume.
The approved 367-drug membership is preserved; other library drugs are assigned
to the nearest frozen centroid for alternate training splits.

For the current split, B=2 means 550 updates and 1,100 treatment presentations
per epoch, with every available class in each file covered within four complete
epochs. Files with missing classes use only their available classes; one-class
files may repeat a class within a batch. Wells remain unique within a batch.
Short files yield short batches. B stays configurable; larger batches rotate
through available classes before repeating a class. Control wells cycle when
B exceeds their count. Control cells are never pooled across lines or plates.

Class/drug/well rotation counts are saved in `progress.sampler_state`, together
with the planned epoch units and their execution cursor. Counts describe planned
exposure, including pending units; completed exposure can be read from the CSV.
Each `batch_samples` entry includes its chemical class and actual control well.
Existing schema 1–11 resumes retain their original sampler. Single-file and
control-model schedules remain unchanged. `passes_per_cell_line` and
`batches_per_unit` no longer set the new full training epoch size; the latter
still determines the fixed validation well count. Validation inputs are unchanged.
