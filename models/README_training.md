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
5. Enter only `dim`, `conc_dim`, `heads`, `layers` and `epochs`. Gene count and
   chemical feature width come from the dataset and MolFormer cache.
6. For an existing run, choose the run and enter the number of additional epochs.
   Saved model, optimizer, data selection and training settings are restored.

`heads`, `layers` and `dropout` have no computational effect in the provided
control architecture: it has no perturbation-attention stack. These settings are
recorded for compatibility, and the launcher explains this before prompting.

## Two models, two training modes

| Model | Context | Target | Extra input |
|---|---|---|---|
| `full` | DMSO cells | Drug-treated cells | MolFormer tokens + dose, treated-cell coordinates |
| `control` | DMSO cells | DMSO cells | DMSO target-cell coordinates; no chemicals |

For both models, context and target always belong to the **same plate and cell
line**. Different DMSO wells and overlapping cells are allowed. All training is
restricted to plates **1-12**, including single-file training.

- **All-file mode:** sample five distinct available plates per training cell line
  per epoch. Up to two target wells per file contribute separate backward passes
  followed by one averaged-gradient optimizer update.
- **Single-file mode:** visit every eligible target well in the chosen file once
  per epoch, in random order, with one backward pass and update per well.
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
256 context cells and 128 target cells, with a target-well eligibility threshold
of 128 cells. Sampling uses replacement when a context or target pool is smaller
than its requested sample count. Duplicate sampled rows are read once and then
expanded back into the requested order.

The full model reads `models/cache/atom_embeddings.pt`. Build it once with:

```powershell
python models/molformer_atoms.py
```

The control path never loads that cache. Both models retain the supplied forward
architecture. The objective is squared error summed over genes, averaged over
cells and examples, and divided by `n_genes ** 0.5`.

Full-grid defaults: AdamW, learning rate `1e-4`, weight decay `0.01`, gradient
clipping `1.0`, dropout `0.1`, width `512`, bottleneck `32`, eight heads and one
perturbation block. Single-file fitting uses learning rate `3e-4`, clipping `10`
and dropout `0`. Query chunk `4096` and readout chunk `16` bound activation memory.

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
optimizer and NumPy/PyTorch CPU/CUDA random states.

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
are subtracted so each row describes its own backward pass. For accumulation,
this requires a temporary copy of the preceding gradients.

## Non-interactive commands

```powershell
python models/train.py --model control --epochs 60
python models/train.py --model full --epochs 60
python models/train_small.py --model control --file plate6_CVCL_1125.h5ad --epochs 10
python models/train_small.py --model full --file plate6_CVCL_1125.h5ad --epochs 10
python models/train.py --resume models/trained_model/<run> --epochs 10
python models/train.py --model control --dry-run
```

On resume, `--epochs` means additional epochs and other configuration overrides
are rejected. Use `--help` for script-only settings, including sampling counts,
precision, memory chunks, save frequency and debug limits. `--max-steps` limits
optimizer updates per invocation; `--max-units-per-epoch` caps sampled units.
`--dry-run` builds a data plan and one example without creating a training run.

## Code organization and future changes

| File | Responsibility |
|---|---|
| `run.py` | Interactive menus and environment checks |
| `train.py`, `train_small.py` | Thin command-line entry points |
| `model.py` | Chemical perturbation architecture |
| `control_model.py` | Provided DMSO reconstruction architecture |
| `tahoe_data.py` | Backed H5AD reading, well grouping, normalization, row sampling |
| `molformer_atoms.py` | Offline drug embedding cache builder/reader |
| `training/config.py` | Defaults, validation, CLI configuration |
| `training/data.py` | Eligible files, splits, epoch schedules, target selection |
| `training/modeling.py` | Model factory, chemical loading, forward/loss dispatch |
| `training/engine.py` | Shared optimization loop and evaluation |
| `training/metrics.py` | Module gradient metrics and one CSV writer |
| `training/checkpoints.py` | Run naming, atomic saves, metadata, RNG restoration |
| `benchmark.py` | Independent one-file hardware capability probe; no model saved |
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
