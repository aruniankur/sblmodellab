# Running the Tahoe-100M repository

Run commands from the repository root. Linux/server examples use Bash; Windows setup is shown separately. Replace example paths, usernames, filenames and run names with your own.

## 1. What this repository runs

| Model | Context cells | Target cells | Additional inputs |
|---|---|---|---|
| `full` | DMSO | Chemically treated | Chemical token embeddings, concentration, target coordinates |
| `control` | DMSO | DMSO | Target coordinates; no chemical cache or perturbation attention stack |

Both models support all-file training and single-file training. Context and target cells always come from the same plate and cell line. Cross-well sampling and overlapping context/target cells are allowed. Training uses plates 1-12.

This guide starts with already prepared H5AD files; the training launcher does not download Tahoe-100M or generate coordinates.

## 2. Create the training environment

### Linux / server

Python 3.11 is a reasonable environment choice. Create and activate an environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install a CUDA-enabled PyTorch build compatible with your GPU and NVIDIA driver, using the command from the [official PyTorch installation selector](https://pytorch.org/get-started/locally/). Then install the repository requirements:

```bash
python -m pip install -r requirements-training.txt
```

The repository was previously validated with PyTorch 2.6.0 + CUDA 12.4; that is a tested environment, not a requirement to use that CUDA build on every server.

### Windows / PowerShell

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Install the appropriate CUDA-enabled PyTorch build, then:

```powershell
python -m pip install -r requirements-training.txt
```

If PowerShell activation is unavailable, invoke `.venv\Scripts\python.exe` directly instead of `python` in subsequent commands.

### Verify the GPU

```bash
nvidia-smi
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'); print('BF16:', torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False)"
```

The interactive launcher checks required imports and GPU availability. It selects BF16 when supported and FP32 otherwise. For direct training commands, explicitly use `--amp off` if BF16 is unsupported. The default direct-training precision is BF16.

The requirements file includes `transformers` and `rdkit` for chemical-cache construction. Training with an existing cache, and control training, do not load the MolFormer encoder.

## 3. Prepare data paths

Default layout:

```text
data_by_plate_cell_line/
  plate1_CVCL_....h5ad
  plate2_CVCL_....h5ad
  ...
Chemcial_info/
  molformer_embeddings.csv
```

`Chemcial_info` is the existing directory spelling used by the code.

Each H5AD file must have CSR raw counts in `X`, matching gene IDs/order in `var_names`, and these `obs` columns:

```text
sample, drug, canonical_smiles, concentration, coord_1, coord_2
```

Use filenames `plate{N}_{CELL_LINE}.h5ad`. Cells with nonfinite coordinates are excluded. Files without usable DMSO context or eligible targets are skipped; the saved plan records the actual selected data.

The loader normalizes sampled raw-count rows to 10,000 counts per cell and applies `log1p`. Supply raw counts, not already log-transformed expression.

For an external data directory, set the environment variable **before launching Python**:

```bash
export TAHOE_DATA_ROOT=/path/to/data_by_plate_cell_line
```

PowerShell equivalent:

```powershell
$env:TAHOE_DATA_ROOT = 'D:\data\data_by_plate_cell_line'
```

Direct training and benchmarking also accept `--root /path/to/data_by_plate_cell_line`.

## 4. Build the chemical cache once (full model only)

Skip this section for control training. If a valid cache already exists, reuse it.

The default input CSV is `Chemcial_info/molformer_embeddings.csv` and must include `drug` and `smiles` columns. Drug identifiers must match the dataset. Build the cache:

```bash
python models/molformer_atoms.py
```

Or supply explicit paths:

```bash
python models/molformer_atoms.py --csv /path/to/drugs.csv --out models/cache/atom_embeddings.pt
```

The first build needs access to download the pretrained MolFormer model. The cache contains frozen token embeddings; training reads it rather than rerunning MolFormer. Full training uses all real SMILES tokens by default, with special-token rows excluded. Atom-only mode is an optional ablation.

Default cache location:

```text
models/cache/atom_embeddings.pt
```

Direct full training and benchmarking accept `--atom-cache /path/to/atom_embeddings.pt`. For the interactive launcher, place the cache at its default location.

## 5. Start training interactively

```bash
python run.py
```

The launcher:

1. Checks required libraries and reports GPU availability.
2. Asks whether to start a new model or continue a saved run.
3. For a new model, asks full/control and all files/one file.
4. Asks for number of heads, dimension per head, layers and epochs.
5. Derives total dimension and concentration/co-expression width automatically.

```text
dim = heads * head_dim
conc_dim = head_dim
```

Gene count comes from the data; chemical feature width comes from the cache. There is no fixed cap on dimension per head; total dimension must be divisible by the number of heads. In the control model, heads and head dimension set widths, but the layer-count and dropout settings do not create an attention stack.

For an existing run, enter how many **additional** epochs to train.

## 6. Full-model starting configuration for a 48 GB GPU

This is a suggested starting candidate, not a measured memory guarantee for your server:

| Setting | Value |
|---|---:|
| Heads | 8 |
| Dimension per head | 128 |
| Derived total dimension | 1024 |
| Derived `conc_dim` | 128 |
| Perturbation layers | 4 |
| Initial epochs | 10 |
| DMSO context cells | 256 |
| Target cells | 128 |
| Learning rate | 0.0001 |
| Gradient clipping | 3 |
| Precision | BF16, when supported |
| Gene embedding residual | Enabled |

First benchmark the intended shape; select a file with eligible treated wells when prompted:

```bash
python models/benchmark.py --model full --heads 8 --head-dim-start 128 --head-dim-end 128 --head-dim-step 32 --layers-start 4 --layers-end 4 --steps 3 --timeout 900
```

If it fits with memory headroom, train:

```bash
python models/train.py --model full --heads 8 --head-dim 128 --layers 4 --epochs 10 --amp bf16
```

Unlike the control architecture, the full model actually uses the requested perturbation layers. More layers or VRAM usage do not guarantee better learning. Ten epochs is an initial assessment run, not a promise of convergence.

## 7. Other training commands

### Full model across eligible files

```bash
python models/train.py --model full --heads 8 --head-dim 64 --layers 2 --epochs 10
```

### Control model across eligible files

```bash
python models/train.py --model control --heads 8 --head-dim 100 --epochs 10
```

### Train on one file

These example filenames must exist in your data directory:

```bash
python models/train_small.py --model full --file plate6_CVCL_1125.h5ad --heads 8 --head-dim 128 --layers 4 --epochs 10
python models/train_small.py --model control --file plate6_CVCL_1125.h5ad --heads 8 --head-dim 100 --epochs 10
```

Omit `--file` to select interactively. Single-file training defaults to learning rate `0.0003` and dropout `0`; all-file training defaults to `0.0001` and dropout `0.1`. Both use clipping at 3.

### Inspect data without training

```bash
python models/train.py --model control --dry-run
python models/train.py --model full --dry-run
```

A dry run checks the data plan and example tensor shapes without creating a training run. It is not a GPU-capacity test; use the benchmark for real forward/backward updates.

### Short pipeline smoke test

```bash
python models/train.py --model full --heads 8 --head-dim 64 --layers 1 --epochs 1 --max-steps 2
```

This creates a real saved run, but limits optimizer updates for that invocation.

### Run on a Linux server after disconnecting

For interactive training, use a terminal multiplexer if available:

```bash
tmux new -s tahoe
source .venv/bin/activate
python run.py
```

Detach with `Ctrl+B`, then `D`. Reattach later with:

```bash
tmux attach -t tahoe
```

Set `TAHOE_DATA_ROOT` inside the session if needed. Keep the dashboard in another terminal/session.

## 8. Resume training

```bash
python models/train.py --resume models/trained_model/YOUR_RUN_FOLDER --epochs 10
```

Here `10` means ten additional epochs. If the saved epoch is incomplete, completing it counts as the first additional epoch. Resume restores model, optimizer, random states and the sampling cursor, and writes into the same run folder.

Resume accepts only the saved run, additional epochs, and optionally a clipping override:

```bash
python models/train.py --resume models/trained_model/YOUR_RUN_FOLDER --epochs 10 --grad-clip 3
```

Do not pass architecture, data-path or sampling overrides with `--resume`; they are rejected. Resume from the run directory or `last.pt`, not `best.pt`.

Saved data paths must remain accessible. Moving a run between machines does not automatically remap its saved absolute dataset paths.

New runs enable LayerNorm, five log-expression statistics and the gene residual:

```text
gene_states = LayerNorm(merger(features) + gene_embeddings)
```

Current checkpoints use schema 4. Older supported checkpoints preserve their original architecture when settings are absent, including a disabled gene residual. Starting a new run uses the current defaults; resuming does not silently upgrade an older architecture.

## 9. Understand saved files and progress

```text
models/trained_model/
  model_YYYYMMDD_HHMMSS_microseconds_full/    # or _control
    config.json
    info.json
    log.csv
    last.pt
    best.pt
```

| File | Contents |
|---|---|
| `config.json` | Resolved training settings |
| `info.json` | Status, data plan, architecture, parameter counts and epoch scores |
| `log.csv` | One row per backward pass: loss, gradients, sample metadata, timing and memory |
| `last.pt` | Latest resumable model, optimizer, progress and random states |
| `best.pt` | Model selected by the best saved evaluation score |

Checkpoints are saved initially, every 50 optimizer updates by default, and at epoch boundaries. After an interruption, resume discards CSV rows beyond the committed checkpoint and replays that unsaved work.

Default all-file sampling uses five available plates per training cell line per epoch and up to two target-well backward passes per optimizer update. Single-file mode visits each eligible target well per epoch.

Full all-file training holds out two eligible cell lines and evaluates their first two available plates. Control and single-file runs use fixed reconstruction probes from training files. Those probes measure fitting, not held-out generalization.

The loss is squared error summed over genes, averaged over cells/examples and divided by `num_genes ** 0.5`. Compare runs only when the loss definition, gene panel and evaluation samples are compatible.

Gradient columns use eight decimal places and report each backward pass's contribution **before clipping**, with accumulation averaging undone. They are not the actual accumulated optimizer-step norm, so counting rows above 3 does not give the clipping rate. CSV formatting does not change tensor precision. There is no TensorBoard dependency; use tqdm and the dashboard.

## 10. Live dashboard in a second terminal

The dashboard is a separate, read-only Streamlit application. It reads saved JSON and CSV files and does not start or modify training.

### Linux / server setup

From the repository root in a second terminal:

```bash
python3 -m venv UIeval/.venv
source UIeval/.venv/bin/activate
python -m pip install -r UIeval/requirements.txt
python UIeval/run.py --port 8501
```

Default listening address is `0.0.0.0`. Open `http://SERVER_IP:8501` on your PC if the server port is reachable. The launcher prints browser addresses; do not enter `0.0.0.0` as the remote browser destination.

### SSH tunnel option

Start the dashboard bound to loopback on the server:

```bash
python UIeval/run.py --host 127.0.0.1 --port 8501
```

On your PC, in a separate terminal:

```bash
ssh -N -L 8501:127.0.0.1:8501 USER@SERVER_IP
```

Then open `http://localhost:8501`. The dashboard has no built-in login; the tunnel avoids exposing its port publicly.

### Windows dashboard setup

```powershell
python -m venv UIeval/.venv
UIeval\.venv\Scripts\python.exe -m pip install -r UIeval/requirements.txt
UIeval\.venv\Scripts\python.exe UIeval/run.py --port 8501
```

Select a run in the sidebar. Live updates default to every two seconds and can be set to 1-10 seconds. Use **Rescan runs** to discover newly created run folders. The dashboard supports loss and gradient plots, epoch scores, filters and run comparisons. A custom run root can be supplied with `--runs-root /path/to/trained_model`.

## 11. Benchmark dimension and depth ranges

Interactive:

```bash
python models/benchmark.py
```

Scripted sweep; replace the example file with an existing one:

```bash
python models/benchmark.py --model full --file plate6_CVCL_1125.h5ad --heads 8 --head-dim-start 64 --head-dim-end 128 --head-dim-step 32 --layers-start 2 --layers-end 6 --layers-step 2 --steps 3 --timeout 900
```

Each candidate runs isolated real training updates, including optimizer state allocation. Benchmark weights are discarded. Reports are written under `models/benchmark_results/` as `results.csv` and `summary.json`.

- `fits`: completed within the requested memory budget.
- `tight`: completed but left insufficient headroom.
- `oom`: allocation failed.
- `timeout`: exceeded the time limit; this does not establish an out-of-memory failure.
- Other statuses describe invalid configurations, nonfinite values, execution failures or CPU-only completion.

The default memory budget is 90% of initially available VRAM. A successful trial on one chemical/file does not guarantee every full training workload fits. Control-model layers are inactive, so its benchmark does not repeat equivalent depths.

## 12. Export a run to your PC

After training finishes:

```bash
python zipfolder.py
```

Select the run folder. ZIP files are created in `models/trained_model_zips/`; source files are preserved. The archive includes the run folder, weights, JSON and logs, with an integrity check.

Non-interactive export:

```bash
python zipfolder.py --folder YOUR_RUN_FOLDER
```

Transfer from your PC, using the ZIP filename printed by the export command:

```bash
scp USER@SERVER_IP:/path/to/repo/models/trained_model_zips/YOUR_EXPORT.zip .
```

Extract the archive directly into `models/trained_model/`, keeping the enclosed run folder intact. The archive does not include repository code, the dataset or chemical cache. Continuing training on a new machine also requires accessible saved dataset paths.

## 13. Troubleshooting and useful options

| Problem | What to check |
|---|---|
| CUDA unavailable | Correct activated environment, CUDA-enabled PyTorch build and server driver/GPU allocation |
| BF16 unsupported | Use the interactive launcher or pass `--amp off` to direct training |
| Missing chemical cache | Build it once or supply `--atom-cache` to direct full training |
| Missing drug identifier | Ensure the chemical table/cache covers the dataset's drug identifiers |
| No usable training data | Root path, plate 1-12 filenames, DMSO wells, finite coordinates and minimum eligible cell counts |
| Gene identity/order mismatch | All selected files must use the same gene vocabulary in the same order |
| Out of memory | Benchmark fewer layers or smaller head dimensions; try smaller query/readout chunks |
| Dashboard unreachable | Use its printed server address or an SSH tunnel; confirm the chosen port |
| Resume fails after transfer | Preserve saved files and make saved dataset paths accessible |

Memory-tuning example for a **new** run:

```bash
python models/train.py --model full --heads 8 --head-dim 128 --layers 4 --epochs 10 --query-chunk 1024 --readout-chunk 8
```

These chunks bound cosine-context and readout memory. They do not chunk the full model's gene self-attention stack, so they cannot solve every memory bottleneck. Match them in the benchmark if you change them.

Useful defaults: weight decay `0.01`, gradient clipping `3`, scale exponent `0.5`, query chunk `4096`, readout chunk `16`, and seed `0`.

For new-run architecture ablations, training accepts `--no-gene-residual`, `--no-gene-stats` and `--normalization none`. All are opt-in departures from the current defaults. Do not use these flags with resume.

## 14. Checks and code reference

Run the model/training/benchmark tests:

```bash
python -m unittest discover -s models/tests -v
```

Run dashboard tests in an environment with its requirements installed:

```bash
python -m unittest discover -s UIeval/tests -v
```

Command help:

```bash
python models/train.py --help
python models/train_small.py --help
python models/benchmark.py --help
python models/molformer_atoms.py --help
python UIeval/run.py --help
python zipfolder.py --help
```

Core code lives under `models/`; the dashboard is under `UIeval/`. See `models/README_training.md` for implementation details and `models/model_flow.mmd` for the architecture diagram. There is no standalone inference CLI documented here; training checkpoints require the matching model architecture and input preparation for custom prediction code.
