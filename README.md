# Tahoe-100M — Local Working Copy

A local working copy of the [Tahoe-100M](https://huggingface.co/datasets/tahoebio/Tahoe-100M) giga-scale single-cell
perturbation atlas (100M+ transcriptomic profiles, 50 cancer cell lines, ~1,100 perturbations), **repartitioned into
one parquet file per sample** for easy per-sample access.

- **Expression data** → `data_by_sample/` (1,344 per-sample parquet files + `sample_index.parquet` manifest)
- **Metadata, gene vocabulary, cell-count summary, pseudobulk DE** → `metadata/`
- **Drug chemistry (MolFormer embeddings + clusters)** → `Chemcial_info/`
- **Per-cell UMAP coordinates** → `data_by_sample_embedded/` (same filenames + `coord_1`/`coord_2`; see [run_instruction.md](run_instruction.md))
- **AnnData loader** → `utils/load_utils.py`
- **Docs & how-to** → [docs/](docs/README.md) — start there

## Quick example

```python
import pandas as pd

idx = pd.read_parquet("data_by_sample/sample_index.parquet")   # lightweight manifest
sample = "smp_1495"
df = pd.read_parquet(f"data_by_sample/{sample}.parquet")       # all cells of one sample
emb = pd.read_parquet(f"data_by_sample_embedded/{sample}.parquet")  # + coord_1/coord_2
```

## Embedding progress

Per-cell-line UMAP coordinates live in `data_by_sample_embedded/` (648 of 1,344
samples as of 15 Nov 2026). The remaining work is split into ten groups in
[`groups.json`](groups.json); run one group with:

```powershell
python utils/embed_all.py --group g1 --workers 3
```

To run a group on another machine, package it first (~18 GB instead of the whole
dataset):

```powershell
python utils/make_group_bundle.py --group g1 --dest D:\tahoe_g1
```

Install with `pip install -r requirements.txt` (see
[run_instruction.md](run_instruction.md) for the torch CUDA wheel and the
3–4 machine workflow).

## License
CC0 1.0 (the upstream `LICENSE.md` is no longer present in this working copy). Dataset from the Tahoe-100M
consortium; see [docs/huggingfacereadme.md](docs/huggingfacereadme.md) for the upstream card and
[`citation`](docs/huggingfacereadme.md#citation) info.

---
**Read the docs:** [docs/README.md](docs/README.md) covers the layout, each folder, and gotchas (including the fact
that the upstream `data/` folder no longer exists and `tutorials/loading_data.ipynb` still targets HuggingFace).
## Model training

Run `python run.py` for the interactive full/control model launcher. It supports
new and resumed runs, all-file or single-file training, tqdm progress, and one
loss/gradient CSV per run. Checkpoints and metadata live under
`models/trained_model/model_<timestamp>_{control,full}/`.
See [models/README_training.md](models/README_training.md) for setup and code layout.
