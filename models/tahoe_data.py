"""Per-well data pipeline for the Tahoe-100M virtual-cell model.

Layout of the dataset (verified against the files on disk):

* ``data_by_plate_cell_line/plate{N}_{CELL_LINE}.h5ad`` -- **700 files** =
  complete grid of **14 plates x 50 cell lines**.
* every file is **one (plate, cell line)**: ~96 wells = ~95 drugs + **exactly 2
  DMSO_TF wells**. One drug per well, one dose per well.
* ``var`` is the same **50,266 Ensembl genes in the same order in every file**.
* ``X`` is a CSR group of **raw integer counts** (``X/data``, ``X/indices``,
  ``X/indptr``) -> ``normalize_total(1e4)`` + ``log1p`` must be applied here.
* ``obs`` carries ``sample``, ``drug``, ``concentration`` (uM), ``coord_1``,
  ``coord_2`` (UMAP position; ~20 NaNs per file, those cells are dropped).
* dose is a *plate* property: plates 1/4/7/10 -> 0.05, 2/5/8/11 -> 0.5,
  3/6/9/12/14 -> 5.0, plate 13 is the 44-drug x 3-dose response plate.

One training example is a **well**:

    control : n_control cells from a DMSO well of the file
    target  : n_target cells from one drug well of the same file + their coords
    chem    : the drug's per-atom MolFormer vectors + log10(dose + 1e-3)

Nothing here touches a model, so ``--dry-run`` can exercise it end to end.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Iterable, Sequence

import anndata as ad
import numpy as np
import scipy.sparse as sp
import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The h5ad files are not in git (593 GB). Put them here, or point the trainer
# elsewhere with `export TAHOE_DATA_ROOT=/scratch/tahoe/data_by_plate_cell_line`.
DEFAULT_ROOT = os.environ.get(
    "TAHOE_DATA_ROOT", os.path.join(_REPO_ROOT, "data_by_plate_cell_line")
)

CONTROL_DRUG = "DMSO_TF"
CONTROL_DRUG_ALIASES = {"DMSO_TF", "DMSO", "DMSO_TF "}

_FILE_RE = re.compile(r"^plate(?P<plate>\d+)_(?P<cell_line>.+)\.h5ad$")


# --------------------------------------------------------------------------- #
# file index
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PlateFile:
    path: str
    plate: int
    cell_line: str


def build_file_index(
    root: str = DEFAULT_ROOT,
    exclude_plates: Sequence[int] = (13, 14),
) -> list[PlateFile]:
    """Index every ``plate{N}_{CELL_LINE}.h5ad`` under ``root``.

    ``exclude_plates`` is enforced by omission, so an excluded plate can never be
    opened by the trainer (plates 13 and 14 are held out by design).
    """
    excluded = {int(p) for p in exclude_plates}
    out: list[PlateFile] = []
    for name in sorted(os.listdir(root)):
        m = _FILE_RE.match(name)
        if not m:
            continue
        plate = int(m.group("plate"))
        if plate in excluded:
            continue
        out.append(PlateFile(os.path.join(root, name), plate, m.group("cell_line")))
    if not out:
        raise FileNotFoundError(f"no plate{N}_*.h5ad files found in {root}")
    return out


def grid(index: Iterable[PlateFile]) -> tuple[list[int], list[str]]:
    """Sorted plates and cell lines present in ``index``."""
    plates = sorted({f.plate for f in index})
    cell_lines = sorted({f.cell_line for f in index})
    return plates, cell_lines


# --------------------------------------------------------------------------- #
# normalisation
# --------------------------------------------------------------------------- #
def normalize_log1p(dense: np.ndarray, target_sum: float = 1e4) -> np.ndarray:
    """``sc.pp.normalize_total(target_sum)`` followed by ``sc.pp.log1p``.

    Same recipe the reference notebook applies to the whole AnnData, but per
    batch so the 19.6 TB dense copy is never built. Empty cells stay all-zero.
    """
    dense = np.asarray(dense, dtype=np.float32)
    library = dense.sum(axis=1, keepdims=True)
    scale = np.divide(
        target_sum, library, out=np.zeros_like(library), where=library > 0
    )
    return np.log1p(dense * scale)


# --------------------------------------------------------------------------- #
# backed file
# --------------------------------------------------------------------------- #
@dataclass
class Well:
    sample: str
    drug: str
    concentration: float
    smiles: str
    rows: np.ndarray  # obs row indices of the usable cells (finite coords)

    @property
    def n_cells(self) -> int:
        return int(self.rows.size)


class BackedFile:
    """One ``plate x cell line`` file: obs table + random-access CSR row reader.

    ``anndata`` is only used for ``obs``; ``X`` is read with ``h5py`` directly so
    reading 384 of 100k rows costs two small fancy-index reads instead of a
    whole-matrix load.
    """

    def __init__(self, path: str, min_cells: int = 1):
        self.path = path
        self._adata = ad.read_h5ad(path, backed="r")
        self.n_obs, self.n_vars = self._adata.shape

        obs = self._adata.obs
        drug = obs["drug"].astype(str).to_numpy()
        sample = obs["sample"].astype(str).to_numpy()
        smiles = obs["canonical_smiles"].astype(str).to_numpy()
        conc = obs["concentration"].to_numpy(np.float64)
        coords = obs[["coord_1", "coord_2"]].to_numpy(np.float32)

        self.coords = coords
        finite = np.isfinite(coords).all(axis=1)

        codes, uniques = _factorize(sample)
        order = np.argsort(codes, kind="stable")
        sorted_codes = codes[order]
        bounds = np.searchsorted(sorted_codes, np.arange(len(uniques) + 1))

        wells: list[Well] = []
        for k, name in enumerate(uniques):
            rows = order[bounds[k] : bounds[k + 1]]
            rows = rows[finite[rows]]
            if rows.size == 0:
                continue
            first = rows[0]
            wells.append(
                Well(
                    sample=str(name),
                    drug=str(drug[first]),
                    concentration=float(conc[first]),
                    smiles=str(smiles[first]),
                    rows=rows,
                )
            )

        self.wells = wells
        self.control_wells = [w for w in wells if w.drug in CONTROL_DRUG_ALIASES]
        self._min_cells = min_cells
        self._drug_wells = [
            w for w in wells if w.drug not in CONTROL_DRUG_ALIASES and w.n_cells >= min_cells
        ]

        x = self._adata.file["X"]
        self._data = x["data"]
        self._indices = x["indices"]
        self._indptr = x["indptr"][:].astype(np.int64)

    # -- wells ------------------------------------------------------------- #
    @property
    def plate(self) -> int:
        return int(re.match(_FILE_RE, os.path.basename(self.path)).group("plate"))

    @property
    def cell_line(self) -> str:
        return _FILE_RE.match(os.path.basename(self.path)).group("cell_line")

    def drug_wells(self, min_cells: int | None = None) -> list[Well]:
        if min_cells is None or min_cells <= self._min_cells:
            return self._drug_wells
        return [w for w in self._drug_wells if w.n_cells >= min_cells]

    # -- rows -------------------------------------------------------------- #
    def read_rows(self, rows: np.ndarray) -> np.ndarray:
        """Dense ``float32 (len(rows), n_vars)`` for arbitrary (unsorted, dup) rows."""
        rows = np.asarray(rows, dtype=np.int64)
        order = np.argsort(rows, kind="stable")
        sorted_rows = rows[order]

        starts = self._indptr[sorted_rows]
        counts = self._indptr[sorted_rows + 1] - starts
        total = int(counts.sum())

        if total:
            # positions = start_i + arange(count_i), vectorised
            offsets = np.repeat(starts - np.concatenate(([0], np.cumsum(counts)[:-1])), counts)
            positions = offsets + np.arange(total, dtype=np.int64)
            values = self._data[positions]
            columns = self._indices[positions]
        else:
            values = np.zeros(0, dtype=np.float32)
            columns = np.zeros(0, dtype=np.int32)

        indptr = np.concatenate(([0], np.cumsum(counts))).astype(np.int64)
        matrix = sp.csr_matrix(
            (values, columns, indptr), shape=(rows.size, self.n_vars)
        )
        dense = matrix.toarray().astype(np.float32, copy=False)

        inverse = np.empty_like(order)
        inverse[order] = np.arange(order.size)
        return dense[inverse]

    def close(self) -> None:
        try:
            self._adata.file.close()
        except Exception:
            pass


def _factorize(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``pd.factorize(sort=False)`` without importing pandas for one call."""
    import pandas as pd

    codes, uniques = pd.factorize(values, sort=False)
    return codes, np.asarray(uniques)


class FileCache:
    """Small LRU of :class:`BackedFile` so a plate group stays resident."""

    def __init__(self, maxsize: int = 5, min_cells: int = 1):
        self.maxsize = max(1, int(maxsize))
        self.min_cells = int(min_cells)
        self._open: dict[str, BackedFile] = {}

    def get(self, path: str) -> BackedFile:
        if path in self._open:
            return self._open[path]  # already resident
        while len(self._open) >= self.maxsize:
            oldest_path = next(iter(self._open))  # insertion order; plates are visited once
            oldest = self._open.pop(oldest_path)
            oldest.close()
        backed = BackedFile(path, min_cells=self.min_cells)
        self._open[path] = backed
        return backed

    def close(self) -> None:
        for f in self._open.values():
            f.close()
        self._open.clear()


# --------------------------------------------------------------------------- #
# example construction
# --------------------------------------------------------------------------- #
@dataclass
class Example:
    control: torch.Tensor  # (G, n_control) float32
    target: torch.Tensor  # (n_target, G) float32
    coords: torch.Tensor  # (n_target, 2)  float32
    drug: str
    concentration: float


def _pick(rows: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """``n`` rows; with replacement only when the well is too small."""
    replace = rows.size < n
    return rng.choice(rows, size=n, replace=replace)


def build_example(
    backed: BackedFile,
    control_well: Well,
    drug_well: Well,
    n_control: int,
    n_target: int,
    rng: np.random.Generator,
) -> Example:
    control_rows = _pick(control_well.rows, n_control, rng)
    target_rows = _pick(drug_well.rows, n_target, rng)

    # one read for both wells: fewer h5py trips than two separate calls
    dense = normalize_log1p(backed.read_rows(np.concatenate([control_rows, target_rows])))

    control = torch.from_numpy(np.ascontiguousarray(dense[:n_control].T))
    target = torch.from_numpy(np.ascontiguousarray(dense[n_control:]))
    coords = torch.from_numpy(np.ascontiguousarray(backed.coords[target_rows]))

    return Example(
        control=control,
        target=target,
        coords=coords,
        drug=drug_well.drug,
        concentration=drug_well.concentration,
    )


if __name__ == "__main__":  # tiny self-check
    index = build_file_index()
    plates, cell_lines = grid(index)
    print(f"files={len(index)} plates={plates} cell_lines={len(cell_lines)}")

    backed = BackedFile(index[0].path)
    print(f"{os.path.basename(index[0].path)}: {backed.n_obs} cells, {backed.n_vars} genes")
    print(f"  wells={len(backed.wells)} control={len(backed.control_wells)} "
          f"drug(>=128 cells)={len(backed.drug_wells(128))}")
    cw = backed.control_wells[0]
    dw = backed.drug_wells(128)[0]
    rng = np.random.default_rng(0)
    ex = build_example(backed, cw, dw, 256, 128, rng)
    print(f"  control={tuple(ex.control.shape)} target={tuple(ex.target.shape)} "
          f"coords={tuple(ex.coords.shape)} drug={ex.drug} dose={ex.concentration}")
    print(f"  control range [{ex.control.min():.3f}, {ex.control.max():.3f}] "
          f"target range [{ex.target.min():.3f}, {ex.target.max():.3f}]")
    backed.close()
