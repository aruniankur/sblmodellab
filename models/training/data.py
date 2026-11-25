"""Experiment indexing and sampling. One example always belongs to one file."""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from models.tahoe_data import BackedFile, build_example, build_file_index


@dataclass
class DataPlan:
    files: list[dict]
    train_lines: list[str]
    val_lines: list[str]
    n_genes: int
    gene_names: list[str]

    def to_dict(self):
        return vars(self)

    @classmethod
    def from_dict(cls, value):
        return cls(**value)


def targets(backed, cfg):
    if cfg.model_type == "control":
        return [well for well in backed.control_wells if well.n_cells >= cfg.min_cells]
    return backed.drug_wells(cfg.min_cells)


def coordinate_status(root):
    """Use the optional verification CSV as a fast all-NaN exclusion hint.

    Eligibility is still checked against actual finite-coordinate wells. No cell
    line ID is hard-coded; external datasets need not have this bookkeeping file.
    """
    path = Path(root) / "_coord_verify.csv"
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8-sig") as fh:
        return {row["file"]: float(row["nan_frac"]) < 1 for row in csv.DictReader(fh)}


def discover_files(cfg):
    if cfg.mode == "single":
        from models.cli import choose_file
        path = Path(choose_file(cfg.root, cfg.file)).resolve()
        import re
        match = re.fullmatch(r"plate(\d+)_(.+)\.h5ad", path.name)
        if not match or not 1 <= int(match[1]) <= 12:
            raise ValueError("Training requires a plate1..plate12_<cell_line>.h5ad file")
        return [{"path": str(path), "plate": int(match[1]), "cell_line": match[2]}]
    status = coordinate_status(cfg.root)
    return [{"path": str(Path(f.path).resolve()), "plate": f.plate, "cell_line": f.cell_line}
            for f in build_file_index(cfg.root) if 1 <= f.plate <= 12
            and status.get(Path(f.path).name, True)]


def make_plan(cfg):
    """Inspect wells once, saving the usable experiment list for exact resume."""
    files = []
    gene_names = None
    for record in tqdm(discover_files(cfg), desc="Indexing usable wells", unit="file"):
        backed = BackedFile(record["path"])
        try:
            if not backed.control_wells or not targets(backed, cfg):
                continue
            names = backed._adata.var_names.astype(str).tolist()
            if gene_names is None:
                gene_names = names
            elif names != gene_names:
                raise ValueError(f"Gene identity/order mismatch: {record['path']}")
            files.append({**record, "target_samples": [w.sample for w in targets(backed, cfg)]})
        finally:
            backed.close()
    if not files:
        raise ValueError("No usable files: check DMSO wells, finite coordinates and min_cells")
    lines = sorted({record["cell_line"] for record in files})
    val = []
    if cfg.model_type == "full" and cfg.mode == "all":
        rng = np.random.default_rng(cfg.seed)
        val = sorted(rng.permutation(lines)[:min(cfg.val_cell_lines, len(lines) - 1)].tolist())
    return DataPlan(files, [line for line in lines if line not in val], val,
                    len(gene_names), gene_names)


def epoch_units(plan, cfg, rng):
    """Five distinct available plates per line; single mode visits all target wells."""
    records = []
    for line in plan.train_lines:
        available = [r for r in plan.files if r["cell_line"] == line]
        count = 1 if cfg.mode == "single" else min(cfg.passes_per_cell_line, len(available))
        records.extend(available[int(i)] for i in rng.choice(len(available), count, replace=False))
    # Shuffle plates and lines without inventing missing grid entries.
    by_plate = {}
    for record in records:
        by_plate.setdefault(record["plate"], []).append(record)
    ordered = []
    for plate in rng.permutation(sorted(by_plate)):
        group = by_plate[int(plate)]
        ordered.extend(group[int(i)] for i in rng.permutation(len(group)))
    units = []
    for record in ordered:
        samples = record["target_samples"]
        count = len(samples) if cfg.mode == "single" else min(cfg.batches_per_unit, len(samples))
        selected = [samples[int(i)] for i in rng.choice(len(samples), count, replace=False)]
        if cfg.mode == "single":
            units.extend({**record, "samples": [sample]} for sample in selected)
        else:
            units.append({**record, "samples": selected})
    return units[:cfg.max_units_per_epoch] if cfg.max_units_per_epoch else units


def validation_units(plan, cfg):
    units = []
    for line in plan.val_lines:
        records = sorted((r for r in plan.files if r["cell_line"] == line), key=lambda r: r["plate"])
        for record in records[:cfg.val_plates]:
            units.append(record)
    return units


def sample_example(backed, sample, cfg, rng, index=0):
    """Cross-well sampling is allowed; cross-file/plate/line sampling is impossible."""
    target = next(w for w in targets(backed, cfg) if w.sample == sample)
    control = backed.control_wells[index % len(backed.control_wells)]
    return build_example(backed, control, target, cfg.n_control, cfg.n_target, rng)
