"""Run configuration and the non-interactive command-line interface."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path

from models.tahoe_data import DEFAULT_ROOT

MODELS_ROOT = Path(__file__).resolve().parents[1]
TRAINED_ROOT = MODELS_ROOT / "trained_model"


@dataclass
class TrainingConfig:
    model_type: str = "full"
    mode: str = "all"
    root: str = DEFAULT_ROOT
    file: str = ""
    dim: int = 512
    conc_dim: int = 32
    heads: int = 8
    layers: int = 1
    epochs: int = 60
    n_control: int = 256
    n_target: int = 128
    min_cells: int = 128
    passes_per_cell_line: int = 5
    batches_per_unit: int = 2
    file_cache: int = 6
    val_cell_lines: int = 2
    val_plates: int = 2
    lr: float = 1e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    dropout: float = 0.1
    scale_exponent: float = 0.5
    amp: str = "bf16"
    query_chunk: int = 4096
    readout_chunk: int = 16
    chem_tokens: str = "all"
    atom_cache: str = str(MODELS_ROOT / "cache" / "atom_embeddings.pt")
    seed: int = 0
    save_every: int = 50
    max_steps: int = 0
    max_units_per_epoch: int = 0

    def validate(self):
        if self.model_type not in ("full", "control") or self.mode not in ("all", "single"):
            raise ValueError("Choose full/control model and all/single training mode")
        for key in ("dim", "conc_dim", "heads", "epochs", "n_control", "n_target",
                    "min_cells", "passes_per_cell_line", "batches_per_unit", "file_cache"):
            if getattr(self, key) <= 0:
                raise ValueError(f"{key} must be positive")
        if self.dim < 4 or self.layers < 0:
            raise ValueError("dim must be >= 4 and layers must be >= 0")
        if self.model_type == "full" and (self.dim % self.heads or self.dim // self.heads > 128):
            raise ValueError("dim must be divisible by heads and dim / heads must be <= 128")
        if self.model_type == "full" and self.layers < 1:
            raise ValueError("The full model needs at least one perturbation layer")
        if self.amp not in ("off", "bf16"):
            raise ValueError("Supported precision modes: off, bf16")
        if self.chem_tokens not in ("all", "atoms"):
            raise ValueError("chem_tokens must be all or atoms")
        if self.lr <= 0 or self.weight_decay < 0 or self.grad_clip < 0:
            raise ValueError("lr must be positive; weight_decay and grad_clip cannot be negative")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        for key in ("query_chunk", "readout_chunk", "save_every", "max_steps",
                    "max_units_per_epoch", "val_cell_lines", "val_plates"):
            if getattr(self, key) < 0:
                raise ValueError(f"{key} cannot be negative")
        if self.mode == "single" and not self.file:
            raise ValueError("Single-file training requires a selected file")

    def to_dict(self):
        return asdict(self)


def parse_args(argv=None, *, single=False):
    parser = argparse.ArgumentParser(description="Tahoe full or DMSO-only training")
    parser.add_argument("--model", dest="model_type", choices=("full", "control"), default=None)
    parser.add_argument("--root", default=None)
    parser.add_argument("--file", default=None)
    parser.add_argument("--resume", default="", help="Run folder or last.pt; --epochs adds epochs")
    parser.add_argument("--dry-run", action="store_true", help="Inspect selection and one batch; no run saved")
    for flag in ("dim", "conc-dim", "heads", "layers", "epochs", "n-control", "n-target",
                 "min-cells", "passes-per-cell-line", "batches-per-unit", "file-cache",
                 "val-cell-lines", "val-plates", "query-chunk", "readout-chunk", "seed",
                 "save-every", "max-steps", "max-units-per-epoch"):
        parser.add_argument(f"--{flag}", type=int, default=None)
    for flag in ("lr", "weight-decay", "grad-clip", "dropout", "scale-exponent"):
        parser.add_argument(f"--{flag}", type=float, default=None)
    parser.add_argument("--amp", choices=("off", "bf16"), default=None)
    parser.add_argument("--chem-tokens", choices=("all", "atoms"), default=None)
    parser.add_argument("--atom-cache", default=None)
    args = parser.parse_args(argv)
    if args.resume:
        # A resumed run uses its saved settings, rather than CLI defaults.
        supplied = {k: v for k, v in vars(args).items()
                    if k not in ("resume", "epochs", "dry_run") and v is not None}
        if supplied or args.dry_run:
            parser.error("With --resume, supply only --epochs (additional epochs)")
        return args, None
    values = {k: v for k, v in vars(args).items()
              if k not in ("resume", "dry_run") and v is not None}
    values["mode"] = "single" if single else "all"
    if single:
        values = {"lr": 3e-4, "dropout": 0.0, "grad_clip": 10.0, "epochs": 10, **values}
        if not values.get("file"):
            from models.cli import choose_file
            values["file"] = choose_file(values.get("root", DEFAULT_ROOT))
    cfg = TrainingConfig(**values)
    cfg.validate()
    return args, cfg
