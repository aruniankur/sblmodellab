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
    conc_dim: int = 64
    heads: int = 8
    head_dim: int | None = None  # None preserves the architecture of legacy checkpoints
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
    grad_clip: float = 3.0
    dropout: float = 0.1
    normalization: str = "layernorm"
    gene_statistics: bool = True
    attention_normalization: bool = True
    gene_residual: bool = True
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

    def __post_init__(self):
        if self.head_dim is not None:
            self.set_head_shape(self.heads, self.head_dim)

    def set_head_shape(self, heads: int, head_dim: int):
        """Single source of truth for new-run sizing; saved legacy widths stay intact."""
        if heads < 1 or head_dim < 1:
            raise ValueError("heads and head_dim must be positive")
        self.heads, self.head_dim = heads, head_dim
        self.dim = heads * head_dim
        self.conc_dim = head_dim

    def validate(self):
        if self.model_type not in ("full", "control") or self.mode not in ("all", "single"):
            raise ValueError("Choose full/control model and all/single training mode")
        for key in ("dim", "conc_dim", "heads", "epochs", "n_control", "n_target",
                    "min_cells", "passes_per_cell_line", "batches_per_unit", "file_cache"):
            if getattr(self, key) <= 0:
                raise ValueError(f"{key} must be positive")
        if self.head_dim is not None and (self.dim != self.heads * self.head_dim or self.conc_dim != self.head_dim):
            raise ValueError("dim must equal heads * head_dim and conc_dim must equal head_dim")
        if self.dim < 4 or self.layers < 0:
            raise ValueError("dim must be >= 4 and layers must be >= 0")
        if self.model_type == "full" and self.dim % self.heads:
            raise ValueError("dim must be divisible by heads")
        if self.model_type == "full" and self.layers < 1:
            raise ValueError("The full model needs at least one perturbation layer")
        if not isinstance(self.gene_residual, bool):
            raise ValueError("gene_residual must be a boolean")
        if not isinstance(self.gene_statistics, bool):
            raise ValueError("gene_statistics must be a boolean")
        if self.normalization not in ("layernorm", "none"):
            raise ValueError("normalization must be layernorm or none")
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
    for flag in ("head-dim", "heads", "layers", "epochs", "n-control", "n-target",
                 "min-cells", "passes-per-cell-line", "batches-per-unit", "file-cache",
                 "val-cell-lines", "val-plates", "query-chunk", "readout-chunk", "seed",
                 "save-every", "max-steps", "max-units-per-epoch"):
        parser.add_argument(f"--{flag}", type=int, default=None)
    for flag in ("lr", "weight-decay", "grad-clip", "dropout", "scale-exponent"):
        parser.add_argument(f"--{flag}", type=float, default=None)
    parser.add_argument("--amp", choices=("off", "bf16"), default=None)
    parser.add_argument("--chem-tokens", choices=("all", "atoms"), default=None)
    parser.add_argument("--atom-cache", default=None)
    parser.add_argument("--normalization", choices=("layernorm", "none"), default=None,
                        help="new-run normalization; none enables an architecture ablation")
    parser.add_argument("--no-gene-stats", dest="gene_statistics", action="store_false", default=None,
                        help="new-run ablation: omit the five log-expression statistics")
    parser.add_argument("--no-gene-residual", dest="gene_residual", action="store_false", default=None,
                        help="new-run ablation: disable the gene embedding residual")
    args = parser.parse_args(argv)
    if args.resume:
        # Resume keeps the saved architecture; epochs and clipping are explicit policies.
        supplied = {k: v for k, v in vars(args).items()
                    if k not in ("resume", "epochs", "grad_clip", "dry_run") and v is not None}
        if supplied or args.dry_run:
            parser.error("With --resume, supply only --epochs and optionally --grad-clip")
        return args, None
    values = {k: v for k, v in vars(args).items()
              if k not in ("resume", "dry_run") and v is not None}
    values.setdefault("head_dim", 64)
    values["mode"] = "single" if single else "all"
    if single:
        values = {"lr": 3e-4, "dropout": 0.0, "epochs": 10, **values}
        if not values.get("file"):
            from models.cli import choose_file
            values["file"] = choose_file(values.get("root", DEFAULT_ROOT))
    cfg = TrainingConfig(**values)
    cfg.validate()
    return args, cfg
