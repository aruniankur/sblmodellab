"""Atomic run metadata and checkpoints, including exact sampler/RNG continuation."""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from models.training.config import TRAINED_ROOT

SCHEMA_VERSION = 10


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def create_run(model_type, root=None):
    root = Path(root or TRAINED_ROOT)
    root.mkdir(parents=True, exist_ok=True)
    # Microseconds avoid collisions when several jobs are launched together.
    run = root / f"model_{datetime.now():%Y%m%d_%H%M%S_%f}_{model_type}"
    run.mkdir()
    return run


def checkpoint_path(value):
    path = Path(value).resolve()
    if path.is_dir():
        path = path / "last.pt"
    if path.name != "last.pt":
        raise ValueError("Resume from the run folder or last.pt (best.pt is for evaluation)")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def load_checkpoint(value):
    path = checkpoint_path(value)
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state.get("schema_version") not in (1, 2, 3, 4, 5, 6, 7, 8, 9, SCHEMA_VERSION):
        raise ValueError("This checkpoint predates the organized pipeline; start a new run")
    if state["schema_version"] == 1:
        # Older runs have no norm parameters or matching optimizer/CSV groups.
        # Resume their original architecture; never silently insert trainable layers.
        state["config"].setdefault("normalization", "none")
    elif "normalization" not in state["config"]:
        raise ValueError("Checkpoint is missing its normalization architecture setting")
    if state["schema_version"] < 3:
        # Old merger weights have no additional statistic columns.
        state["config"].setdefault("gene_statistics", False)
    elif "gene_statistics" not in state["config"]:
        raise ValueError("Checkpoint is missing its gene-statistics architecture setting")
    if state["schema_version"] < 4:
        # Residuals change forward behavior even though no parameter keys change.
        state["config"].setdefault("gene_residual", False)
    elif "gene_residual" not in state["config"]:
        raise ValueError("Checkpoint is missing its gene-residual architecture setting")
    if state["schema_version"] < 5:
        state["config"].setdefault("attention_normalization", False)
    elif "attention_normalization" not in state["config"]:
        raise ValueError("Checkpoint is missing its attention-normalization architecture setting")
    if state["schema_version"] < 6:
        # This residual changes behavior without adding parameter tensors.
        state["config"].setdefault("chem_ffn_residual", False)
    elif "chem_ffn_residual" not in state["config"]:
        raise ValueError("Checkpoint is missing its chemical-FFN-residual architecture setting")
    if state["schema_version"] < 8:
        # Earlier configs did not record FFN width / joint-norm placement.
        # Infer from tensors, including experimental older-schema checkpoints.
        weights = state.get("model", {})
        first_ffn = weights.get("layers.0.chem_ffn.0.weight")
        gene_ffn = weights.get("layers.0.gene_ffn.0.weight")
        if first_ffn is not None:
            width = first_ffn.shape[1]
            if first_ffn.shape[0] % width or (gene_ffn is not None and gene_ffn.shape != first_ffn.shape):
                raise ValueError("Checkpoint has inconsistent chemical/gene FFN widths")
            inferred_multiplier = first_ffn.shape[0] // width
        else:
            inferred_multiplier = 4
        state["config"].setdefault("ffn_multiplier", inferred_multiplier)
        has_joint_norm = any(key.startswith("layers.") and ".joint_norm." in key for key in weights)
        state["config"].setdefault("joint_normalization", has_joint_norm if weights else True)
    elif "ffn_multiplier" not in state["config"] or "joint_normalization" not in state["config"]:
        raise ValueError("Checkpoint is missing FFN-width or joint-normalization architecture settings")
    if state["schema_version"] < 9:
        # Old runs have no gate parameters, optimizer slots or CSV gradient groups.
        state["config"].setdefault("chem_skip_gate", False)
        state["config"].setdefault("chem_skip_init", 0.1)
    elif "chem_skip_gate" not in state["config"] or "chem_skip_init" not in state["config"]:
        raise ValueError("Checkpoint is missing chemical-skip-gate architecture settings")
    if state["schema_version"] < 10:
        # Preserve old one-well forwards, accumulation, CSV and RNG continuation.
        state["config"].setdefault("batch_size", 1)
        state["config"].setdefault("checkpoint_layers", False)
        state["config"].setdefault("accumulate_batches", True)
    elif "batch_size" not in state["config"] or "checkpoint_layers" not in state["config"] or "accumulate_batches" not in state["config"]:
        raise ValueError("Checkpoint is missing its forward batch_size or checkpoint_layers setting")
    return path.parent, state


def save_checkpoint(path, model, optimizer, cfg, plan, progress, rng, info):
    state = {
        "schema_version": SCHEMA_VERSION,
        "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "config": cfg.to_dict(), "plan": plan.to_dict(), "progress": progress,
        "rng": rng.bit_generator.state, "torch_rng": torch.get_rng_state(),
        "numpy_rng": np.random.get_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "info": info,
    }
    path = Path(path)
    tmp = path.with_suffix(".pt.tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


def restore_rng(state, rng):
    rng.bit_generator.state = state["rng"]
    torch.set_rng_state(state["torch_rng"])
    np.random.set_state(state["numpy_rng"])
    if state.get("cuda_rng") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda_rng"])
