"""Train on ONE (plate, cell line) file — the overfitting test.

Pick a file from ``data_by_plate_cell_line/``, answer a few shape questions, and
it trains on that file only, saving a run under ``models/runtime/model_<stamp>/``.
The point is to find out whether this architecture *can* fit a single perturbation
experiment at all, before spending GPU-days on the full grid.

There is no train/held-out split: every drug well of the file is trained on, which
makes the curves a pure optimisation/memorisation measure.

    python models/train_small.py                     # interactive
    python models/train_small.py --file plate6_CVCL_1125.h5ad \
        --dim 512 --layers 1 --epochs 10

Layout of a run:

    models/runtime/model_20261124_153012/
        info.json     config + per-epoch metrics, rewritten every epoch
        model.pt      weights, overwritten every epoch (atomic)
        log.csv       one row per step
        tb/           TensorBoard events

An **epoch** is one pass over the file's drug wells, so
``steps_per_epoch = ceil(n_drug_wells / batch)``. Every epoch reshuffles the well
order. A fixed probe set (4 wells, fixed cells) is scored at the end of each
epoch so the epoch-to-epoch numbers are comparable.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.cli import ask, choose_file
from models.model import PerturbationModel, perturbation_loss
from models.molformer_atoms import AtomEmbeddingStore
from models.tahoe_data import DEFAULT_ROOT, BackedFile, build_example
from models.tb import TbLogger

try:
    from tqdm import tqdm as _tqdm

    def progress(iterable, **kwargs):
        return _tqdm(iterable, **kwargs)

    def say(message: str) -> None:
        _tqdm.write(message)

    HAVE_TQDM = True
except ImportError:  # keeps the script usable without tqdm installed
    def progress(iterable, **kwargs):
        return iterable

    def say(message: str) -> None:
        print(message)

    HAVE_TQDM = False

GB = 1e9
HERE = Path(__file__).resolve().parent
# bars only in a real terminal: redirected output (nohup, > log.txt, journald)
# gets clean periodic lines instead of thousands of carriage returns
USE_BARS = HAVE_TQDM and sys.stderr.isatty()


# --------------------------------------------------------------------------- #
# parameter / gradient grouping
# --------------------------------------------------------------------------- #
def group_of(name: str) -> str:
    """Map a parameter name to the part it belongs to.

    ``layers.0.kv_linear.weight``        -> ``layer0.kv_linear``
    ``layers.0.chem_ffn.0.weight``       -> ``layer0.chem_ffn``
    ``gene_embedding.weight``            -> ``gene_embedding``
    """
    if name.startswith("layers."):
        _, index, *rest = name.split(".")
        return f"layer{index}.{rest[0] if rest else '?'}"
    return name.split(".")[0]


def parameter_counts(model) -> dict[str, int]:
    counts: dict[str, int] = {}
    for name, parameter in model.named_parameters():
        key = group_of(name)
        counts[key] = counts.get(key, 0) + parameter.numel()
    return counts


def gradient_norms(model, order: list[str]) -> tuple[dict[str, float], float]:
    """L2 norm per part, plus the global norm (both measured before clipping)."""
    squared = {key: 0.0 for key in order}
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        key = group_of(name)
        squared[key] = squared.get(key, 0.0) + float(parameter.grad.detach().pow(2).sum())
    norms = {key: math.sqrt(value) for key, value in squared.items()}
    total = math.sqrt(sum(value * value for value in norms.values()))
    return norms, total


def print_parameter_table(counts: dict[str, int], total: int) -> None:
    print(f"\nparameters    : {total:,}   "
          f"({total * 4 / 1e6:.1f} MB fp32, {total * 12 / 1e6:.1f} MB with AdamW state)")
    print(f"  {'part':<28} {'params':>14} {'share':>8}")
    print(f"  {'-'*28} {'-'*14} {'-'*8}")
    for key, value in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {key:<28} {value:>14,} {100 * value / total:>7.1f}%")
    print(f"  {'total':<28} {total:>14,} {100.0:>7.1f}%")


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def make_batch(backed, store, wells, control_wells, cfg, rng, step):
    """Stack ``cfg.batch`` wells into one forward pass."""
    examples = []
    for j, well in enumerate(wells):
        control = control_wells[(step + j) % len(control_wells)]
        examples.append(
            build_example(
                backed, control, well, cfg.n_control, cfg.n_target, rng
            )
        )

    chem, mask = store.collate(
        [e.drug for e in examples],
        [e.concentration for e in examples],
        cfg.device,
        mode=cfg.chem_tokens,
    )
    return {
        "control": torch.stack([e.control for e in examples]).to(cfg.device),
        "chem": chem,
        "mask": mask,
        "coords": torch.stack([e.coords for e in examples]).to(cfg.device),
        "target": torch.stack([e.target for e in examples]).to(cfg.device),
        "drugs": [e.drug for e in examples],
    }


def forward(batch, cfg):
    with cfg.amp:
        predicted = cfg.model(
            batch["control"],
            batch["chem"],
            batch["coords"],
            batch["mask"],
            query_chunk=cfg.query_chunk,
            readout_chunk=cfg.readout_chunk,
        )
    return predicted.float()


def score(probes, cfg) -> float:
    cfg.model.eval()
    losses = []
    with torch.no_grad():
        for batch in probes:
            predicted = forward(batch, cfg)
            losses.append(float(perturbation_loss(predicted, batch["target"], 0.5)))
    cfg.model.train()
    return float(np.mean(losses))


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="train on a single h5ad file and watch it overfit",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--root", default=DEFAULT_ROOT)
    parser.add_argument("--file", default=None,
                        help="file name or path; omit to pick interactively")
    parser.add_argument("--dim", type=int, default=None)
    parser.add_argument("--heads", type=int, default=None)
    parser.add_argument("--layers", type=int, default=None)
    parser.add_argument("--conc-dim", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None, help="wells per step")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--n-control", type=int, default=256)
    parser.add_argument("--n-target", type=int, default=128)
    parser.add_argument("--min-cells", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.0,
                        help="0 by default: the goal is to fit, not generalise")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=10.0, help="0 disables")
    parser.add_argument("--amp", choices=["off", "bf16"], default="bf16")
    parser.add_argument("--chem-tokens", choices=["all", "atoms"], default="all")
    parser.add_argument("--query-chunk", type=int, default=4096, help="0 disables")
    parser.add_argument("--readout-chunk", type=int, default=16, help="0 disables")
    parser.add_argument("--probes", type=int, default=4,
                        help="fixed wells scored at the end of every epoch")
    parser.add_argument("--max-steps-per-epoch", type=int, default=0,
                        help="0 = one full pass over the file's wells")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-name", default=None,
                        help="default: model_<YYYYmmdd_HHMMSS>")
    parser.add_argument("--runtime", default=str(HERE / "runtime"))
    parser.add_argument("--no-tb", dest="tb", action="store_false", default=True)
    cfg = parser.parse_args(argv)

    # ---- file ------------------------------------------------------------
    path = choose_file(cfg.root, cfg.file)
    name = os.path.basename(path)
    print(f"\nfile: {name}")

    backed = BackedFile(path, cfg.min_cells)
    store = AtomEmbeddingStore()
    n_genes = backed.n_vars
    wells = backed.drug_wells(cfg.min_cells)
    control_wells = backed.control_wells
    if not wells or not control_wells:
        raise SystemExit(
            f"{name} has {len(wells)} drug wells with >= {cfg.min_cells} cells and "
            f"{len(control_wells)} DMSO wells -- nothing to train on"
        )

    # ---- shape -----------------------------------------------------------
    print(f"\ndataset (genes always come from the file):")
    print(f"  cells         : {backed.n_obs}")
    print(f"  genes         : {n_genes}")
    print(f"  drug wells    : {len(wells)}  (>= {cfg.min_cells} cells)")
    print(f"  control wells : {len(control_wells)} DMSO")
    print(f"  doses         : {sorted({w.concentration for w in wells})}")

    print(f"\nmodel shape:")
    dim = cfg.dim if cfg.dim is not None else ask("  dim (model width)", 512)
    heads = cfg.heads if cfg.heads is not None else ask("  heads", 8)
    if dim % heads:
        raise SystemExit(f"dim {dim} is not divisible by heads {heads}")
    if dim // heads > 128:
        raise SystemExit(
            f"head_dim {dim // heads} > 128: the gene self-attention over "
            f"{n_genes} tokens cannot use a fused kernel. Use --heads "
            f"{-(-dim // 128)} or more."
        )
    layers = cfg.layers if cfg.layers is not None else ask("  layers", 1)
    conc_dim = cfg.conc_dim if cfg.conc_dim is not None else ask("  conc_dim", 32)
    batch = cfg.batch if cfg.batch is not None else ask("  batch (wells per step)", 1)
    epochs = cfg.epochs if cfg.epochs is not None else ask("  epochs", 10)

    batch = max(1, min(batch, len(wells)))
    full_pass = math.ceil(len(wells) / batch)
    steps_per_epoch = full_pass
    if cfg.max_steps_per_epoch:
        steps_per_epoch = min(steps_per_epoch, cfg.max_steps_per_epoch)

    # ---- model -----------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg.device = device
    cfg.batch = batch
    cfg.query_chunk = cfg.query_chunk or None
    cfg.readout_chunk = cfg.readout_chunk or None
    cfg.chem_tokens = cfg.chem_tokens

    model = PerturbationModel(
        dim=dim,
        conc_dim=conc_dim,
        num_heads=heads,
        dropout=cfg.dropout,
        num_layers=layers,
        num_of_genes=n_genes,
        chem_dim=store.dim + 1,
        cell_coordi_dim=2,
    ).to(device)
    cfg.model = model

    amp_dtype = {"off": None, "bf16": torch.bfloat16}[cfg.amp]
    cfg.amp_name = cfg.amp
    cfg.amp = (
        contextlib.nullcontext()
        if amp_dtype is None
        else torch.autocast(device_type=device.type, dtype=amp_dtype)
    )

    n_params = sum(p.numel() for p in model.parameters())
    counts = parameter_counts(model)
    group_order = list(counts)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )

    print_parameter_table(counts, n_params)
    print(f"\n  head_dim      : {dim // heads}")
    print(f"  chem keys     : mode={cfg.chem_tokens}")
    print(f"  dropout       : {cfg.dropout}")
    print(f"  lr / clip     : {cfg.lr} / {cfg.grad_clip}")
    print(f"  batch         : {batch} well(s)")
    print(f"  epoch size    : {steps_per_epoch} steps"
          + (f"  (capped; a full pass over {len(wells)} wells is {full_pass})"
             if steps_per_epoch != full_pass else f"  (one pass over {len(wells)} wells)"))
    print(f"  epochs        : {epochs}")

    # ---- run directory ---------------------------------------------------
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = cfg.run_name or f"model_{stamp}"
    run_dir = Path(cfg.runtime) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    weights_path = run_dir / "model.pt"
    info_path = run_dir / "info.json"
    log_path = run_dir / "log.csv"

    tb = TbLogger(str(run_dir / "tb") if cfg.tb else None, strict=False)

    info = {
        "run": run_name,
        "created": datetime.now().isoformat(timespec="seconds"),
        "file": name,
        "file_path": str(path),
        "plate": backed.plate,
        "cell_line": backed.cell_line,
        "dataset": {
            "cells": int(backed.n_obs),
            "genes": int(n_genes),
            "drug_wells": len(wells),
            "control_wells": len(control_wells),
            "doses": sorted({w.concentration for w in wells}),
            "min_cells": cfg.min_cells,
            "n_control_cells": cfg.n_control,
            "n_target_cells": cfg.n_target,
        },
        "model": {
            "dim": dim, "heads": heads, "layers": layers,
            "conc_dim": conc_dim, "dropout": cfg.dropout,
            "chem_dim": store.dim + 1, "chem_tokens": cfg.chem_tokens,
            "params": int(n_params),
            "parameters": {k: int(v) for k, v in counts.items()},
        },
        "training": {
            "batch": batch, "epochs": epochs, "steps_per_epoch": steps_per_epoch,
            "lr": cfg.lr, "weight_decay": cfg.weight_decay,
            "grad_clip": cfg.grad_clip, "amp": cfg.amp_name,
            "seed": cfg.seed, "device": str(device),
        },
        "epochs": [],
        "status": "running",
    }
    # write it before training so a crashed run still documents its configuration
    info_path.write_text(json.dumps(info, indent=2), encoding="utf-8")

    print(f"\nrun           : {run_dir}")
    print(f"  info.json, model.pt (each epoch), log.csv, grad.csv, tb/")

    log_fh = open(log_path, "w", newline="", encoding="utf-8")
    log = csv.writer(log_fh)
    log.writerow(["epoch", "step", "global_step", "loss", "seconds", "drugs"])

    grad_path = run_dir / "grad.csv"
    grad_fh = open(grad_path, "w", newline="", encoding="utf-8")
    grad = csv.writer(grad_fh)
    grad.writerow(
        ["epoch", "step", "global_step", "loss", "grad_total"]
        + [f"grad_{key}" for key in group_order]
    )

    # ---- fixed probes + start loss ---------------------------------------
    rng = np.random.default_rng(cfg.seed)
    probe_idx = rng.permutation(len(wells))[: min(cfg.probes, len(wells))]
    probes = [
        make_batch(backed, store, [wells[i]], control_wells, cfg, rng, 0)
        for i in probe_idx
    ]
    probe_label = [wells[i].drug for i in probe_idx]

    start_loss = score(probes, cfg)
    zero_ref = float(np.mean([
        float(perturbation_loss(torch.zeros_like(p["target"]), p["target"], 0.5))
        for p in probes
    ]))
    info["probes"] = {"wells": probe_label, "start_loss": start_loss, "zero_loss": zero_ref}

    print(f"\nprobe wells   : {', '.join(probe_label)}")
    print(f"  untrained   : {start_loss:.4f}   (predict-zero reference {zero_ref:.4f})")

    # ---- train -----------------------------------------------------------
    model.train()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    global_step = 0
    t_start = time.time()
    step_times: list[float] = []
    peak_alloc = peak_res = 0.0

    epoch_bar = progress(
        range(1, epochs + 1), desc="epochs", unit="ep", position=0,
        disable=not USE_BARS,
    )
    for epoch in epoch_bar:
        order = rng.permutation(len(wells))
        epoch_losses = []
        epoch_start = time.time()

        step_bar = progress(
            range(1, steps_per_epoch + 1),
            desc=f"  epoch {epoch}", unit="step", position=1, leave=False,
            disable=not USE_BARS,
        )
        for step in step_bar:
            chosen = [wells[order[(step - 1) * batch + j]] for j in range(batch)
                      if (step - 1) * batch + j < len(wells)]
            batch_data = make_batch(backed, store, chosen, control_wells, cfg, rng, step)

            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
            t0 = time.time()

            optimizer.zero_grad(set_to_none=True)
            predicted = forward(batch_data, cfg)
            loss = perturbation_loss(predicted, batch_data["target"], 0.5)
            loss.backward()
            # measured before clipping, so the numbers are not rescaled
            norms, grad_total = gradient_norms(model, group_order)
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()

            if device.type == "cuda":
                torch.cuda.synchronize()
                peak_alloc = max(peak_alloc, torch.cuda.max_memory_allocated() / GB)
                peak_res = max(peak_res, torch.cuda.max_memory_reserved() / GB)
            elapsed = time.time() - t0

            global_step += 1
            value = float(loss.detach())
            epoch_losses.append(value)
            step_times.append(elapsed)

            log.writerow([epoch, step, global_step, f"{value:.6f}",
                          f"{elapsed:.3f}", "|".join(batch_data["drugs"])])
            grad.writerow(
                [epoch, step, global_step, f"{value:.6f}", f"{grad_total:.6e}"]
                + [f"{norms[key]:.6e}" for key in group_order]
            )

            if tb.enabled:
                tb.log_scalars(
                    {
                        "loss/train": value,
                        "grad/total": grad_total,
                        "perf/seconds_per_step": elapsed,
                        **{
                            f"grad/{key}": norms[key]
                            for key in ("gene_embedding", "chem_embedding",
                                        "merger", "coordi_layer")
                            if key in norms
                        },
                        **{k: v for k, v in TbLogger.step_vram().items()
                           if k.startswith("step_peak") or k == "total_gb"},
                    },
                    global_step,
                )

            if USE_BARS:
                mean_recent = float(np.mean(epoch_losses[-cfg.log_every:]))
                step_bar.set_postfix(
                    {
                        "loss": f"{mean_recent:.4f}",
                        "s/it": f"{elapsed:.2f}",
                        "vram": f"{peak_alloc:.2f}G",
                        "|g|": f"{grad_total:.1e}",
                    }
                )

            if not USE_BARS and (step % cfg.log_every == 0 or step == steps_per_epoch):
                mean_recent = float(np.mean(epoch_losses[-cfg.log_every:]))
                left = ((epochs - epoch) * steps_per_epoch + (steps_per_epoch - step))
                eta = left * float(np.mean(step_times[-50:])) / 60
                say(f"  epoch {epoch}/{epochs}  step {step}/{steps_per_epoch}  "
                    f"loss {mean_recent:.4f}  {elapsed:.2f}s/it  "
                    f"vram {peak_alloc:.2f}G  |g| {grad_total:.1e}  eta {eta:.1f} min")

        eval_loss = score(probes, cfg)
        epoch_seconds = time.time() - epoch_start
        epoch_mean = float(np.mean(epoch_losses))

        if USE_BARS:
            epoch_bar.set_postfix(
                {"train": f"{epoch_mean:.4f}", "probe": f"{eval_loss:.4f}"}
            )

        info["epochs"].append(
            {
                "epoch": epoch,
                "steps": len(epoch_losses),
                "train_loss": epoch_mean,
                "train_loss_last_step": epoch_losses[-1],
                "probe_loss": eval_loss,
                "seconds": round(epoch_seconds, 2),
                "peak_vram_gb": round(peak_alloc, 3),
            }
        )
        info["status"] = "running"
        info["total_seconds"] = round(time.time() - t_start, 1)
        info_path.write_text(json.dumps(info, indent=2), encoding="utf-8")

        # atomic overwrite so a crash mid-save cannot corrupt the weights
        tmp = weights_path.with_suffix(".pt.tmp")
        torch.save(
            {
                "model": model.state_dict(),
                "epoch": epoch,
                "file": name,
                "genes": n_genes,
                "model_config": info["model"],
                "loss": epoch_mean,
            },
            tmp,
        )
        os.replace(tmp, weights_path)

        if tb.enabled:
            tb.log_scalars(
                {"loss/epoch_train": epoch_mean, "loss/probe": eval_loss},
                epoch,
            )

        say(
            f"== epoch {epoch}/{epochs}: train {epoch_mean:.4f}  "
            f"probe {eval_loss:.4f}  {epoch_seconds/60:.1f} min  "
            f"-> saved {weights_path.name}"
        )
        log_fh.flush()
        grad_fh.flush()

    info["status"] = "finished"
    info["total_seconds"] = round(time.time() - t_start, 1)
    info["final"] = {
        "train_loss": info["epochs"][-1]["train_loss"],
        "probe_loss": info["epochs"][-1]["probe_loss"],
        "start_probe_loss": start_loss,
        "reduced_by": round(start_loss - info["epochs"][-1]["probe_loss"], 4),
    }
    info_path.write_text(json.dumps(info, indent=2), encoding="utf-8")

    log_fh.close()
    grad_fh.close()
    if tb.enabled:
        tb.close()
    backed.close()

    wall = time.time() - t_start
    print(f"\n=== done ===")
    print(f"  run        : {run_dir}")
    print(f"  wall       : {wall/60:.1f} min  ({np.mean(step_times):.2f} s/step)")
    print(f"  probe loss : {start_loss:.4f} -> {info['final']['probe_loss']:.4f}  "
          f"({info['final']['reduced_by']:+.4f})")
    print(f"  peak vram  : {peak_alloc:.2f} GB allocated / {peak_res:.2f} GB reserved")
    print(f"  weights    : {weights_path}")
    if tb.enabled:
        print(f"  curves     : python -m tensorboard.main --logdir {run_dir}")


if __name__ == "__main__":
    main()
