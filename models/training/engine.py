"""Shared full-grid and single-file training engine for both architectures."""
from __future__ import annotations

import copy
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from models.tahoe_data import FileCache
from models.training.checkpoints import (create_run, load_checkpoint, restore_rng,
                                          save_checkpoint, write_json)
from models.training.config import TrainingConfig
from models.training.data import (DataPlan, epoch_units, make_plan, sample_example,
                                  targets, validation_units)
from models.training.metrics import PassLog, gradient_norms, parameter_counts
from models.training.modeling import create_model, load_chemistry, loss_for


def device_for(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if cfg.amp == "bf16" and device.type == "cuda" and not torch.cuda.is_bf16_supported():
        raise ValueError("This GPU does not support BF16. Use --amp off")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    return device


def checked_file(cache, record, plan):
    backed = cache.get(record["path"])
    if backed.n_vars != plan.n_genes or backed._adata.var_names.astype(str).tolist() != plan.gene_names:
        raise ValueError(f"Gene vocabulary changed: {record['path']}")
    return backed


def evaluate(model, store, plan, cfg, device, cache):
    """Fixed-seed held-out score, or a reconstruction probe for control/single runs."""
    records = validation_units(plan, cfg)
    label = "validation"
    if not records:
        if cfg.model_type == "full" and cfg.mode == "all":
            return None, "validation"
        # These are training-file probes, deliberately not labeled generalization.
        records = plan.files[:min(4, len(plan.files))]
        label = "reconstruction_probe"
    rng = np.random.default_rng(cfg.seed + 100_000)
    losses = []
    model.eval()
    try:
        with torch.no_grad():
            for record in tqdm(records, desc=label.replace("_", " "), leave=False, unit="file"):
                backed = checked_file(cache, record, plan)
                wells = targets(backed, cfg)
                picks = rng.permutation(len(wells))[:min(cfg.batches_per_unit, len(wells))]
                for index, pick in enumerate(picks):
                    example = sample_example(backed, wells[int(pick)].sample, cfg, rng, index)
                    losses.append(float(loss_for(model, example, cfg, device, store)))
    finally:
        model.train()
    return float(np.mean(losses)) if losses else None, label


def dry_run(cfg):
    cfg.validate()
    plan = make_plan(cfg)
    store = load_chemistry(cfg)
    cache = FileCache(1)
    try:
        record = next(r for r in plan.files if r["cell_line"] in plan.train_lines)
        backed = cache.get(record["path"])
        example = sample_example(backed, targets(backed, cfg)[0].sample, cfg,
                                 np.random.default_rng(cfg.seed))
        print(f"Usable files: {len(plan.files)}; train lines: {len(plan.train_lines)}; held out: {plan.val_lines}")
        print(f"genes={plan.n_genes}; control={tuple(example.control.shape)}; "
              f"target={tuple(example.target.shape)}; coords={tuple(example.coords.shape)}")
        if store is not None:
            chem, mask = store.collate([example.drug], [example.concentration], mode=cfg.chem_tokens)
            print(f"chem={tuple(chem.shape)}; mask={tuple(mask.shape)}")
        else:
            print("DMSO-only: no chemical cache or chemical tensors")
    finally:
        cache.close()


def train(cfg=None, *, resume="", additional_epochs=None):
    """Start a fresh run, or restore the saved configuration and add epochs.

    A unit ends with one optimizer update. Checkpoints are written only at unit
    boundaries, so accumulated gradients never need to be serialized. Following
    an interruption, resume replays only work newer than the last checkpoint.
    """
    state = None
    if resume:
        run_dir, state = load_checkpoint(resume)
        cfg = TrainingConfig(**state["config"])
        if additional_epochs is None or additional_epochs <= 0:
            raise ValueError("Specify a positive number of additional epochs")
        progress = state["progress"]
        # An incomplete current epoch counts as the first additional epoch.
        cfg.epochs = progress["epoch"] + additional_epochs
        plan = DataPlan.from_dict(state["plan"])
        info = copy.deepcopy(state["info"])
    else:
        if cfg is None:
            raise ValueError("A new run requires configuration")
        cfg.validate()
        cfg.root = str(Path(cfg.root).expanduser().resolve())
        cfg.atom_cache = str(Path(cfg.atom_cache).expanduser().resolve())
        if cfg.file:
            candidate = Path(cfg.file).expanduser()
            if not candidate.is_file():
                candidate = Path(cfg.root) / candidate
            cfg.file = str(candidate.resolve())
        run_dir = create_run(cfg.model_type)
        info = {"schema_version": 1, "run": run_dir.name,
                "created": datetime.now().isoformat(timespec="seconds"),
                "status": "indexing", "config": cfg.to_dict(), "epochs": []}
        write_json(run_dir / "info.json", info)
        try:
            plan = make_plan(cfg)
        except BaseException as exc:
            info.update(status="failed", error=str(exc))
            write_json(run_dir / "info.json", info)
            raise
        progress = {"epoch": 0, "cursor": 0, "units": None, "optimizer_step": 0,
                    "backward_pass": 0, "loss_sum": 0.0, "passes_in_epoch": 0,
                    "best_score": None}
    try:
        cfg.validate()
        device = device_for(cfg)
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
        rng = np.random.default_rng(cfg.seed)
        store = load_chemistry(cfg)
        chem_dim = store.dim + 1 if store is not None else None
        model = create_model(cfg, plan.n_genes, chem_dim).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        if state is not None:
            model.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            restore_rng(state, rng)
        groups = parameter_counts(model)
        info.update(status="running", config=cfg.to_dict(), data=plan.to_dict(),
                    model={"n_genes": plan.n_genes, "chem_dim": chem_dim,
                           "parameters": groups, "parameter_total": sum(groups.values()),
                           "uses_attention_stack": cfg.model_type == "full"},
                    device=str(device), target_epochs=cfg.epochs)
        write_json(run_dir / "info.json", info)
        write_json(run_dir / "config.json", cfg.to_dict())
        print(f"Run: {run_dir}\nModel: {cfg.model_type}; mode: {cfg.mode}; "
              f"train cell lines: {len(plan.train_lines)}; validation lines: {plan.val_lines}; "
              f"genes: {plan.n_genes}; parameters: {sum(groups.values()):,}")
        if cfg.model_type == "control":
            print("DMSO-only reconstruction. heads/layers/dropout do not alter the control architecture.")
        cache = FileCache(cfg.file_cache)
        log = PassLog(run_dir / "log.csv", groups, progress["backward_pass"])

    except BaseException as exc:
        info.update(status="failed", error=str(exc))
        write_json(run_dir / "info.json", info)
        raise

    def save(name="last.pt"):
        save_checkpoint(run_dir / name, model, optimizer, cfg, plan, progress, rng, info)

    # A fresh run is resumable even if the first unit fails or is interrupted.
    if state is None:
        save()
    model.train()
    stopped = False
    started_steps = progress["optimizer_step"]
    epoch_bar = tqdm(total=cfg.epochs, initial=progress["epoch"], desc="Epochs", unit="epoch")
    try:
        while progress["epoch"] < cfg.epochs:
            epoch_start = time.perf_counter()
            if progress["units"] is None:
                progress["units"] = epoch_units(plan, cfg, rng)
            units = progress["units"]
            remaining = units[progress["cursor"]:]
            pass_total = sum(len(u["samples"]) for u in units)
            done_passes = sum(len(u["samples"]) for u in units[:progress["cursor"]])
            with tqdm(total=pass_total, initial=done_passes,
                      desc=f"Epoch {progress['epoch'] + 1} backward passes", unit="pass", leave=False) as bar:
                for unit in remaining:
                    backed = checked_file(cache, unit, plan)
                    optimizer.zero_grad(set_to_none=True)
                    count = len(unit["samples"])
                    for micro, sample in enumerate(unit["samples"]):
                        start = time.perf_counter()
                        if device.type == "cuda":
                            torch.cuda.reset_peak_memory_stats(device)
                        example = sample_example(backed, sample, cfg, rng, micro)
                        previous = None
                        if micro:
                            previous = {name: p.grad.detach().clone() for name, p in model.named_parameters()
                                        if p.grad is not None}
                        loss = loss_for(model, example, cfg, device, store)
                        if not torch.isfinite(loss):
                            raise FloatingPointError(f"Non-finite loss in {unit['path']} / {sample}")
                        backward_loss = loss / count
                        backward_loss.backward()
                        norms, grad_total = gradient_norms(model, groups, previous, multiplier=count)
                        del previous
                        if not np.isfinite(grad_total):
                            raise FloatingPointError(f"Non-finite gradients in {unit['path']} / {sample}")
                        value = float(loss.detach())
                        progress["backward_pass"] += 1
                        progress["passes_in_epoch"] += 1
                        progress["loss_sum"] += value
                        log.write({"epoch": progress["epoch"] + 1, "unit": progress["cursor"] + 1,
                                   "optimizer_step": progress["optimizer_step"] + 1,
                                   "backward_pass": progress["backward_pass"], "micro_batch": micro + 1,
                                   "accumulation_count": count, "file": Path(unit["path"]).name,
                                   "plate": unit["plate"], "cell_line": unit["cell_line"],
                                   "control_sample": backed.control_wells[micro % len(backed.control_wells)].sample,
                                   "target_sample": sample, "drug": example.drug,
                                   "dose": example.concentration, "loss": value,
                                   "backward_loss": float(backward_loss.detach()), "grad_total": grad_total,
                                   "seconds": time.perf_counter() - start,
                                   "peak_vram_gb": torch.cuda.max_memory_allocated(device) / 1e9 if device.type == "cuda" else 0,
                                   **{f"grad_{key}": value for key, value in norms.items()}})
                        bar.update()
                        bar.set_postfix(loss=f"{value:.4f}", grad=f"{grad_total:.2e}",
                                        file=Path(unit["path"]).stem)
                    if cfg.grad_clip:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip,
                                                       error_if_nonfinite=True)
                    optimizer.step()
                    progress["optimizer_step"] += 1
                    progress["cursor"] += 1
                    if cfg.save_every and progress["optimizer_step"] % cfg.save_every == 0:
                        save()
                    if (cfg.max_steps and progress["optimizer_step"] - started_steps >= cfg.max_steps
                            and progress["cursor"] < len(units)):
                        stopped = True
                        break
            if stopped:
                # Mid-epoch position, sampler state and CSV rows are committed together.
                save()
                break
            score, label = evaluate(model, store, plan, cfg, device, cache)
            train_loss = progress["loss_sum"] / max(1, progress["passes_in_epoch"])
            info["epochs"].append({"epoch": progress["epoch"] + 1, "train_loss": train_loss,
                                   label: score, "seconds": time.perf_counter() - epoch_start,
                                   "optimizer_step": progress["optimizer_step"],
                                   "backward_pass": progress["backward_pass"]})
            improved = score is not None and (progress["best_score"] is None or score < progress["best_score"])
            if improved:
                progress["best_score"] = score
            progress.update(epoch=progress["epoch"] + 1, cursor=0, units=None, loss_sum=0.0,
                            passes_in_epoch=0)
            save()
            if improved:
                save("best.pt")
            write_json(run_dir / "info.json", info)
            tqdm.write(f"Epoch {progress['epoch']}: train={train_loss:.5f}, {label}={score}")
            epoch_bar.update()
            if cfg.max_steps and progress["optimizer_step"] - started_steps >= cfg.max_steps:
                stopped = progress["epoch"] < cfg.epochs
                break
        info["status"] = "stopped" if stopped else "finished"
    except KeyboardInterrupt:
        # A backward pass can be interrupted halfway through a unit. Preserve the
        # last completed checkpoint rather than saving partially updated state.
        info["status"] = "interrupted"
        tqdm.write("Interrupted. Resume last.pt; work since its last save will be replayed.")
    except BaseException as exc:
        info.update(status="failed", error=str(exc))
        raise
    finally:
        info["updated"] = datetime.now().isoformat(timespec="seconds")
        write_json(run_dir / "info.json", info)
        log.close()
        cache.close()
        epoch_bar.close()
    return run_dir
