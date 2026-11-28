"""Shared full-grid and single-file training engine for both architectures."""
from __future__ import annotations

import copy
import json
import math
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from models.expression_features import GENE_STAT_NAMES
from models.tahoe_data import FileCache
from models.training.checkpoints import (SCHEMA_VERSION, create_run, load_checkpoint, restore_rng,
                                          save_checkpoint, write_json)
from models.training.config import TrainingConfig
from models.training.data import (DataPlan, epoch_units, make_plan, sample_example,
                                  targets, validation_units)
from models.training.metrics import PassLog, gradient_norms, parameter_counts
from models.training.modeling import create_model, load_chemistry, losses_for_batch, mean_baseline_losses


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
                examples = [sample_example(backed, wells[int(pick)].sample, cfg, rng, index)
                            for index, pick in enumerate(picks)]
                for offset in range(0, len(examples), cfg.batch_size):
                    losses.extend(losses_for_batch(model, examples[offset:offset + cfg.batch_size],
                                                   cfg, device, store).cpu().tolist())
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
        wells = targets(backed, cfg)
        examples = [sample_example(backed, well.sample, cfg, np.random.default_rng(cfg.seed + i), i)
                    for i, well in enumerate(wells[:cfg.batch_size])]
        print(f"Usable files: {len(plan.files)}; train lines: {len(plan.train_lines)}; held out: {plan.val_lines}")
        print(f"B={len(examples)}; genes={plan.n_genes}; control={tuple(torch.stack([e.control for e in examples]).shape)}; "
              f"target={tuple(torch.stack([e.target for e in examples]).shape)}; "
              f"coords={tuple(torch.stack([e.coords for e in examples]).shape)}")
        if store is not None:
            chem, mask = store.collate([e.drug for e in examples], [e.concentration for e in examples], mode=cfg.chem_tokens)
            print(f"chem={tuple(chem.shape)}; mask={tuple(mask.shape)}; drugs={[e.drug for e in examples]}")
        else:
            print("DMSO-only: no chemical cache or chemical tensors")
    finally:
        cache.close()


def train(cfg=None, *, resume="", additional_epochs=None, grad_clip=None, batch_size=None):
    """Start a fresh run, or restore the saved configuration and add epochs.

    A unit ends with one optimizer update. Checkpoints are written only at unit
    boundaries, so accumulated gradients never need to be serialized. Following
    an interruption, resume replays only work newer than the last checkpoint.
    """
    state = None
    if resume:
        run_dir, state = load_checkpoint(resume)
        cfg = TrainingConfig(**state["config"])
        # Keep the saved architecture, but use the current requested clipping policy.
        cfg.grad_clip = TrainingConfig().grad_clip if grad_clip is None else grad_clip
        if additional_epochs is None or additional_epochs <= 0:
            raise ValueError("Specify a positive number of additional epochs")
        progress = state["progress"]
        if batch_size is not None:
            if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
                raise ValueError("batch_size must be a positive integer")
            cfg.batch_size = batch_size
            cfg.accumulate_batches = False
            if progress["units"] is not None:
                done = progress["units"][:progress["cursor"]]
                pending = progress["units"][progress["cursor"]:]
                progress["units"] = done + [{**unit, "samples": unit["samples"][i:i + batch_size]}
                    for unit in pending for i in range(0, len(unit["samples"]), batch_size)]

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
        info = {"schema_version": SCHEMA_VERSION, "run": run_dir.name,
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
        info.update(schema_version=SCHEMA_VERSION, status="running", config=cfg.to_dict(), data=plan.to_dict(),
                    model={"n_genes": plan.n_genes, "chem_dim": chem_dim,
                           "parameters": groups, "parameter_total": sum(groups.values()),
                           "uses_attention_stack": cfg.model_type == "full",
                           "normalization": cfg.normalization,
                           "attention_normalization": cfg.attention_normalization if cfg.model_type == "full" else False,
                           "legacy_attention_norms": cfg.legacy_attention_norms if cfg.model_type == "full" else False,
                           "chem_skip_gate": cfg.chem_skip_gate if cfg.model_type == "full" else False,
                           "chem_skip_init": cfg.chem_skip_init if cfg.model_type == "full" and cfg.chem_skip_gate else None,
                           "chem_ffn_residual": cfg.chem_ffn_residual if cfg.model_type == "full" else False,
                           "ffn_multiplier": cfg.ffn_multiplier if cfg.model_type == "full" else None,
                           "joint_normalization": cfg.joint_normalization if cfg.model_type == "full" else False,
                           "gene_residual": cfg.gene_residual,
                           "gene_statistics": list(GENE_STAT_NAMES) if cfg.gene_statistics else []},
                    device=str(device), target_epochs=cfg.epochs,
                    batching={"forward_batch_size": cfg.batch_size, "max_wells_per_update": cfg.batches_per_unit if cfg.accumulate_batches else cfg.batch_size,
                              "checkpoint_layers": cfg.checkpoint_layers if cfg.model_type == "full" else False,
                              "update_every_forward": not cfg.accumulate_batches})
        write_json(run_dir / "info.json", info)
        write_json(run_dir / "config.json", cfg.to_dict())
        print(f"Run: {run_dir}\nModel: {cfg.model_type}; mode: {cfg.mode}; "
              f"train cell lines: {len(plan.train_lines)}; validation lines: {plan.val_lines}; "
              f"genes: {plan.n_genes}; parameters: {sum(groups.values()):,}; normalization: {cfg.normalization}; "
              f"forward batch size: {cfg.batch_size}")
        if cfg.model_type == "control":
            print("DMSO-only reconstruction. Head sizing sets width/bottleneck; layers/dropout are inactive.")
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
            pass_total = sum(math.ceil(len(u["samples"]) / cfg.batch_size) for u in units)
            done_passes = sum(math.ceil(len(u["samples"]) / cfg.batch_size) for u in units[:progress["cursor"]])
            with tqdm(total=pass_total, initial=done_passes,
                      desc=f"Epoch {progress['epoch'] + 1} backward passes", unit="pass", leave=False) as bar:
                for unit in remaining:
                    backed = checked_file(cache, unit, plan)
                    optimizer.zero_grad(set_to_none=True)
                    count = len(unit["samples"])
                    if not cfg.accumulate_batches and count > cfg.batch_size:
                        raise ValueError("Optimizer units must contain a single forward batch")
                    accumulation_count = math.ceil(count / cfg.batch_size)
                    for micro, offset in enumerate(range(0, count, cfg.batch_size)):
                        start = time.perf_counter()
                        if device.type == "cuda":
                            torch.cuda.reset_peak_memory_stats(device)
                        samples = unit["samples"][offset:offset + cfg.batch_size]
                        examples = [sample_example(backed, sample, cfg, rng, offset + index)
                                    for index, sample in enumerate(samples)]
                        size = len(examples)
                        previous = ({name: p.grad.detach().clone() for name, p in model.named_parameters()
                                     if p.grad is not None} if micro else None)
                        per_well = losses_for_batch(model, examples, cfg, device, store)
                        loss = per_well.mean()
                        if not torch.isfinite(per_well).all():
                            raise FloatingPointError(f"Non-finite loss in {unit['path']} / {samples}")
                        # Equal weight per well, including a short final forward batch.
                        backward_loss = loss * (size / count)
                        backward_loss.backward()
                        norms, grad_total = gradient_norms(model, groups, previous, multiplier=count / size)
                        del previous
                        if not np.isfinite(grad_total):
                            raise FloatingPointError(f"Non-finite gradients in {unit['path']} / {samples}")
                        values = per_well.detach().cpu().tolist()
                        baselines = [mean_baseline_losses(example, cfg) for example in examples]
                        value = float(loss.detach())
                        t_mean_loss = float(np.mean([v[0] for v in baselines]))
                        i_mean_loss = float(np.mean([v[1] for v in baselines]))
                        progress["backward_pass"] += 1
                        progress["passes_in_epoch"] += size  # wells, for weighted epoch loss
                        progress["loss_sum"] += sum(values)
                        details = [{"control_sample": backed.control_wells[(offset + i) % len(backed.control_wells)].sample,
                                    "target_sample": samples[i], "drug": example.drug,
                                    "dose": example.concentration, "loss": values[i],
                                    "t_mean_loss": baselines[i][0], "i_mean_loss": baselines[i][1]}
                                   for i, example in enumerate(examples)]
                        doses = {example.concentration for example in examples}
                        log.write({"epoch": progress["epoch"] + 1, "unit": progress["cursor"] + 1,
                                   "optimizer_step": progress["optimizer_step"] + 1,
                                   "backward_pass": progress["backward_pass"], "micro_batch": micro + 1,
                                   "accumulation_count": accumulation_count, "batch_size": size,
                                   "batch_samples": json.dumps(details), "file": Path(unit["path"]).name,
                                   "plate": unit["plate"], "cell_line": unit["cell_line"],
                                   "control_sample": " | ".join(d["control_sample"] for d in details),
                                   "target_sample": " | ".join(samples),
                                   "drug": " | ".join(example.drug for example in examples),
                                   "dose": next(iter(doses)) if len(doses) == 1 else "", "loss": value,
                                   "t_mean_loss": t_mean_loss, "i_mean_loss": i_mean_loss,
                                   "backward_loss": float(backward_loss.detach()), "grad_total": grad_total,
                                   "seconds": time.perf_counter() - start,
                                   "peak_vram_gb": torch.cuda.max_memory_allocated(device) / 1e9 if device.type == "cuda" else 0,
                                   **{f"grad_{key}": v for key, v in norms.items()}})
                        bar.update()
                        bar.set_postfix({"loss": f"{value:.4f}", "t_mean_loss": f"{t_mean_loss:.4f}",
                                         "i_mean_loss": f"{i_mean_loss:.4f}", "B": size,
                                         "grad": f"{grad_total:.2e}", "file": Path(unit["path"]).stem})
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
            if cfg.model_type == "full" and cfg.chem_skip_gate:
                info["epochs"][-1]["chem_skip_gates"] = [
                    float(layer.chem_skip_gate_logit.detach().sigmoid()) for layer in model.layers]
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
