"""Isolated training-capacity trial. Invoked by models/benchmark.py, not by users."""
from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import torch

from models.tahoe_data import BackedFile
from models.training.checkpoints import write_json
from models.training.config import TrainingConfig
from models.training.data import sample_example, targets
from models.training.engine import device_for
from models.training.metrics import gradient_norms, parameter_counts
from models.training.modeling import create_model, load_chemistry, loss_for


def run_trial(cfg, *, warmup=1, steps=2, max_vram_fraction=0.9):
    """Measure complete optimizer units with the same accumulation/metrics as training.

    Warmup allocates AdamW state. Peak memory includes warmup and every measured
    unit, while timings exclude warmup. Only a small fixed set of real examples
    is prepared once, outside the measured compute section.
    """
    cfg.validate()
    device = device_for(cfg)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    result = {"dim": cfg.dim, "head_dim": cfg.head_dim, "conc_dim": cfg.conc_dim,
              "layers": cfg.layers, "heads": cfg.heads,
              "model_type": cfg.model_type, "normalization": cfg.normalization, "gene_statistics": cfg.gene_statistics, "gene_residual": cfg.gene_residual, "status": "error", "completed_steps": 0,
              "seconds_per_unit": None, "peak_allocated_gb": 0., "peak_reserved_gb": 0.,
              "loss": None, "grad_total": None, "reason": ""}
    model = optimizer = backed = None
    try:
        if device.type == "cuda":
            torch.cuda.empty_cache()
            free, total = torch.cuda.mem_get_info(device)
            result.update(device=torch.cuda.get_device_name(device), total_vram_gb=total / 1e9,
                          available_vram_gb=free / 1e9)
            torch.cuda.reset_peak_memory_stats(device)
        else:
            result.update(device="cpu", total_vram_gb=None, available_vram_gb=None)
        started = time.perf_counter()
        backed = BackedFile(cfg.file)
        if not 1 <= backed.plate <= 12:
            raise ValueError("Benchmark file must belong to plates 1-12")
        wells = targets(backed, cfg)
        if not wells or not backed.control_wells:
            raise ValueError("File needs DMSO context and eligible finite-coordinate target wells")
        count = min(cfg.batches_per_unit, len(wells))
        picks = rng.permutation(len(wells))[:count]
        examples = [sample_example(backed, wells[int(pick)].sample, cfg, rng, index)
                    for index, pick in enumerate(picks)]
        result.update(n_genes=backed.n_vars, file=str(Path(cfg.file).resolve()),
                      micro_batches_per_unit=count, data_seconds=time.perf_counter() - started)
        backed.close()
        backed = None
        store = load_chemistry(cfg)
        model = create_model(cfg, result["n_genes"], store.dim + 1 if store else None).to(device)
        model.train()
        groups = parameter_counts(model)
        result["parameters"] = sum(groups.values())
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        times = []
        for step in range(warmup + steps):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            start = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            values = []
            for micro, example in enumerate(examples):
                previous = ({name: p.grad.detach().clone() for name, p in model.named_parameters()
                             if p.grad is not None} if micro else None)
                loss = loss_for(model, example, cfg, device, store)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite training loss")
                (loss / count).backward()
                _, norm = gradient_norms(model, groups, previous, multiplier=count)
                del previous
                if not np.isfinite(norm):
                    raise FloatingPointError("Non-finite training gradients")
                values.append(float(loss.detach()))
            if cfg.grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip, error_if_nonfinite=True)
            optimizer.step()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - start
            if step >= warmup:
                times.append(elapsed)
                result["completed_steps"] += 1
            result.update(loss=float(np.mean(values)), grad_total=norm)
        result["seconds_per_unit"] = float(np.mean(times))
        result["seconds_per_backward"] = result["seconds_per_unit"] / count
        if device.type == "cuda":
            reserved = torch.cuda.max_memory_reserved(device) / 1e9
            result.update(peak_allocated_gb=torch.cuda.max_memory_allocated(device) / 1e9,
                          peak_reserved_gb=reserved)
            ceiling = min(result["total_vram_gb"], result["available_vram_gb"])
            if reserved > max_vram_fraction * ceiling:
                result.update(status="tight", reason="Completed, but below the requested VRAM headroom")
            else:
                result["status"] = "fits"
        else:
            result.update(status="cpu_completed", reason="CPU training completed; no GPU capacity claim")
    except torch.cuda.OutOfMemoryError as exc:
        result.update(status="oom", reason=str(exc)[:1000])
    except FloatingPointError as exc:
        result.update(status="nonfinite", reason=str(exc))
    except Exception as exc:
        result.update(status="error", reason=f"{type(exc).__name__}: {exc}")
    finally:
        if device.type == "cuda":
            result["peak_allocated_gb"] = torch.cuda.max_memory_allocated(device) / 1e9
            result["peak_reserved_gb"] = torch.cuda.max_memory_reserved(device) / 1e9
        if backed is not None:
            backed.close()
        del model, optimizer
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True)
    parser.add_argument("--result", required=True)
    args = parser.parse_args()
    request = json.loads(Path(args.request).read_text(encoding="utf-8"))
    try:
        result = run_trial(TrainingConfig(**request["config"]), **request["trial"])
    except Exception as exc:
        result = {"status": "error", "reason": f"{type(exc).__name__}: {exc}"}
    write_json(args.result, result)


if __name__ == "__main__":
    main()
