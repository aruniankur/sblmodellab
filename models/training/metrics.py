"""Parameter groups and one durable CSV row per backward pass."""
from __future__ import annotations

import csv
import math
import os
from pathlib import Path



def parameter_group(name):
    parts = name.split(".")
    return f"layer{parts[1]}.{parts[2]}" if parts[0] == "layers" else parts[0]


def parameter_counts(model):
    groups = {}
    for name, parameter in model.named_parameters():
        key = parameter_group(name)
        groups[key] = groups.get(key, 0) + parameter.numel()
    return groups


def gradient_norms(model, groups, previous=None, multiplier=1):
    """Norms of this backward pass, before clipping or accumulation averaging.

    With accumulation, subtract the saved gradients from preceding micro-batches
    and undo loss / accumulation_count. This separates each pass's contribution.
    """
    squared = {key: 0.0 for key in groups}
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        grad = parameter.grad.detach().float()
        if previous is not None and name in previous:
            grad = grad - previous[name]
        norm2 = float(grad.square().sum()) * multiplier ** 2
        squared[parameter_group(name)] += norm2
    norms = {key: math.sqrt(value) for key, value in squared.items()}
    return norms, math.sqrt(sum(squared.values()))


class PassLog:
    def __init__(self, path, groups, committed_passes=0):
        self.path = Path(path)
        self.fields = ["epoch", "unit", "optimizer_step", "backward_pass", "micro_batch",
                       "accumulation_count", "file", "plate", "cell_line", "control_sample",
                       "target_sample", "drug", "dose", "loss", "backward_loss", "grad_total",
                       "seconds", "peak_vram_gb"] + [f"grad_{key}" for key in groups]
        if self.path.exists():
            # Drop rows newer than the checkpoint after a crash. Never duplicate steps.
            tmp = self.path.with_suffix(".csv.tmp")
            with self.path.open(newline="", encoding="utf-8") as src, tmp.open("w", newline="", encoding="utf-8") as dst:
                reader = csv.DictReader(src)
                if reader.fieldnames != self.fields:
                    raise ValueError("CSV schema differs from saved model parameter groups")
                writer = csv.DictWriter(dst, fieldnames=self.fields)
                writer.writeheader()
                for row in reader:
                    if int(row["backward_pass"]) <= committed_passes:
                        writer.writerow(row)
            os.replace(tmp, self.path)
        self.fh = self.path.open("a", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.fh, fieldnames=self.fields)
        if self.path.stat().st_size == 0:
            self.writer.writeheader()
            self.fh.flush()

    def write(self, values):
        self.writer.writerow(values)
        self.fh.flush()

    def close(self):
        self.fh.close()
