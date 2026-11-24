"""Optional TensorBoard logging for train.py / benchmark.py.

Writes to ``<out>/tb`` so ``tensorboard --logdir models/runs`` shows every run as
its own set of curves. Tags:

    loss/train                 per optimizer step
    loss/val                   per epoch (held-out cell lines)
    loss/epoch_train           per epoch mean
    loss/by_plate/plate06      the same step value, grouped by plate
    loss/by_cell_line/CVCL_x   ... and by cell line
    loss/by_drug/<drug>        only with --tb-per-drug (380 series)

    file/current               text: the h5ad file, the well and its drugs

    vram/allocated_gb          tensors alive right now
    vram/reserved_gb           what the caching allocator is holding
    vram/peak_gb               max allocated since start
    vram/peak_reserved_gb      max reserved since start
    vram/total_gb              the card's capacity, for scale

    perf/seconds_per_unit
    perf/wells_per_second

TensorBoard has no categorical axis, so "which file" is shown as a text panel
(``file/current``) and simultaneously folded into the grouped loss curves.
"""

from __future__ import annotations

import os

import torch

GB = 1e9


class TbLogger:
    """Thin wrapper; a no-op when ``log_dir`` is falsy."""

    def __init__(self, log_dir: str | None, strict: bool = True):
        self.writer = None
        self.log_dir = log_dir
        self._run_peak = 0.0
        self._run_peak_reserved = 0.0
        if not log_dir:
            return
        try:
            from torch.utils.tensorboard import SummaryWriter
        except Exception as exc:  # noqa: BLE001
            if strict:
                raise SystemExit(
                    "TensorBoard requested but not installed. Run:\n"
                    "    python -m pip install tensorboard"
                ) from exc
            print(
                f"  (tensorboard not installed: {type(exc).__name__} -- "
                f"continuing without logging; 'python -m pip install tensorboard')"
            )
            return
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=log_dir, flush_secs=10)

    @property
    def enabled(self) -> bool:
        return self.writer is not None

    # ------------------------------------------------------------------ #
    @staticmethod
    def step_vram() -> dict[str, float]:
        """High-water mark of the step that just finished.

        Call ``torch.cuda.reset_peak_memory_stats()`` *before* the step, otherwise
        this reports the peak of everything since the process started.
        """
        if not torch.cuda.is_available():
            return {}
        return {
            "step_peak_gb": torch.cuda.max_memory_allocated() / GB,
            "step_peak_reserved_gb": torch.cuda.max_memory_reserved() / GB,
            "total_gb": torch.cuda.get_device_properties(0).total_memory / GB,
        }

    def _log_vram(self, writer, step: int, vram: dict[str, float]) -> None:
        peak = vram.get("step_peak_gb")
        reserved = vram.get("step_peak_reserved_gb")
        if peak is not None:
            self._run_peak = max(self._run_peak, peak)
            writer.add_scalar("vram/step_peak_gb", peak, step)
            writer.add_scalar("vram/run_peak_gb", self._run_peak, step)
        if reserved is not None:
            self._run_peak_reserved = max(self._run_peak_reserved, reserved)
            writer.add_scalar("vram/step_peak_reserved_gb", reserved, step)
            writer.add_scalar("vram/run_peak_reserved_gb", self._run_peak_reserved, step)
        total = vram.get("total_gb")
        if total is not None:
            writer.add_scalar("vram/total_gb", total, step)

    def log_step(
        self,
        step: int,
        *,
        loss: float,
        plate: int,
        cell_line: str,
        drugs: list[str],
        seconds: float,
        wells: int,
        vram: dict[str, float] | None = None,
        per_drug: bool = False,
    ) -> None:
        writer = self.writer
        if writer is None:
            return

        writer.add_scalar("loss/train", loss, step)
        writer.add_scalar(f"loss/by_plate/plate{plate:02d}", loss, step)
        writer.add_scalar(f"loss/by_cell_line/{cell_line}", loss, step)
        if per_drug:
            for drug in drugs:
                writer.add_scalar(f"loss/by_drug/{drug}", loss, step)

        writer.add_text(
            "file/current",
            f"plate{plate}_{cell_line}.h5ad  |  drugs: {', '.join(drugs)}",
            step,
        )

        if seconds > 0:
            writer.add_scalar("perf/seconds_per_unit", seconds, step)
            writer.add_scalar("perf/wells_per_second", wells / seconds, step)

        if vram:
            self._log_vram(writer, step, vram)

    def log_epoch(self, epoch: int, train_loss: float, val_loss: float | None) -> None:
        writer = self.writer
        if writer is None:
            return
        writer.add_scalar("loss/epoch_train", train_loss, epoch)
        if val_loss is not None:
            writer.add_scalar("loss/val", val_loss, epoch)

    def log_scalars(self, values: dict[str, float], step: int) -> None:
        """Several scalar tags at one step."""
        if self.writer is None:
            return
        for tag, value in values.items():
            self.writer.add_scalar(tag, value, step)

    def log_file(self, step: int, text: str) -> None:
        """One-off text entry, e.g. which h5ad a benchmark run is probing."""
        if self.writer is not None:
            self.writer.add_text("file/current", text, step)

    def log_benchmark(
        self,
        step: int,
        *,
        loss: float,
        seconds: float,
        vram: dict[str, float] | None = None,
    ) -> None:
        writer = self.writer
        if writer is None:
            return
        writer.add_scalar("loss/benchmark", loss, step)
        writer.add_scalar("perf/seconds_per_step", seconds, step)
        if vram:
            self._log_vram(writer, step, vram)

    def close(self) -> None:
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()
            self.writer = None
