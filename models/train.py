"""Training pipeline for the Tahoe-100M virtual-cell model.

One training example is a **well** (see ``tahoe_data.py``)::

    control : 256 cells from a DMSO well of the file   -> (1, 50266, 256)
    chem    : per-atom MolFormer vectors of the drug   -> (1, A, 769)
    coords  : 128 target cells' UMAP positions         -> (1, 128, 2)
    target  : those 128 cells' log1p expression        -> (1, 128, 50266)

Epoch layout (as specified): every cell line visits ``--passes-per-cell-line``
different plates; each (cell line, plate) unit produces ``--batches-per-unit``
micro-batches -- one per DMSO control well, each with its own random drug -- and
the unit's micro-batches share one optimizer step. Files are opened in groups of
``--load-group`` and never mixed across plates.

    python models/train.py --dry-run                 # check the data path only
    python models/train.py --epochs 1 --max-steps 4  # smoke run
    python models/train.py --epochs 60               # real run

Memory: with G = 50,266 the model needs two memory-only knobs, both optional and
numerically exact (``model.py`` is otherwise untouched):

* ``--query-chunk``  splits the part-1 gene-query axis. Without it the math
  backend materialises ``(1, 1, G, G)`` fp32 = ~40 GB.
* ``--readout-chunk`` splits the cell axis of the readout. The full
  ``(1, 128, G, dim+2)`` tensor is 13.4 GB at ``dim=512``.
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
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

if __package__ in (None, ""):  # allow `python models/train.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.model import PerturbationModel, perturbation_loss
from models.molformer_atoms import AtomEmbeddingStore
from models.tb import TbLogger
from models.tahoe_data import (
    DEFAULT_ROOT,
    BackedFile,
    FileCache,
    build_example,
    build_file_index,
    grid,
)

PLATE_FILE = "plate{plate}_{cell_line}.h5ad"


# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Tahoe-100M per-well perturbation training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    g = p.add_argument_group("data")
    g.add_argument("--root", default=DEFAULT_ROOT)
    g.add_argument("--exclude-plates", default="13,14",
                   help="held-out plates, never opened")
    g.add_argument("--limit-files", type=int, default=0,
                   help="debug: use only the first N files")
    g.add_argument("--n-control", type=int, default=256, help="DMSO cells per batch")
    g.add_argument("--n-target", type=int, default=128, help="predicted cells per batch")
    g.add_argument("--min-cells", type=int, default=128,
                   help="a drug well needs this many cells to be usable "
                        "(below n-target the cells are sampled with replacement)")
    g.add_argument("--scale-exponent", type=float, default=0.5,
                   help="perturbation_loss z: divides by n_genes ** z")

    g.add_argument("--atom-cache", default=None, help="defaults to models/cache/atom_embeddings.pt")

    g = p.add_argument_group("epoch layout")
    g.add_argument("--epochs", type=int, default=60)
    g.add_argument("--passes-per-cell-line", type=int, default=5,
                   help="distinct plates a cell line visits per epoch")
    g.add_argument("--batches-per-unit", type=int, default=2,
                   help="micro-batches (= distinct drugs) per (cell line, plate)")
    g.add_argument("--load-group", type=int, default=5,
                   help="files held open at once")
    g.add_argument("--file-cache", type=int, default=6, help="backed-file LRU size")

    g = p.add_argument_group("model")
    g.add_argument("--dim", type=int, default=512)
    g.add_argument("--heads", type=int, default=8)
    g.add_argument("--layers", type=int, default=1)
    g.add_argument("--conc-dim", type=int, default=32)
    g.add_argument("--dropout", type=float, default=0.1)
    g.add_argument("--chem-tokens", choices=["all", "atoms"], default="all",
                   help="stage-1 keys per drug: every MolFormer token (atoms + "
                        "structure tokens) or atoms only")
    g.add_argument("--num-genes", type=int, default=0, help="0 = read from the h5ad files")

    g = p.add_argument_group("optimisation")
    g.add_argument("--lr", type=float, default=1e-4)
    g.add_argument("--weight-decay", type=float, default=0.01)
    g.add_argument("--grad-clip", type=float, default=1.0)
    g.add_argument("--amp", choices=["off", "bf16", "fp16"], default="bf16",
                   help="autocast dtype. bf16 is the supported default; fp16 is "
                        "NOT usable on this model (every step overflows, see "
                        "models/testing/amp_benchmark.py)")
    g.add_argument("--tf32", dest="tf32", action="store_true", default=True,
                   help="TF32 matmuls for the ops that stay fp32 (costs nothing)")
    g.add_argument("--no-tf32", dest="tf32", action="store_false")
    g.add_argument("--query-chunk", type=int, default=4096,
                   help="part-1 query chunk; 0 disables")
    g.add_argument("--readout-chunk", type=int, default=16,
                   help="readout cell chunk; 0 disables")

    g = p.add_argument_group("validation / io")
    g.add_argument("--val-cell-lines", type=int, default=2,
                   help="cell lines held out of training and used for validation")
    g.add_argument("--val-plates", type=int, default=2)
    g.add_argument("--val-every", type=int, default=1, help="validate every N epochs")
    g.add_argument("--out", default=str(Path(__file__).resolve().parent / "runs" / "default"))
    g.add_argument("--tb", dest="tb", action="store_true", default=True,
                   help="write TensorBoard logs to <out>/tb")
    g.add_argument("--no-tb", dest="tb", action="store_false")
    g.add_argument("--tb-dir", default="", help="override the TensorBoard directory")
    g.add_argument("--tb-per-drug", action="store_true",
                   help="also log loss/by_drug/<drug> (380 series)")
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--log-every", type=int, default=10)
    g.add_argument("--save-every", type=int, default=50, help="steps; 0 disables")
    g.add_argument("--max-steps", type=int, default=0, help="debug: stop after N steps")
    g.add_argument("--resume", default="")
    g.add_argument("--dry-run", action="store_true", help="build a batch, skip the model")
    return p.parse_args(argv)


def resolve_config(cfg: argparse.Namespace) -> argparse.Namespace:
    cfg.exclude_plates = tuple(
        int(x) for x in str(cfg.exclude_plates).replace(" ", "").split(",") if x
    )
    cfg.query_chunk = cfg.query_chunk or None
    cfg.readout_chunk = cfg.readout_chunk or None
    if cfg.atom_cache is None:
        cfg.atom_cache = str(
            Path(__file__).resolve().parent / "cache" / "atom_embeddings.pt"
        )
    return cfg


# --------------------------------------------------------------------------- #
# epoch planning
# --------------------------------------------------------------------------- #
@dataclass
class Plan:
    plates: list[int]
    cell_lines: list[str]
    val_cell_lines: list[str]
    paths: dict[tuple[int, str], str] = field(default_factory=dict)

    def path(self, plate: int, cell_line: str) -> str:
        return self.paths[(plate, cell_line)]

    @property
    def train_cell_lines(self) -> list[str]:
        return [c for c in self.cell_lines if c not in set(self.val_cell_lines)]


def make_plan(cfg: argparse.Namespace, rng: np.random.Generator) -> Plan:
    index = build_file_index(cfg.root, cfg.exclude_plates)
    if cfg.limit_files:
        index = index[: cfg.limit_files]
    plates, cell_lines = grid(index)

    shuffled = list(cell_lines)
    rng.shuffle(shuffled)
    n_val = min(cfg.val_cell_lines, max(0, len(shuffled) - 1))
    val = sorted(shuffled[:n_val])

    return Plan(
        plates=plates,
        cell_lines=cell_lines,
        val_cell_lines=val,
        paths={(f.plate, f.cell_line): f.path for f in index},
    )


def iter_epoch(plan: Plan, cfg: argparse.Namespace, rng: np.random.Generator):
    """Yield ``(plate, [cell lines])`` groups, grouped by plate for I/O locality.

    Every cell line visits ``min(passes_per_cell_line, n_plates)`` distinct plates
    per epoch, so a full epoch is
    ``n_cell_lines * passes * batches_per_unit`` micro-batches.
    """
    by_plate: dict[int, list[str]] = defaultdict(list)
    for cell_line in plan.train_cell_lines:
        k = min(cfg.passes_per_cell_line, len(plan.plates))
        for plate in rng.choice(plan.plates, size=k, replace=False):
            by_plate[int(plate)].append(cell_line)

    for plate in rng.permutation(plan.plates):
        cell_lines = list(by_plate[int(plate)])
        rng.shuffle(cell_lines)
        for start in range(0, len(cell_lines), cfg.load_group):
            group = cell_lines[start : start + cfg.load_group]
            if group:
                yield int(plate), group


# --------------------------------------------------------------------------- #
# one (cell line, plate) unit
# --------------------------------------------------------------------------- #
def run_unit(
    model: PerturbationModel,
    store: AtomEmbeddingStore,
    cache: FileCache,
    backed: BackedFile,
    cfg: argparse.Namespace,
    rng: np.random.Generator,
    device: torch.device,
    amp: contextlib.AbstractContextManager,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler | None = None,
) -> tuple[float, list[str]] | None:
    """Forward/backward ``cfg.batches_per_unit`` wells; step if ``optimizer`` given.

    Returns ``(mean_loss, [drug names used])``.
    """
    control_wells = backed.control_wells
    drug_wells = backed.drug_wells(cfg.min_cells)
    if not control_wells or not drug_wells:
        return None

    k = min(cfg.batches_per_unit, len(drug_wells))
    picks = rng.choice(len(drug_wells), size=k, replace=False)

    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)

    total = 0.0
    drugs_used: list[str] = []
    for i, pick in enumerate(picks):
        control_well = control_wells[i % len(control_wells)]  # rotate control samples
        example = build_example(
            backed,
            control_well,
            drug_wells[int(pick)],
            cfg.n_control,
            cfg.n_target,
            rng,
        )
        drugs_used.append(example.drug)

        control = example.control.unsqueeze(0).to(device, non_blocking=True)
        target = example.target.unsqueeze(0).to(device, non_blocking=True)
        coords = example.coords.unsqueeze(0).to(device, non_blocking=True)
        chem, chem_mask = store.collate(
            [example.drug], [example.concentration], device, mode=cfg.chem_tokens
        )

        with amp:
            predicted = model(
                control,
                chem,
                coords,
                chem_mask,
                query_chunk=cfg.query_chunk,
                readout_chunk=cfg.readout_chunk,
            )

        loss = perturbation_loss(predicted.float(), target, cfg.scale_exponent) / k
        if optimizer is not None:
            if scaler is not None:
                # fp16 only: scale before backward, unscale before clip/step,
                # otherwise gradients below ~6e-8 flush to zero.
                scaler.scale(loss).backward()
            else:
                loss.backward()
        total += float(loss.detach())

    if optimizer is not None:
        if scaler is not None:
            scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

    return total, drugs_used


def validate(
    model: PerturbationModel,
    store: AtomEmbeddingStore,
    cache: FileCache,
    plan: Plan,
    cfg: argparse.Namespace,
    device: torch.device,
    amp: contextlib.AbstractContextManager,
    seed: int,
) -> float | None:
    if not plan.val_cell_lines:
        return None

    model.eval()
    rng = np.random.default_rng(seed)
    plates = plan.plates[: cfg.val_plates]

    losses = []
    with torch.no_grad():
        for cell_line in plan.val_cell_lines:
            for plate in plates:
                backed = cache.get(plan.path(int(plate), cell_line))
                result = run_unit(
                    model, store, cache, backed, cfg, rng, device, amp, optimizer=None
                )
                if result is not None:
                    losses.append(result[0])
    model.train()
    return float(np.mean(losses)) if losses else None


# --------------------------------------------------------------------------- #
# checkpointing
# --------------------------------------------------------------------------- #
def save_checkpoint(
    path: str, model, optimizer, epoch: int, step: int, cfg,
    epoch_done: bool = False, scaler=None,
) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "epoch_done": epoch_done,
        "step": step,
        "config": vars(cfg),
    }
    if scaler is not None:
        payload["scaler"] = scaler.state_dict()
    torch.save(payload, path)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> None:
    cfg = resolve_config(parse_args(argv))

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    # bf16 autocast covers every matmul/attention; tf32 only shapes the few ops
    # that stay fp32. It costs nothing and avoids the slower cutlass fp32 path.
    torch.backends.cuda.matmul.allow_tf32 = cfg.tf32
    torch.backends.cudnn.allow_tf32 = cfg.tf32

    rng = np.random.default_rng(cfg.seed)
    plan = make_plan(cfg, rng)
    print(
        f"plates={len(plan.plates)} {plan.plates[0]}..{plan.plates[-1]}  "
        f"cell_lines={len(plan.cell_lines)}  val_cell_lines={plan.val_cell_lines}",
        flush=True,
    )

    store = AtomEmbeddingStore(cfg.atom_cache)
    print(f"atom cache: {len(store)} drugs, dim={store.dim}", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # gene count comes from the data, never guessed
    probe = BackedFile(plan.path(plan.plates[0], plan.cell_lines[0]), cfg.min_cells)
    n_genes = probe.n_vars
    probe.close()
    if cfg.num_genes and cfg.num_genes != n_genes:
        raise ValueError(f"--num-genes {cfg.num_genes} != file var count {n_genes}")
    print(f"genes={n_genes}", flush=True)

    if cfg.dry_run:
        backed = BackedFile(plan.path(plan.plates[0], plan.cell_lines[0]), cfg.min_cells)
        example = build_example(
            backed,
            backed.control_wells[0],
            backed.drug_wells(cfg.min_cells)[0],
            cfg.n_control,
            cfg.n_target,
            rng,
        )
        chem, mask = store.collate(
            [example.drug], [example.concentration], mode=cfg.chem_tokens
        )
        print(f"control  {tuple(example.control.shape)}   (G, n_control)")
        print(f"target   {tuple(example.target.shape)}")
        print(f"coords   {tuple(example.coords.shape)}")
        print(f"chem     {tuple(chem.shape)}  dose={example.concentration:g} -> "
              f"log10={float(chem[0, 0, -1]):.3f}")
        print(f"mask     {tuple(mask.shape)}  real={int(mask.sum())} "
              f"({store.n_atoms(example.drug)} atoms of "
              f"{store.n_tokens(example.drug)} tokens, mode={cfg.chem_tokens})")
        print(f"drug     {example.drug}")
        backed.close()
        return

    chem_dim = store.dim + 1  # + log10(dose)
    if cfg.dim % cfg.heads:
        raise ValueError("--dim must be divisible by --heads")
    if cfg.dim // cfg.heads > 128:
        raise ValueError(
            f"head_dim={cfg.dim // cfg.heads} > 128: the gene self-attention over "
            f"{n_genes} tokens would fall back to the math backend and OOM. "
            "Increase --heads."
        )

    model = PerturbationModel(
        dim=cfg.dim,
        conc_dim=cfg.conc_dim,
        num_heads=cfg.heads,
        dropout=cfg.dropout,
        num_layers=cfg.layers,
        num_of_genes=n_genes,
        chem_dim=chem_dim,
        cell_coordi_dim=2,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )

    # --- mixed precision --------------------------------------------------- #
    # bf16 needs no gradient scaling: its 8-bit exponent has the same range as
    # fp32, so the large gradients this model produces cannot overflow.
    # fp16 does overflow here (measured: |g| = inf on every step, GradScaler
    # halves 32768 -> 1024 and skips 6/6 optimizer steps). Kept only so the
    # failure is reproducible with --amp fp16.
    if cfg.amp == "fp16" and device.type != "cuda":
        raise RuntimeError(
            "fp16 mixed precision needs a CUDA device; torch.amp does not support "
            "float16 autocast on CPU."
        )
    amp_dtype = {"off": None, "bf16": torch.bfloat16, "fp16": torch.float16}[cfg.amp]
    amp = (
        contextlib.nullcontext()
        if amp_dtype is None
        else torch.autocast(
            device_type="cuda" if device.type == "cuda" else "cpu", dtype=amp_dtype
        )
    )
    scaler = torch.amp.GradScaler("cuda") if cfg.amp == "fp16" else None

    start_epoch, step = 0, 0
    if cfg.resume:
        state = torch.load(cfg.resume, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        if "scaler" in state:
            scaler.load_state_dict(state["scaler"])
        start_epoch = int(state.get("epoch", 0))
        if state.get("epoch_done", False):
            start_epoch += 1
        step = int(state.get("step", 0))
        print(f"resumed {cfg.resume} @ epoch {start_epoch} step {step}", flush=True)

    cache = FileCache(maxsize=max(cfg.file_cache, cfg.load_group), min_cells=1)
    os.makedirs(cfg.out, exist_ok=True)
    with open(os.path.join(cfg.out, "config.json"), "w") as fh:
        json.dump(vars(cfg), fh, indent=2)

    tb = TbLogger(
        (cfg.tb_dir or os.path.join(cfg.out, "tb")) if cfg.tb else None
    )
    if tb.enabled:
        print(f"tensorboard: {tb.log_dir}  (tensorboard --logdir {cfg.out})", flush=True)

    log_path = os.path.join(cfg.out, "log.csv")
    new_log = not os.path.exists(log_path)
    log_fh = open(log_path, "a", newline="")
    log = csv.writer(log_fh)
    if new_log:
        log.writerow(["epoch", "step", "unit", "loss", "val_loss", "lr", "seconds", "scale"])

    units_per_epoch = len(plan.train_cell_lines) * min(
        cfg.passes_per_cell_line, len(plan.plates)
    )
    print(
        f"epoch: {units_per_epoch} units = "
        f"{units_per_epoch * cfg.batches_per_unit} micro-batches "
        f"({len(plan.train_cell_lines)} cell lines x "
        f"{min(cfg.passes_per_cell_line, len(plan.plates))} plates)",
        flush=True,
    )
    print(
        f"model: dim={cfg.dim} heads={cfg.heads} layers={cfg.layers} "
        f"chem_dim={chem_dim} params={sum(p.numel() for p in model.parameters())/1e6:.2f}M",
        flush=True,
    )

    model.train()
    best_val = math.inf
    stopped = False

    for epoch in range(start_epoch, cfg.epochs):
        epoch_start = time.time()
        unit_index = 0
        running = 0.0
        skipped = 0
        n_units = 0

        for plate, group in iter_epoch(plan, cfg, rng):
            for cell_line in group:
                unit_index += 1
                backed = cache.get(plan.path(plate, cell_line))
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats()  # per-unit high-water mark
                t0 = time.time()
                result = run_unit(
                    model, store, cache, backed, cfg, rng, device, amp, optimizer,
                    scaler=scaler,
                )
                if result is None:
                    skipped += 1
                    continue

                loss, drugs = result
                step += 1
                n_units += 1
                running += loss
                elapsed = time.time() - t0

                tb.log_step(
                    step,
                    loss=loss,
                    plate=plate,
                    cell_line=cell_line,
                    drugs=drugs,
                    seconds=elapsed,
                    wells=min(cfg.batches_per_unit, len(backed.drug_wells(cfg.min_cells))),
                    vram=TbLogger.step_vram(),
                    per_drug=cfg.tb_per_drug,
                )

                if step % cfg.log_every == 0:
                    scale_note = (
                        f" fp16-scale {scaler.get_scale():.0f}"
                        if scaler is not None
                        else ""
                    )
                    gpu_note = (
                        f"gpu {torch.cuda.max_memory_allocated()/1e9:.1f}GB"
                        if device.type == "cuda"
                        else ""
                    )
                    print(
                        f"epoch {epoch} step {step} unit {unit_index} "
                        f"plate {plate} {cell_line} loss {loss:.5f} "
                        f"{elapsed:.2f}s/it {gpu_note}"
                        f"{scale_note}",
                        flush=True,
                    )

                log.writerow(
                    [epoch, step, f"{plate}:{cell_line}", f"{loss:.6f}", "", cfg.lr,
                     f"{elapsed:.3f}",
                     "" if scaler is None else f"{scaler.get_scale():.0f}"]
                )

                if cfg.save_every and step % cfg.save_every == 0:
                    save_checkpoint(
                        os.path.join(cfg.out, "last.pt"), model, optimizer, epoch, step,
                        cfg, scaler=scaler,
                    )
                if cfg.max_steps and step >= cfg.max_steps:
                    stopped = True
                    break
            if stopped:
                break
        if stopped:
            break

        train_loss = running / max(n_units, 1)
        val_loss = None
        if cfg.val_every and (epoch + 1) % cfg.val_every == 0:
            val_loss = validate(model, store, cache, plan, cfg, device, amp, cfg.seed + epoch)

        epoch_seconds = time.time() - epoch_start
        tb.log_epoch(epoch, train_loss, val_loss)
        print(
            f"== epoch {epoch}: train {train_loss:.5f} "
            f"val {val_loss if val_loss is None else f'{val_loss:.5f}'} "
            f"units {n_units} skipped {skipped} {epoch_seconds/60:.1f} min",
            flush=True,
        )
        log_fh.flush()

        if val_loss is not None and val_loss < best_val:
            best_val = val_loss
            save_checkpoint(
                os.path.join(cfg.out, "best.pt"), model, optimizer, epoch, step, cfg,
                epoch_done=True, scaler=scaler,
            )
        save_checkpoint(
            os.path.join(cfg.out, "last.pt"), model, optimizer, epoch, step, cfg,
            epoch_done=True, scaler=scaler,
        )

    log_fh.close()
    cache.close()
    tb.close()
    print(f"done: {step} steps, checkpoints in {cfg.out}", flush=True)


if __name__ == "__main__":
    main()
