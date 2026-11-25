"""Single-file capability probe: how much can *this* machine train at once?

Loads ONE ``plate{N}_{CELL_LINE}.h5ad`` file and actually trains on it, so the
numbers are real (forward + backward + AdamW step), not a synthetic matmul.

You choose everything that shapes memory except the gene count, which always
comes from the file:

    dim (width) | heads | layers | conc_dim | batch (wells stacked per step)

    python models/benchmark.py                      # asks interactively
    python models/benchmark.py --file plate6_CVCL_1125.h5ad \
        --dim 512 --heads 8 --layers 1 --conc-dim 32 --batch 1

Interactive prompts default to the last used values, so pressing Enter accepts
them. Non-interactive runs (piped stdin) automatically take the defaults.

Reports per step: data time, compute time, loss, peak GPU allocation and peak
GPU reservation (what the allocator actually holds). On CUDA OOM it stops and
prints exactly which knob to lower. Any plate can be probed, including the
held-out 13/14 -- this script never trains the real model.

``dim / heads`` must be <= 128. Above that the gene self-attention over 50,266
tokens cannot use a fused kernel and falls back to a materialised
``(B, heads, G, G)`` score matrix, which is a kernel limitation rather than a
measurement of your hardware -- so it is rejected with a suggestion instead.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.cli import ask, choose_file
from models.model import PerturbationModel, perturbation_loss
from models.molformer_atoms import AtomEmbeddingStore
from models.tahoe_data import DEFAULT_ROOT, BackedFile, build_example

GB = 1e9


# --------------------------------------------------------------------------- #
# batches
# --------------------------------------------------------------------------- #
def build_batch(backed, store, drugs, controls, cfg, rng, offset):
    """``cfg.batch`` wells stacked into one forward pass."""
    examples = []
    for j in range(cfg.batch):
        well = drugs[(offset + j) % len(drugs)]
        control = controls[j % len(controls)]
        examples.append(
            build_example(
                backed, control, well, cfg.n_control, cfg.n_target, rng
            )
        )

    device = cfg.device
    chem, mask = store.collate(
        [e.drug for e in examples], [e.concentration for e in examples], device,
        mode=cfg.chem_tokens,
    )
    return (
        torch.stack([e.control for e in examples]).to(device),
        chem,
        mask,
        torch.stack([e.coords for e in examples]).to(device),
        torch.stack([e.target for e in examples]).to(device),
        [e.drug for e in examples],
    )


def human(x: float) -> str:
    return f"{x / GB:.2f} GB"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="train one h5ad file at a chosen size to find this machine's limit",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--file", default=None,
                   help="file name or path; omit to pick interactively")
    p.add_argument("--root", default=DEFAULT_ROOT)
    p.add_argument("--dim", type=int, default=None, help="model width")
    p.add_argument("--heads", type=int, default=None)
    p.add_argument("--layers", type=int, default=None,
                   help="stacked ExternalPerturbAttention blocks")
    p.add_argument("--conc-dim", type=int, default=None,
                   help="co-expression bottleneck width")
    p.add_argument("--batch", type=int, default=None,
                   help="wells stacked in one forward pass (= memory knob)")
    p.add_argument("--steps", type=int, default=5, help="optimizer steps to run")
    p.add_argument("--warmup", type=int, default=1,
                   help="untimed steps (kernel autotune / allocator growth)")
    p.add_argument("--time-budget", type=float, default=90.0,
                   help="give up when one step exceeds this many seconds "
                        "(a box that spills to system RAM is not a result)")
    p.add_argument("--n-control", type=int, default=256)
    p.add_argument("--n-target", type=int, default=128)
    p.add_argument("--min-cells", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--amp", choices=["off", "bf16"], default="bf16")
    p.add_argument("--chem-tokens", choices=["all", "atoms"], default="all")
    p.add_argument("--query-chunk", type=int, default=None,
                   help="omit for automatic (~1 GB of scores), 0 to disable")
    p.add_argument("--readout-chunk", type=int, default=None,
                   help="omit for automatic (~1 GB of activations), 0 to disable")
    p.add_argument("--atom-cache", default=None)
    args = p.parse_args(argv)

    # ---- file ------------------------------------------------------------
    path = choose_file(args.root, args.file)
    name = os.path.basename(path)
    print(f"\nfile: {name}")

    # ---- the five knobs --------------------------------------------------
    print("\nmodel shape (genes are fixed to the file):")
    dim = args.dim if args.dim is not None else ask("  dim (model width)", 512)
    heads = args.heads if args.heads is not None else ask("  heads", 8)

    # validate before asking the rest, so a bad shape fails on the spot
    if dim % heads:
        raise SystemExit(f"dim {dim} is not divisible by heads {heads}")
    head_dim = dim // heads
    if head_dim > 128:
        raise SystemExit(
            f"head_dim = dim/heads = {dim}/{heads} = {head_dim} > 128.\n"
            f"The gene self-attention over 50,266 tokens would fall back to the\n"
            f"math backend and try to allocate (B, {heads}, 50266, 50266) scores --\n"
            f"that is a kernel limit, not your PC's limit.\n"
            f"Use at least --heads {max(1, -(-dim // 128))} (e.g. --heads "
            f"{-(-dim // 128)} for dim={dim})."
        )

    layers = args.layers if args.layers is not None else ask("  layers", 1)
    conc_dim = args.conc_dim if args.conc_dim is not None else ask("  conc_dim", 32)
    batch = args.batch if args.batch is not None else ask("  batch (wells per step)", 1)

    # write the resolved values back so build_batch() sees them
    args.dim, args.heads, args.layers = dim, heads, layers
    args.conc_dim, args.batch = conc_dim, batch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device

    # ---- guards ----------------------------------------------------------
    # (dim/heads were validated as soon as they were chosen)

    store = AtomEmbeddingStore(
        args.atom_cache
        or str(Path(__file__).resolve().parent / "cache" / "atom_embeddings.pt")
    )

    backed = BackedFile(path, args.min_cells)
    n_genes = backed.n_vars
    controls = backed.control_wells
    drugs = backed.drug_wells(args.min_cells)
    if not controls or not drugs:
        raise SystemExit(
            f"this file has {len(controls)} control wells and {len(drugs)} drug "
            f"wells with >= {args.min_cells} cells -- nothing to train on"
        )

    print(f"\n  genes     : {n_genes}   (from the file, not asked)")
    print(f"  control wells: {len(controls)}   drug wells: {len(drugs)}")
    print(f"  device    : {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu'}"
          + (f"  {human(torch.cuda.get_device_properties(0).total_memory)} total"
             if device.type == "cuda" else ""))

    # ---- build -----------------------------------------------------------
    model = PerturbationModel(
        dim=dim,
        conc_dim=conc_dim,
        num_heads=heads,
        dropout=0.1,
        num_layers=layers,
        num_of_genes=n_genes,
        chem_dim=store.dim + 1,
        cell_coordi_dim=2,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    amp_dtype = {"off": None, "bf16": torch.bfloat16}[args.amp]
    amp = (
        contextlib.nullcontext()
        if amp_dtype is None
        else torch.autocast(device_type=device.type, dtype=amp_dtype)
    )

    static = n_params * 4 * 3  # fp32 weights + 2 AdamW moments
    print(f"\n  params    : {n_params / 1e6:.2f} M")
    print(f"  static    : {human(static)} (weights + AdamW moments, fp32)")

    # The chunk sizes are chosen so a single chunk costs ~1 GB: with a fixed
    # chunk, a bigger dim/batch would fail on the probe's own bookkeeping rather
    # than on the architecture, which is what we are trying to measure.
    if args.query_chunk is None or args.readout_chunk is None:
        auto_query = max(256, int(1e9 / (batch * n_genes * 4)))
        auto_readout = max(1, int(1e9 / (batch * n_genes * (dim + 2) * 4)))
        query_chunk = args.query_chunk if args.query_chunk is not None else auto_query
        readout_chunk = args.readout_chunk if args.readout_chunk is not None else auto_readout
        if args.query_chunk is None or args.readout_chunk is None:
            print(f"  chunks    : query {query_chunk}, readout {readout_chunk} (auto, ~1 GB each)")
    else:
        query_chunk, readout_chunk = args.query_chunk, args.readout_chunk
    query_chunk = query_chunk or None
    readout_chunk = readout_chunk or None

    rng = np.random.default_rng(0)
    model.train()

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    print(f"\nrunning {args.warmup} warmup + {args.steps} timed steps "
          f"(batch={batch} well(s), {args.n_control} control + {args.n_target} target cells)\n")
    print(f"{'step':>4} {'data s':>7} {'step s':>7} {'loss':>10} {'peak alloc':>11} "
          f"{'peak reserved':>14}")

    times, losses = [], []
    status = "ok"
    total = args.warmup + args.steps
    peak_alloc_all = 0.0
    peak_res_all = 0.0

    for step in range(total):
        # per-step high-water mark, so vram/* is the cost of this shape and not
        # the peak since the process started
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        try:
            t0 = time.time()
            control, chem, mask, coords, target, drug_names = build_batch(
                backed, store, drugs, controls, args, rng, step * batch
            )
            data_time = time.time() - t0

            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()

            optimizer.zero_grad(set_to_none=True)
            with amp:
                predicted = model(
                    control,
                    chem,
                    coords,
                    mask,
                    query_chunk=query_chunk,
                    readout_chunk=readout_chunk,
                )
            loss = perturbation_loss(predicted.float(), target, 0.5)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if device.type == "cuda":
                torch.cuda.synchronize()
            step_time = time.time() - t0

            if device.type == "cuda":
                peak_alloc_all = max(
                    peak_alloc_all, torch.cuda.max_memory_allocated() / GB
                )
                peak_res_all = max(
                    peak_res_all, torch.cuda.max_memory_reserved() / GB
                )

        except torch.cuda.OutOfMemoryError:
            status = "oom"
            torch.cuda.empty_cache()
            print(f"\n  CUDA OUT OF MEMORY on step {step}")
            print(f"  peak allocated before failure: "
                  f"{human(torch.cuda.max_memory_allocated())}")
            print(f"\n  lower one of these and re-run:")
            print(f"    --batch {max(1, batch // 2)}      (currently {batch})")
            print(f"    --dim {max(64, dim // 2)}       (currently {dim})")
            print(f"    --layers {max(1, layers - 1)}     (currently {layers})")
            print(f"    --readout-chunk 8   (currently {readout_chunk})")
            print(f"    --query-chunk 2048  (currently {query_chunk})")
            break

        if step_time > args.time_budget:
            status = "slow"
            peak = torch.cuda.max_memory_allocated() / GB if device.type == "cuda" else 0.0
            print(f"\n  step {step} took {step_time:.1f} s, over the "
                  f"{args.time_budget:.0f} s budget (peak {peak:.2f} GB)")
            print("  this shape is either oversubscribed (spilling to system RAM) or "
                  "far too large; stopping instead of wasting your time.")
            print("  lower --batch / --dim / --layers, or pass a bigger "
                  "--time-budget to measure it anyway")
            break

        if step >= args.warmup:
            times.append(step_time)
            losses.append(float(loss.detach()))
            alloc = torch.cuda.max_memory_allocated() / GB if device.type == "cuda" else 0
            reserved = torch.cuda.max_memory_reserved() / GB if device.type == "cuda" else 0
            print(f"{step - args.warmup:>4} {data_time:>7.2f} {step_time:>7.2f} "
                  f"{float(loss.detach()):>10.4f} {alloc:>10.2f}G {reserved:>13.2f}G")

    backed.close()

    # ---- verdict ---------------------------------------------------------
    if status == "oom":
        return 2

    def fitted_verdict(peak_res: float, total_mem: float) -> str:
        if total_mem <= 0:
            return "cpu"
        share = peak_res / total_mem
        if share > 1.0:
            return (f"DOES NOT FIT -- {100 * share:.0f}% of the card, spilling to "
                    f"system RAM (timings below are meaningless)")
        if share > 0.92:
            return f"tight -- {100 * share:.0f}% of the card, no room for a bigger batch"
        return f"fits -- {100 * share:.0f}% of the card"

    if times:
        mean_step = float(np.mean(times))
        per_well = mean_step / batch
        epochs_wells = 480.0  # the real trainer's wells per epoch
        epoch_min = epochs_wells * per_well / 60.0
        peak_alloc = peak_alloc_all if device.type == "cuda" else 0
        peak_res = peak_res_all if device.type == "cuda" else 0
        total_mem = (
            torch.cuda.get_device_properties(0).total_memory / GB
            if device.type == "cuda" else 0
        )

        print(f"\n{'='*70}")
        print(f"  dim {dim} | heads {heads} (head_dim {head_dim}) | layers {layers} "
              f"| conc_dim {conc_dim} | batch {batch}")
        print(f"  params          {n_params / 1e6:.2f} M")
        print(f"  mean step       {mean_step:.2f} s  ({per_well:.2f} s per well)")
        print(f"  loss            {losses[0]:.4f} -> {losses[-1]:.4f} "
              f"({len(losses)} steps)")
        if device.type == "cuda":
            print(f"  peak allocated  {peak_alloc:.2f} GB")
            print(f"  peak reserved   {peak_res:.2f} GB  "
                  f"({100 * peak_res / total_mem:.0f}% of {total_mem:.1f} GB)")
            print(f"  verdict         {fitted_verdict(peak_res, total_mem)}")
        print(f"  projected epoch (480 wells)  ~{epoch_min:.0f} min")
        print(f"{'='*70}")

        if device.type == "cuda" and peak_res > 0.92 * total_mem:
            print(f"  -> smaller: --batch 1, --dim {max(64, dim // 2)}, --layers "
                  f"{max(1, layers - 1)}")
    elif status == "slow":
        print(f"\n  no timed step completed within {args.time_budget:.0f} s -- the "
              f"shape does not fit this machine at a usable speed.")

    return 3 if status == "slow" else 0


if __name__ == "__main__":
    raise SystemExit(main())
