"""fp16 vs bf16 head-to-head on the real model and real batches.

Runs identical weights and identical batches through three precision modes and
reports, per step: wall time, peak GPU memory, loss, gradient norm, the largest
single parameter gradient, how many parameter gradients are non-finite, the
GradScaler scale, and -- the thing that actually matters -- whether the optimizer
step was applied or skipped because of an overflow.

    python models/testing/amp_benchmark.py
    python models/testing/amp_benchmark.py --steps 8 --mode fp16
"""

from __future__ import annotations

import argparse
import contextlib
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from models.model import PerturbationModel, perturbation_loss
from models.molformer_atoms import AtomEmbeddingStore
from models.tahoe_data import BackedFile, build_example, build_file_index

MODES = {
    "fp32": None,
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
}


def build_batches(n: int, seed: int, n_control: int, n_target: int, min_cells: int):
    """``n`` real (control, chem, coords, target) batches from one plate file."""
    index = build_file_index()
    store = AtomEmbeddingStore()
    backed = BackedFile(index[0].path, min_cells)

    rng = np.random.default_rng(seed)
    wells = backed.drug_wells(min_cells)[:n]
    out = []
    for i, well in enumerate(wells):
        example = build_example(
            backed,
            backed.control_wells[i % len(backed.control_wells)],
            well,
            n_control,
            n_target,
            rng,
        )
        chem, mask = store.collate([example.drug], [example.concentration])
        out.append(
            {
                "control": example.control.unsqueeze(0).cuda(),
                "chem": chem.cuda(),
                "mask": mask.cuda(),
                "coords": example.coords.unsqueeze(0).cuda(),
                "target": example.target.unsqueeze(0).cuda(),
                "drug": example.drug,
            }
        )
    backed.close()
    return out, store


def run(mode: str, batches, n_genes: int, chem_dim: int, args, query_chunk, readout_chunk):
    dtype = MODES[mode]
    torch.manual_seed(0)
    model = PerturbationModel(
        dim=args.dim,
        conc_dim=32,
        num_heads=args.heads,
        dropout=0.1,
        num_layers=1,
        num_of_genes=n_genes,
        chem_dim=chem_dim,
        cell_coordi_dim=2,
    ).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

    scaler = torch.amp.GradScaler("cuda") if mode == "fp16" else None
    ctx = (
        contextlib.nullcontext()
        if dtype is None
        else torch.autocast(device_type="cuda", dtype=dtype)
    )

    # a stable reference parameter to detect whether the step was applied
    reference = model.coordi_layer[0].weight.detach().clone()

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    rows = []

    for step in range(args.steps):
        batch = batches[step % len(batches)]
        optimizer.zero_grad(set_to_none=True)

        torch.cuda.synchronize()
        t0 = time.time()
        with ctx:
            predicted = model(
                batch["control"],
                batch["chem"],
                batch["coords"],
                batch["mask"],
                query_chunk=query_chunk,
                readout_chunk=readout_chunk,
            )
        loss = perturbation_loss(predicted.float(), batch["target"], 0.5)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
        else:
            loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        max_grad, nonfinite = 0.0, 0
        for p in model.parameters():
            if p.grad is None:
                continue
            if not torch.isfinite(p.grad).all():
                nonfinite += 1
                continue
            max_grad = max(max_grad, p.grad.abs().max().item())

        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        torch.cuda.synchronize()
        elapsed = time.time() - t0

        applied = not torch.equal(reference, model.coordi_layer[0].weight.detach())

        rows.append(
            {
                "step": step,
                "seconds": elapsed,
                "loss": float(loss.detach()),
                "grad_norm": float(grad_norm),
                "max_grad": max_grad,
                "nonfinite": nonfinite,
                "scale": scaler.get_scale() if scaler is not None else None,
                "applied": applied,
            }
        )

    peak = torch.cuda.max_memory_allocated() / 1e9
    return rows, peak


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--mode", default="all", choices=["all", *MODES])
    parser.add_argument("--dim", type=int, default=512)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--tf32", action="store_true",
                        help="allow TF32 matmuls (what train.py does); only affects fp32")
    parser.add_argument("--n-control", type=int, default=256)
    parser.add_argument("--n-target", type=int, default=128)
    parser.add_argument("--min-cells", type=int, default=128)
    args = parser.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = args.tf32
    torch.backends.cudnn.allow_tf32 = True
    print(f"tf32 matmul={args.tf32} (only affects the fp32 mode)")

    batches, store = build_batches(
        args.steps, 0, args.n_control, args.n_target, args.min_cells
    )
    n_genes = batches[0]["control"].shape[1]
    chem_dim = batches[0]["chem"].shape[-1]
    print(f"#{args.steps} real batches, genes={n_genes}, chem_dim={chem_dim}, "
          f"drug0={batches[0]['drug']}\n")

    modes = list(MODES) if args.mode == "all" else [args.mode]
    summary = {}

    for mode in modes:
        # chunking is required for fp32/bf16 part-1 at this G; keep it for all modes
        rows, peak = run(mode, batches, n_genes, chem_dim, args, 4096, 16)
        summary[mode] = (rows, peak)
        print(f"=== {mode} ===")
        print(f"{'step':>4} {'loss':>10} {'s/step':>7} {'|g|':>10} {'max|g_p|':>10} "
              f"{'nonfinite':>10} {'scale':>8} {'applied':>8}")
        for r in rows:
            scale = "-" if r["scale"] is None else f"{r['scale']:.0f}"
            print(
                f"{r['step']:>4} {r['loss']:>10.4f} {r['seconds']:>7.2f} "
                f"{r['grad_norm']:>10.3e} {r['max_grad']:>10.3e} {r['nonfinite']:>10d} "
                f"{scale:>8} {str(r['applied']):>8}"
            )
        applied = sum(r["applied"] for r in rows)
        print(f"peak mem {peak:.2f} GB | mean {np.mean([r['seconds'] for r in rows]):.2f} s/step "
              f"| steps applied {applied}/{len(rows)}\n")


if __name__ == "__main__":
    main()
