"""Train on ONE (plate, cell line) file and diagnose whether it learns.

Everything is measured against fixed held-out wells from the same file, so the
loss curve is comparable across steps (the trainer's per-step loss is on a fresh
random well and mostly measures which drug got drawn).

Phases
  A  split the file's drug wells into train / eval; build the eval batches once
  B  loss baselines: predict 0, and predict the control cells' mean profile
  C  train; log eval loss, per-module gradient norms, peak VRAM
  D  depth probe: a fresh N-layer model, one backward, per-layer gradient norms
  E  drug ablation: same control + coords, different drug -> does the output move
  F  write a markdown report

    python models/testing/single_file_train.py --file plate6_CVCL_1125.h5ad --steps 100
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from models.model import PerturbationModel, perturbation_loss
from models.molformer_atoms import AtomEmbeddingStore
from models.tahoe_data import DEFAULT_ROOT, BackedFile, build_example

GB = 1e9


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def make_model(cfg, n_genes, chem_dim, device, layers=None):
    return PerturbationModel(
        dim=cfg.dim,
        conc_dim=cfg.conc_dim,
        num_heads=cfg.heads,
        dropout=cfg.dropout,
        num_layers=cfg.layers if layers is None else layers,
        num_of_genes=n_genes,
        chem_dim=chem_dim,
        cell_coordi_dim=2,
    ).to(device)


def forward_batch(model, batch, cfg):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        predicted = model(
            batch["control"],
            batch["chem"],
            batch["coords"],
            batch["mask"],
            query_chunk=cfg.query_chunk,
            readout_chunk=cfg.readout_chunk,
        )
    return predicted.float()


def evaluate(model, batches, cfg):
    model.eval()
    losses = []
    with torch.no_grad():
        for batch in batches:
            predicted = forward_batch(model, batch, cfg)
            losses.append(float(perturbation_loss(predicted, batch["target"], 0.5)))
    model.train()
    return float(np.mean(losses))


def group_grad_norms(model) -> dict[str, float]:
    """L2 norm of the gradient per module."""
    groups: dict[str, float] = {}
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        if name.startswith("layers."):
            key = f"layer{name.split('.')[1]}"
        elif name.startswith(("conc_emb", "conc_ffn")):
            key = "conc"
        elif name.startswith("coordi_layer"):
            key = "coordi_layer"
        elif name.startswith("gene_embedding"):
            key = "gene_embedding"
        elif name.startswith("chem_embedding"):
            key = "chem_embedding"
        elif name.startswith("merger"):
            key = "merger"
        else:
            key = name.split(".")[0]
        groups[key] = groups.get(key, 0.0) + float(parameter.grad.detach().pow(2).sum())
    return {k: math.sqrt(v) for k, v in groups.items()}


def build_eval_batches(backed, store, wells, cfg, device, n_control, n_target, seed):
    rng = np.random.default_rng(seed)
    batches = []
    for i, well in enumerate(wells):
        example = build_example(
            backed, backed.control_wells[0], well, n_control, n_target, rng
        )
        chem, mask = store.collate(
            [example.drug], [example.concentration], device, mode=cfg.chem_tokens
        )
        batches.append(
            {
                "control": example.control.unsqueeze(0).to(device),
                "chem": chem,
                "mask": mask,
                "coords": example.coords.unsqueeze(0).to(device),
                "target": example.target.unsqueeze(0).to(device),
                "drug": example.drug,
            }
        )
    return batches


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", default="plate6_CVCL_1125.h5ad")
    parser.add_argument("--root", default=DEFAULT_ROOT)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--eval-every", type=int, default=20)
    parser.add_argument("--eval-wells", type=int, default=8)
    parser.add_argument("--n-control", type=int, default=256)
    parser.add_argument("--n-target", type=int, default=128)
    parser.add_argument("--min-cells", type=int, default=128)
    parser.add_argument("--dim", type=int, default=512)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--conc-dim", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0,
                        help="0 disables clipping")
    parser.add_argument("--chem-tokens", choices=["all", "atoms"], default="all")
    parser.add_argument("--query-chunk", type=int, default=4096)
    parser.add_argument("--readout-chunk", type=int, default=16)
    parser.add_argument("--depth-probe-layers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="models/testing/single_file_report.md")
    cfg = parser.parse_args()
    cfg.query_chunk = cfg.query_chunk or None
    cfg.readout_chunk = cfg.readout_chunk or None

    torch.manual_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    path = os.path.join(cfg.root, cfg.file)
    backed = BackedFile(path, cfg.min_cells)
    store = AtomEmbeddingStore()
    n_genes = backed.n_vars
    wells = backed.drug_wells(cfg.min_cells)

    rng = np.random.default_rng(cfg.seed)
    order = rng.permutation(len(wells))
    eval_wells = [wells[i] for i in order[: cfg.eval_wells]]
    train_wells = [wells[i] for i in order[cfg.eval_wells :]]

    print(f"file        : {cfg.file}  ({backed.n_obs} cells, {n_genes} genes)")
    print(f"control     : {len(backed.control_wells)} DMSO wells")
    print(f"drug wells  : {len(wells)}  ->  train {len(train_wells)}, "
          f"held out {len(eval_wells)}")
    print(f"dose        : {sorted({w.concentration for w in wells})}")
    print(f"device      : {torch.cuda.get_device_name(0)}")

    eval_batches = build_eval_batches(
        backed, store, eval_wells, cfg, device, cfg.n_control, cfg.n_target,
        seed=cfg.seed + 1,
    )

    # ---------------- B. baselines ------------------------------------- #
    model = make_model(cfg, n_genes, store.dim + 1, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=0.01)

    zero_loss, mean_loss = [], []
    for batch in eval_batches:
        zeros = torch.zeros_like(batch["target"])
        zero_loss.append(float(perturbation_loss(zeros, batch["target"], 0.5)))
        control_mean = batch["control"][0].mean(dim=1)  # (G,) mean over control cells
        flat = control_mean.view(1, 1, -1).expand(1, cfg.n_target, -1)
        mean_loss.append(float(perturbation_loss(flat, batch["target"], 0.5)))
    zero_loss = float(np.mean(zero_loss))
    mean_loss = float(np.mean(mean_loss))

    start_loss = evaluate(model, eval_batches, cfg)
    print(f"\nbaselines on the held-out wells:")
    print(f"  predict 0                : {zero_loss:.4f}")
    print(f"  predict control mean     : {mean_loss:.4f}")
    print(f"  untrained model          : {start_loss:.4f}")

    # ---------------- C. train ----------------------------------------- #
    model.train()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    history = []
    running = []
    steps_per_sec = []
    peak_alloc = peak_res = 0.0
    grad_snapshot = None
    input_grad_snapshot = None
    step = 0
    t_start = time.time()

    print(f"\n{'step':>5} {'train':>9} {'eval':>9} {'|g|':>10} "
          f"{'vram':>8} {'s/step':>7}")
    while step < cfg.steps:
        well = train_wells[step % len(train_wells)]
        control_well = backed.control_wells[step % len(backed.control_wells)]
        example = build_example(
            backed, control_well, well, cfg.n_control, cfg.n_target, rng
        )
        chem, mask = store.collate(
            [example.drug], [example.concentration], device, mode=cfg.chem_tokens
        )
        batch = {
            "control": example.control.unsqueeze(0).to(device).requires_grad_(True),
            "chem": chem,
            "mask": mask,
            "coords": example.coords.unsqueeze(0).to(device),
            "target": example.target.unsqueeze(0).to(device),
        }

        torch.cuda.synchronize() if device.type == "cuda" else None
        t0 = time.time()
        optimizer.zero_grad(set_to_none=True)
        predicted = forward_batch(model, batch, cfg)
        loss = perturbation_loss(predicted, batch["target"], 0.5)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), cfg.grad_clip if cfg.grad_clip > 0 else float("inf")
        )
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.time() - t0

        step += 1
        running.append(float(loss.detach()))
        steps_per_sec.append(elapsed)
        grad_snapshot = group_grad_norms(model)
        grad_snapshot["_total"] = float(grad_norm)

        # per-gene input gradient on the last step (silent-gene bottleneck)
        if batch["control"].grad is not None:
            per_gene = batch["control"].grad[0].norm(dim=1)
            input_grad_snapshot = {
                "total": float(per_gene.norm()),
                "max": float(per_gene.max()),
                "median": float(per_gene.median()),
                "zero_fraction": float((per_gene == 0).float().mean()),
                "max_over_median": float(per_gene.max() / per_gene.median().clamp_min(1e-30)),
            }

        if device.type == "cuda":
            peak_alloc = max(peak_alloc, torch.cuda.max_memory_allocated() / GB)
            peak_res = max(peak_res, torch.cuda.max_memory_reserved() / GB)

        if step % cfg.eval_every == 0 or step == cfg.steps:
            eval_loss = evaluate(model, eval_batches, cfg)
            train_recent = float(np.mean(running[-cfg.eval_every :]))
            history.append(
                {
                    "step": step,
                    "train": train_recent,
                    "eval": eval_loss,
                    "grad": grad_snapshot,
                    "vram": peak_alloc,
                    "seconds": float(np.mean(steps_per_sec[-cfg.eval_every :])),
                }
            )
            print(f"{step:>5} {train_recent:>9.4f} {eval_loss:>9.4f} "
                  f"{grad_snapshot['_total']:>10.3e} {peak_alloc:>7.2f}G "
                  f"{history[-1]['seconds']:>7.2f}")

    wall = time.time() - t_start
    final_loss = history[-1]["eval"]

    # ---------------- D. depth probe ----------------------------------- #
    depth = None
    try:
        probe = make_model(cfg, n_genes, store.dim + 1, device,
                           layers=cfg.depth_probe_layers)
        batch = eval_batches[0]
        predicted = forward_batch(probe, batch, cfg)
        perturbation_loss(predicted, batch["target"], 0.5).backward()
        norms = group_grad_norms(probe)
        depth = {k: v for k, v in norms.items() if k.startswith("layer")}
        probe.zero_grad(set_to_none=True)
        del probe, predicted
        torch.cuda.empty_cache()
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        depth = {"oom": True}
        print(f"\ndepth probe with {cfg.depth_probe_layers} layers: CUDA OOM (skipped)")

    # ---------------- E. drug ablation --------------------------------- #
    model.eval()
    ablation = []
    with torch.no_grad():
        for batch in eval_batches[:4]:
            true_pred = forward_batch(model, batch, cfg)
            swapped = dict(batch)
            other = store.collate(
                [eval_batches[0]["drug"] if batch["drug"] != eval_batches[0]["drug"]
                 else eval_batches[1]["drug"]],
                [5.0], device, mode=cfg.chem_tokens,
            )
            swapped["chem"], swapped["mask"] = other
            swapped_pred = forward_batch(model, swapped, cfg)
            relative = float(
                (swapped_pred - true_pred).norm() / true_pred.norm().clamp_min(1e-12)
            )
            ablation.append(relative)
            masked = dict(batch)
            masked["mask"] = torch.zeros_like(batch["mask"])
            no_drug = forward_batch(model, masked, cfg)
            ablation.append(
                float((no_drug - true_pred).norm() / true_pred.norm().clamp_min(1e-12))
            )
    model.train()

    # ---------------- F. report ---------------------------------------- #
    first, last = history[0], history[-1]
    learned = mean_loss - final_loss

    lines = []
    add = lines.append
    add(f"# Single-file training report — `{cfg.file}`\n")
    add(f"- cells {backed.n_obs}, genes {n_genes}, drug wells {len(wells)} "
        f"(train {len(train_wells)} / held out {len(eval_wells)})")
    add(f"- dose(s) {sorted({w.concentration for w in wells})}, "
        f"chem tokens `{cfg.chem_tokens}`, dropout {cfg.dropout}")
    add(f"- dim {cfg.dim}, heads {cfg.heads}, layers {cfg.layers}, "
        f"conc_dim {cfg.conc_dim}, lr {cfg.lr}")
    add(f"- {cfg.steps} steps in {wall/60:.1f} min, "
        f"{np.mean(steps_per_sec):.2f} s/step, peak VRAM {peak_alloc:.2f} GB "
        f"(reserved {peak_res:.2f} GB of 12.88)\n")

    add("## Is it learning?\n")
    add("| reference | loss |")
    add("|---|---|")
    add(f"| predict 0 | {zero_loss:.4f} |")
    add(f"| predict control mean | {mean_loss:.4f} |")
    add(f"| untrained model | {start_loss:.4f} |")
    add(f"| after {first['step']} steps | {first['eval']:.4f} |")
    add(f"| after {last['step']} steps | **{final_loss:.4f}** |")
    add("")
    add(f"Eval loss dropped {first['eval'] - final_loss:.4f} over the run "
        f"({100 * (first['eval'] - final_loss) / first['eval']:.1f}%), and is "
        f"{learned:+.4f} relative to the control-mean baseline.")
    if final_loss >= mean_loss:
        add("")
        add("**It never beat 'predict the control mean'.** Everything it learned is "
            "the cell line's baseline expression; nothing drug-specific is being "
            "captured yet.")
    else:
        add("")
        add("**It beat the control-mean baseline**, so some drug-dependent signal "
            "is being captured.")

    add("\n## Eval curve\n")
    add("| step | train (last window) | eval | total |g| | s/step |")
    add("|---|---|---|---|---|")
    for h in history:
        add(f"| {h['step']} | {h['train']:.4f} | {h['eval']:.4f} | "
            f"{h['grad']['_total']:.3e} | {h['seconds']:.2f} |")

    add("\n## Gradient flow (final step, before clipping)\n")
    add("| module | ‖grad‖ | share of total |")
    add("|---|---|---|")
    total_sq = sum(v ** 2 for k, v in grad_snapshot.items() if k != "_total")
    for key in sorted((k for k in grad_snapshot if k != "_total"),
                      key=lambda k: -grad_snapshot[k]):
        value = grad_snapshot[key]
        add(f"| {key} | {value:.3e} | {100 * value ** 2 / max(total_sq, 1e-30):.1f}% |")

    if isinstance(depth, dict) and "oom" not in depth:
        add(f"\n### Vanishing check — {cfg.depth_probe_layers} stacked layers, "
            f"same batch\n")
        add("| layer | ‖grad‖ | vs layer0 |")
        add("|---|---|---|")
        base = depth.get("layer0", 0.0) or 1e-30
        for key in sorted(depth, key=lambda k: int(k.replace("layer", ""))):
            add(f"| {key} | {depth[key]:.3e} | {depth[key] / base:.3f}x |")
        values = [depth[k] for k in sorted(depth, key=lambda k: int(k.replace('layer','')))]
        add("")
        add(f"Layer ratio (last/first): **{values[-1] / base:.3f}x** — "
            + ("no vanishing." if values[-1] / base > 0.5 else
               "gradients do *not* decay with depth."))

    if input_grad_snapshot:
        add("\n## Input gradient (the silent-gene bottleneck)\n")
        add(f"- ‖dL/dcontrol‖ = {input_grad_snapshot['total']:.3e}")
        add(f"- per-gene: max {input_grad_snapshot['max']:.3e}, "
            f"median {input_grad_snapshot['median']:.3e} "
            f"(**max/median {input_grad_snapshot['max_over_median']:.2e}**)")
        add(f"- genes with exactly zero gradient: "
            f"{100 * input_grad_snapshot['zero_fraction']:.1f}%")

    add("\n## Is the chemical path alive? (drug ablation)\n")
    add(f"- swap the drug, keep control + coords: output moves "
        f"{100 * float(np.mean(ablation[0::2])):.1f}% (relative L2)")
    add(f"- mask the chemical entirely: output moves "
        f"{100 * float(np.mean(ablation[1::2])):.1f}%")
    if float(np.mean(ablation[0::2])) < 0.01:
        add("\n**The drug barely affects the prediction** — the chemical branch is "
            "being ignored.")

    Path(cfg.out).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.out).write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"\n=== summary ===")
    print(f"  baselines : zero {zero_loss:.3f} | control-mean {mean_loss:.3f} | "
          f"untrained {start_loss:.3f}")
    print(f"  after {cfg.steps} steps : {final_loss:.3f}  "
          f"(vs control-mean {mean_loss - final_loss:+.3f})")
    print(f"  peak VRAM : {peak_alloc:.2f} GB allocated / {peak_res:.2f} GB reserved")
    print(f"  mean s/step {np.mean(steps_per_sec):.2f}  wall {wall/60:.1f} min")
    print(f"  drug ablation: swap {100*float(np.mean(ablation[0::2])):.1f}%  "
          f"mask {100*float(np.mean(ablation[1::2])):.1f}%")
    print(f"  report -> {cfg.out}")
    backed.close()


if __name__ == "__main__":
    main()
