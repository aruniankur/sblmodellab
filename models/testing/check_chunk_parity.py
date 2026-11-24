"""Parity check: the memory-chunking knobs in ``model.py`` must not change the model.

Compares the default code path (``query_chunk=None, readout_chunk=None``) against
the chunked path on identical inputs and identical weights, for the forward pass
and for gradients w.r.t. every parameter and the control tensor.

    python models/testing/check_chunk_parity.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from models.model import PerturbationModel, perturbation_loss


def run(query_chunk, readout_chunk):
    torch.manual_seed(0)
    batch, num_genes, n_control, n_target = 1, 512, 64, 32
    num_atoms, chem_dim = 8, 17

    model = PerturbationModel(
        dim=64,
        conc_dim=16,
        num_heads=4,
        dropout=0.0,
        num_layers=2,
        num_of_genes=num_genes,
        chem_dim=chem_dim,
        cell_coordi_dim=2,
    ).eval()

    generator = torch.Generator().manual_seed(0)
    control = torch.rand(batch, num_genes, n_control, generator=generator, requires_grad=True)
    chem = torch.randn(batch, num_atoms, chem_dim, generator=generator)
    coords = torch.randn(batch, n_target, 2, generator=generator)
    target = torch.rand(batch, n_target, num_genes, generator=generator)
    mask = torch.ones(batch, 1, 1, num_atoms, dtype=torch.bool)
    mask[0, 0, 0, -2:] = False  # exercise masking of padded atoms

    predicted = model(
        control,
        chem,
        coords,
        mask,
        query_chunk=query_chunk,
        readout_chunk=readout_chunk,
    )
    loss = perturbation_loss(predicted, target, 0.5)
    loss.backward()

    grads = {name: p.grad.detach().clone() for name, p in model.named_parameters()}
    return predicted.detach(), grads, control.grad.detach().clone(), float(loss)


def main() -> int:
    ref_out, ref_grads, ref_ctrl, ref_loss = run(None, None)
    chk_out, chk_grads, chk_ctrl, chk_loss = run(128, 8)

    out_diff = (ref_out - chk_out).abs().max().item()
    ctrl_diff = (ref_ctrl - chk_ctrl).abs().max().item()
    grad_diff = max((ref_grads[k] - chk_grads[k]).abs().max().item() for k in ref_grads)
    loss_diff = abs(ref_loss - chk_loss)

    print(f"forward max |diff|      : {out_diff:.3e}")
    print(f"dL/dcontrol max |diff|  : {ctrl_diff:.3e}")
    print(f"param grads max |diff|  : {grad_diff:.3e}")
    print(f"loss |diff|             : {loss_diff:.3e}")

    tolerance = 1e-5
    ok = max(out_diff, ctrl_diff, grad_diff, loss_diff) < tolerance
    print("PASS" if ok else f"FAIL (tolerance {tolerance:g})")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
