"""Construct either architecture and dispatch batches without chemical work for controls."""
from __future__ import annotations

import torch

from models.control_model import ControlPerturbationModel
from models.model import PerturbationModel, perturbation_loss


def create_model(cfg, n_genes, chem_dim=None):
    kwargs = dict(dim=cfg.dim, conc_dim=cfg.conc_dim, num_heads=cfg.heads,
                  num_layers=cfg.layers, dropout=cfg.dropout, num_of_genes=n_genes,
                  cell_coordi_dim=2, normalization=cfg.normalization == "layernorm",
                  gene_statistics=cfg.gene_statistics, gene_residual=cfg.gene_residual)
    if cfg.model_type == "control":
        return ControlPerturbationModel(**kwargs)
    if chem_dim is None:
        raise ValueError("Chemical feature width must be inferred from the token cache")
    return PerturbationModel(chem_dim=chem_dim, attention_normalization=cfg.attention_normalization,
                             chem_ffn_residual=cfg.chem_ffn_residual, ffn_multiplier=cfg.ffn_multiplier,
                             joint_normalization=cfg.joint_normalization, **kwargs)


def load_chemistry(cfg):
    if cfg.model_type == "control":
        return None
    from models.molformer_atoms import AtomEmbeddingStore
    return AtomEmbeddingStore(cfg.atom_cache)


def predict(model, example, cfg, device, store=None):
    control = example.control.unsqueeze(0).to(device)
    coords = example.coords.unsqueeze(0).to(device)
    chunks = dict(query_chunk=cfg.query_chunk or None, readout_chunk=cfg.readout_chunk or None)
    if cfg.model_type == "control":
        return model(control, coords, **chunks)
    chem, mask = store.collate([example.drug], [example.concentration], device, mode=cfg.chem_tokens)
    return model(control, chem, coords, mask, **chunks)


def loss_for(model, example, cfg, device, store=None):
    enabled = cfg.amp == "bf16"
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=enabled):
        predicted = predict(model, example, cfg, device, store)
    target = example.target.unsqueeze(0).to(device)
    return perturbation_loss(predicted.float(), target, cfg.scale_exponent)


@torch.no_grad()
def mean_baseline_losses(example, cfg):
    """Comparable per-well constant predictions, using pre-L2 log expression.

    t_mean_loss uses the sampled target cells' per-gene mean (an oracle
    constant-prediction reference). i_mean_loss uses the sampled input DMSO
    cells' per-gene mean. Neither baseline participates in optimization.
    """
    target = example.target.detach().float().unsqueeze(0)
    with torch.autocast(device_type=target.device.type, enabled=False):
        target_mean = target.mean(dim=1, keepdim=True).expand_as(target)
        input_mean = example.control.detach().float().mean(dim=-1)
        input_mean = input_mean.to(target.device)[None, None, :].expand_as(target)
        t_mean_loss = perturbation_loss(target_mean, target, cfg.scale_exponent)
        i_mean_loss = perturbation_loss(input_mean, target, cfg.scale_exponent)
    return float(t_mean_loss), float(i_mean_loss)
