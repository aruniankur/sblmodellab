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
                             joint_normalization=cfg.joint_normalization,
                             chem_skip_gate=cfg.chem_skip_gate, chem_skip_init=cfg.chem_skip_init,
                             checkpoint_layers=cfg.checkpoint_layers, **kwargs)


def load_chemistry(cfg):
    if cfg.model_type == "control":
        return None
    from models.molformer_atoms import AtomEmbeddingStore
    return AtomEmbeddingStore(cfg.atom_cache)


def predict_batch(model, examples, cfg, device, store=None):
    """Each batch item is an independent treatment well with its own controls."""
    if not examples:
        raise ValueError("A forward batch needs at least one example")
    control = torch.stack([example.control for example in examples]).to(device)
    coords = torch.stack([example.coords for example in examples]).to(device)
    chunks = dict(query_chunk=cfg.query_chunk or None, readout_chunk=cfg.readout_chunk or None)
    if cfg.model_type == "control":
        return model(control, coords, **chunks)
    chem, mask = store.collate([example.drug for example in examples],
                               [example.concentration for example in examples], device,
                               mode=cfg.chem_tokens)
    return model(control, chem, coords, mask, **chunks)


def predict(model, example, cfg, device, store=None):
    return predict_batch(model, [example], cfg, device, store)


def losses_for_batch(model, examples, cfg, device, store=None):
    """One comparable loss per well; callers choose accumulation weighting."""
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=cfg.amp == "bf16"):
        predicted = predict_batch(model, examples, cfg, device, store)
    target = torch.stack([example.target for example in examples]).to(device)
    return (predicted.float() - target.float()).square().sum(-1).mean(-1) / target.shape[-1]**cfg.scale_exponent


def loss_for(model, example, cfg, device, store=None):
    return losses_for_batch(model, [example], cfg, device, store).mean()


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
