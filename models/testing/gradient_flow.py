"""Gradient-flow diagnostics for ``models/model.py``.

The model is NOT modified: everything here only observes it through autograd
hooks, retained grads and ``torch.autograd``. ``F.scaled_dot_product_attention``
is never patched -- attention statistics are rebuilt from the model's own q/k
tensors (captured by forward hooks) and validated against the model's real SDPA
output.

Phases
------
1. ``structure``  one deterministic forward/backward in eval mode with hooks on
                  every block, to see where backward signal is lost.
2. ``attention``  exact (chunked) softmax statistics for all three SDPA calls.
3. ``train``      Adam then SGD steps on dummy data, tracking grad norms, update
                  ratios, cross-batch gradient noise and loss.
4. ``depth``      sweep ``num_layers`` to see whether the stack attenuates the
                  gradient that reaches the input.

Examples
--------
    python models/testing/gradient_flow.py --genes 20000 --cells 32
    python models/testing/gradient_flow.py --genes 20000 --cells 64 --steps 20
    python models/testing/gradient_flow.py --data both --genes 20000
    python models/testing/gradient_flow.py --genes 2000 --quick

Outputs (written next to this file):
    gradient_flow_report.md, gradient_flow_report.json, *.png
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from models.model import PerturbationModel, perturbation_loss  # noqa: E402


# --------------------------------------------------------------------------- #
# configuration
# --------------------------------------------------------------------------- #
@dataclass
class Config:
    genes: int = 20000
    cells: int = 32
    batch: int = 1
    chems: int = 10
    chem_dim: int = 128           # drug fingerprint width (model consumes chem_dim + 1)
    dim: int = 256
    conc_dim: int = 32
    heads: int = 4
    layers: int = 2
    dropout: float = 0.1
    sparsity: float = 0.70        # zero-inflation of the synthetic expression matrix
    steps: int = 20
    lr: float = 1e-3
    sgd_lr: float = 1e-3
    seed: int = 0
    device: str = "auto"
    data: str = "structured"      # structured | noise | both
    depth_sweep: str = "1,2,4,8"
    out_dir: str = HERE
    tag: str = ""

    @property
    def model_chem_dim(self) -> int:
        return self.chem_dim + 1

    @property
    def silent_fraction(self) -> float:
        return 0.02


# --------------------------------------------------------------------------- #
# small utils
# --------------------------------------------------------------------------- #
def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def t_norm(x) -> float:
    if isinstance(x, torch.Tensor):
        return float(x.detach().float().norm())
    return 0.0


def act_norm(grads) -> float:
    """Norm of the *activation* gradient only.

    ``Linear.grad_input`` is ``(grad_input, grad_weight, grad_bias)``; only the
    first entry belongs to the input activation, so parameter grads must not be
    folded into the transmission ratio.
    """
    if grads is None:
        return 0.0
    if isinstance(grads, torch.Tensor):
        return float(grads.detach().float().norm())
    for g in grads:
        if isinstance(g, torch.Tensor):
            return float(g.detach().float().norm())
    return 0.0


def rms(x: torch.Tensor) -> float:
    return float(x.detach().float().pow(2).mean().sqrt())


def free(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def pctl(x: torch.Tensor, qs=(0.5, 0.9, 0.99)) -> list[float]:
    x = x.detach().float().flatten()
    q = torch.tensor(qs, device=x.device, dtype=x.dtype)
    return [float(v) for v in torch.quantile(x, q)]


# --------------------------------------------------------------------------- #
# dummy data
# --------------------------------------------------------------------------- #
def _make_expression(cfg: Config, gen: torch.Generator) -> torch.Tensor:
    """Synthetic (G, C) expression matrix.

    ``structured`` = sparse, Poisson-sampled, co-expression modules -- chosen so
    the gene x gene cosine distribution looks like sparse scRNA-seq instead of
    the degenerate "every gene correlates with every gene" you get from dense
    non-negative noise. ``noise`` = i.i.d. uniform.

    ~2% of genes are silenced (all-zero) in both modes so the model's
    ``clamp_min(1e-12)`` branch is actually exercised.
    """
    G, C = cfg.genes, cfg.cells

    if cfg.data == "noise":
        expr = torch.rand(G, C, generator=gen)
    elif cfg.data == "structured":
        n_programs = 48
        membership = (torch.rand(G, n_programs, generator=gen) < 0.03).float()
        programs = torch.randn(n_programs, C, generator=gen)
        gene_base = torch.randn(G, 1, generator=gen) * 0.8
        cell_lib = torch.randn(C, generator=gen) * 0.5
        log_rate = gene_base + 0.8 * (membership @ programs) + cell_lib
        counts = torch.poisson(torch.exp(log_rate) * 2.0, generator=gen).float()
        if cfg.sparsity > 0:
            keep = (torch.rand(G, C, generator=gen) > cfg.sparsity).float()
            counts = counts * keep
        expr = torch.log1p(counts)
    else:
        raise ValueError(f"unknown data mode {cfg.data!r}")

    silent = (expr.sum(dim=1) == 0) | (torch.rand(G, generator=gen) < cfg.silent_fraction)
    expr[silent] = 0.0
    return expr


def make_batch(cfg: Config, gen: torch.Generator, device: torch.device) -> dict:
    """One synthetic well.

    control (B,G,C) · chem (B,Nc,F+1) · mask (B,1,1,Nc) · coords (B,C,2) · target (B,C,G)
    """
    expr = _make_expression(cfg, gen)                                   # (G, C)
    control = expr.unsqueeze(0).expand(cfg.batch, -1, -1).contiguous()
    target = expr.t().unsqueeze(0).expand(cfg.batch, -1, -1).contiguous()
    target = target * (1.0 + 0.05 * torch.randn(target.shape, generator=gen))

    chem = torch.randn(cfg.batch, cfg.chems, cfg.chem_dim, generator=gen)
    conc = torch.rand(cfg.batch, 1, 1, generator=gen)
    chem = torch.cat([chem, conc.expand(-1, cfg.chems, -1)], dim=-1)

    mask = torch.rand(cfg.batch, cfg.chems, generator=gen) > 0.3
    mask[:, 0] = True                                                   # never fully mask
    mask = mask[:, None, None, :]

    coords = torch.randn(cfg.batch, cfg.cells, 2, generator=gen)

    return {
        "control": control.to(device),
        "chem": chem.to(device),
        "mask": mask.to(device),
        "coords": coords.to(device),
        "target": target.to(device),
    }


def build_model(cfg: Config, device: torch.device, layers: int | None = None) -> PerturbationModel:
    torch.manual_seed(cfg.seed)
    return PerturbationModel(
        dim=cfg.dim,
        conc_dim=cfg.conc_dim,
        num_heads=cfg.heads,
        dropout=cfg.dropout,
        num_layers=cfg.layers if layers is None else layers,
        num_of_genes=cfg.genes,
        chem_dim=cfg.model_chem_dim,
        cell_coordi_dim=2,
    ).to(device)


def build_pool(cfg: Config, device: torch.device, gen: torch.Generator, size: int) -> list[dict]:
    """A small fixed dataset that the training loop cycles through."""
    return [make_batch(cfg, gen, device) for _ in range(size)]


def input_grad_stats(control: torch.Tensor, grad: torch.Tensor) -> dict:
    """Split ‖∂L/∂control‖ into silenced-gene vs expressed-gene contributions.

    This matters because ``x / clamp_min(‖x‖, 1e-12)`` gives all-zero genes a
    1e12 backward gain, so the raw norm is dominated by genes that carry no signal.
    """
    row_norm = control.detach().norm(dim=-1)                     # (B, G)
    silent = row_norm <= 1e-12                                   # (B, G)
    sq = grad.detach().pow(2).sum(-1)                            # (B, G)
    total = float(sq.sum().sqrt())
    silent_norm = float(sq[silent].sum().sqrt()) if silent.any() else 0.0
    clean_norm = float(sq[~silent].sum().sqrt()) if (~silent).any() else 0.0
    return {
        "total": total,
        "silent": silent_norm,
        "nonsilent": clean_norm,
        "silent_share": silent_norm / max(total, 1e-30),
        "n_silent": int(silent.sum()),
    }


# --------------------------------------------------------------------------- #
# exact attention statistics (chunked; a full GxG matrix is never materialised)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def attention_stats(
    q: torch.Tensor,
    k: torch.Tensor,
    scale: float,
    mask: torch.Tensor | None = None,
    chunk: int = 1024,
    want_reconstruction: torch.Tensor | None = None,
    self_aligned: bool = False,
) -> dict:
    """q,k: (H, Lq, d) / (H, Lk, d); mask: (H, 1, Lk) bool (True = keep)."""
    H, Lq, _ = q.shape
    Lk = k.shape[1]
    entropy, max_prob, effective, self_prob = [], [], [], []
    out_chunks = [] if want_reconstruction is not None else None
    logit_sum = logit_sq = 0.0
    logit_n = 0.0

    for h in range(H):
        qh, kh = q[h], k[h]
        for s in range(0, Lq, chunk):
            e = min(s + chunk, Lq)
            logits = (qh[s:e] @ kh.t()) * scale                     # (c, Lk)
            if mask is not None:
                keep = mask[h, :, :].expand_as(logits)
                kept = logits.masked_fill(~keep, 0.0)
                logit_sum += float(kept.sum())
                logit_sq += float(kept.pow(2).sum())
                logit_n += float(keep.sum())
                logits = logits.masked_fill(~keep, float("-inf"))
            else:
                logit_sum += float(logits.sum())
                logit_sq += float(logits.pow(2).sum())
                logit_n += float(logits.numel())
            probs = logits.softmax(-1)
            entropy.append(-(probs * probs.clamp_min(1e-30).log()).sum(-1))
            max_prob.append(probs.max(-1).values)
            effective.append((-(probs * probs.clamp_min(1e-30).log()).sum(-1)).exp())
            if self_aligned:
                idx = torch.arange(s, e, device=probs.device)
                self_prob.append(probs.gather(1, idx[:, None]).squeeze(1))
            if out_chunks is not None:
                out_chunks.append(probs @ want_reconstruction[h])

    ent = torch.cat(entropy).mean().item()
    logit_mean = logit_sum / max(logit_n, 1.0)
    logit_var = max(logit_sq / max(logit_n, 1.0) - logit_mean**2, 0.0)
    stats = {
        "entropy_bits": ent / math.log(2),
        "entropy_nats": ent,
        "max_prob_mean": torch.cat(max_prob).mean().item(),
        "effective_support": torch.cat(effective).mean().item(),
        "uniform_entropy_bits": math.log2(Lk),
        "support_fraction": torch.cat(effective).mean().item() / Lk,
        "logit_mean": logit_mean,
        "logit_std": math.sqrt(logit_var),
    }
    if self_aligned:
        stats["self_prob_mean"] = torch.cat(self_prob).mean().item()
    if out_chunks is not None:
        stats["_reconstruction"] = torch.cat(out_chunks, dim=0)
    return stats


def split_heads(x: torch.Tensor, heads: int) -> torch.Tensor:
    b, t, c = x.shape
    return x.view(b, t, heads, c // heads).transpose(1, 2)


# --------------------------------------------------------------------------- #
# phase 1: structural gradient flow
# --------------------------------------------------------------------------- #
class Recorder:
    def __init__(self) -> None:
        self.handles: list = []
        self.captured: dict[str, torch.Tensor] = {}
        self.bwd: dict[str, dict] = {}
        self.activations: dict[str, dict] = {}
        self.gelu: dict[str, dict] = {}

    def capture_output(self, name: str, module: nn.Module) -> None:
        def hook(_m, _inp, out):
            if isinstance(out, torch.Tensor):
                if out.is_floating_point():
                    out.retain_grad()
                self.captured[name] = out
                self.activations[name] = {
                    "rms": rms(out),
                    "absmax": float(out.detach().abs().max()),
                    "frac_zero": float((out.detach() == 0).float().mean()),
                    "has_nan": bool(torch.isnan(out).any()),
                    "has_inf": bool(torch.isinf(out).any()),
                }

        self.handles.append(module.register_forward_hook(hook))

    def capture_input(self, name: str, module: nn.Module) -> None:
        def hook(_m, inp):
            t = inp[0]
            if isinstance(t, torch.Tensor) and t.is_floating_point():
                t.retain_grad()
                self.captured[name] = t

        self.handles.append(module.register_forward_pre_hook(hook))

    def backward_ratio(self, name: str, module: nn.Module) -> None:
        def hook(_m, grad_in, grad_out):
            gi, go = act_norm(grad_in), act_norm(grad_out)
            self.bwd[name] = {
                "grad_in": gi,
                "grad_out": go,
                "ratio": (gi / go) if go > 0 else float("nan"),
            }

        self.handles.append(module.register_full_backward_hook(hook))

    def gelu_hook(self, name: str, module: nn.Module) -> None:
        def fwd(_m, inp, _out):
            self.gelu[name] = {"frac_negative": float((inp[0].detach() < 0).float().mean())}

        self.handles.append(module.register_forward_hook(fwd))

    def close(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles.clear()


def instrument(model: PerturbationModel, rec: Recorder) -> None:
    rec.capture_output("gene_embedding", model.gene_embedding)
    rec.capture_output("chem_embedding", model.chem_embedding)
    rec.capture_output("conc_emb", model.conc_emb)
    rec.capture_input("part1_sdpa_out", model.conc_ffn)
    rec.capture_output("conc_ffn", model.conc_ffn)
    rec.capture_output("merger", model.merger)
    rec.backward_ratio("conc_emb", model.conc_emb)
    rec.backward_ratio("conc_ffn", model.conc_ffn)
    rec.backward_ratio("merger", model.merger)

    rec.backward_ratio("coordi_layer_first", model.coordi_layer[0])
    rec.backward_ratio("coordi_layer_last", model.coordi_layer[-1])

    for i, layer in enumerate(model.layers):
        p = f"L{i}."
        rec.capture_output(p + "out", layer)
        rec.capture_output(p + "q", layer.q_linear)
        rec.capture_output(p + "kv", layer.kv_linear)
        rec.capture_output(p + "chem_ffn", layer.chem_ffn)
        rec.capture_output(p + "kqv", layer.kqv)
        rec.capture_output(p + "gene_ffn", layer.gene_ffn)
        rec.backward_ratio(p + "block", layer)
        rec.backward_ratio(p + "q_linear", layer.q_linear)
        rec.backward_ratio(p + "kv_linear", layer.kv_linear)
        rec.backward_ratio(p + "kqv", layer.kqv)
        rec.backward_ratio(p + "chem_ffn", layer.chem_ffn)
        rec.backward_ratio(p + "gene_ffn", layer.gene_ffn)

    for name, mod in model.named_modules():
        if isinstance(mod, nn.GELU):
            rec.gelu_hook(name, mod)


def structural_flow(cfg: Config, device: torch.device) -> dict:
    free(device)
    model = build_model(cfg, device)
    model.eval()

    gen = torch.Generator().manual_seed(cfg.seed + 11)
    batch = make_batch(cfg, gen, device)

    control = batch["control"].clone().requires_grad_(True)
    chem = batch["chem"].clone().requires_grad_(True)
    coords = batch["coords"].clone().requires_grad_(True)

    rec = Recorder()
    instrument(model, rec)

    loss = perturbation_loss(
        model(control, chem, coords, batch["mask"]), batch["target"], 0.5
    )
    loss.backward()

    activations = {
        k: {
            "grad_norm": None if v.grad is None else float(v.grad.detach().float().norm()),
            "rms": rms(v),
            "grad_rms": None if v.grad is None else rms(v.grad),
        }
        for k, v in rec.captured.items()
    }

    # ---- how is the attention gradient split between q, k and v? -----------
    # (reveals whether a saturated/uniform softmax starves the q & k paths)
    attn_grad_split = {}
    for i in range(cfg.layers):
        gq = rec.captured[f"L{i}.q"].grad
        gkv = rec.captured[f"L{i}.kv"].grad
        gkqv = rec.captured[f"L{i}.kqv"].grad
        if gq is not None and gkv is not None:
            gk1, gv1 = gkv.chunk(2, dim=-1)
            attn_grad_split[f"L{i}.stage1"] = {
                "q_rms": rms(gq), "k_rms": rms(gk1), "v_rms": rms(gv1),
            }
        if gkqv is not None:
            gk2, gq2, gv2 = gkqv.chunk(3, dim=-1)   # chunk order is (k, q, v)
            attn_grad_split[f"L{i}.stage2"] = {
                "q_rms": rms(gq2), "k_rms": rms(gk2), "v_rms": rms(gv2),
            }

    # ---------------- attention statistics ---------------------------------
    attn = {}
    d = cfg.dim // cfg.heads
    with torch.no_grad():
        ctrl = control.detach()
        row_norm = ctrl.norm(dim=-1)                                     # (B, G)
        xn = ctrl / row_norm.unsqueeze(-1).clamp_min(1e-12)              # (B, G, C)
        concise = rec.captured["conc_emb"].detach()                      # (B, G, conc_dim)

        # part 1 -- mirror of the model's cosine kernel
        s1 = attention_stats(
            xn[0].unsqueeze(0), xn[0].unsqueeze(0), scale=10.0,
            want_reconstruction=concise[0].unsqueeze(0), self_aligned=True,
        )
        recon = s1.pop("_reconstruction")                                # (G, conc_dim)
        ref = rec.captured["part1_sdpa_out"].detach()[0]                 # (G, conc_dim)
        s1["reconstruction_max_abs_err"] = float((recon - ref).abs().max())
        s1["reconstruction_rel_err"] = float((recon - ref).norm() / ref.norm().clamp_min(1e-12))

        # cosine distribution on a 512-gene sample (full GxG would be 400M entries)
        sample = torch.randperm(cfg.genes)[: min(512, cfg.genes)]
        sim = xn[0][sample] @ xn[0].t()
        off = sim[~torch.eye(len(sample), cfg.genes, dtype=torch.bool)]
        s1["cos_mean"] = float(off.mean())
        s1["cos_std"] = float(off.std())
        s1["cos_p50"], s1["cos_p90"], s1["cos_p99"] = pctl(off)
        attn["part1_cosine_kernel"] = s1

        # part 2 stage 1 -- genes query chemicals
        q1 = split_heads(rec.captured["L0.q"].detach(), cfg.heads)[0]     # (H, G, d)
        kv = rec.captured["L0.kv"].detach()
        k1 = split_heads(kv.chunk(2, -1)[0], cfg.heads)[0]                # (H, Nc, d)
        m1 = batch["mask"].detach()[0].expand(cfg.heads, -1, -1)          # (H, 1, Nc)
        attn["part2_genes_to_chem"] = attention_stats(q1, k1, 1.0 / math.sqrt(d), mask=m1)

        # part 2 stage 2 -- genes <-> genes (chunk order is k, q, v)
        kk, qq, _vv = rec.captured["L0.kqv"].detach().chunk(3, -1)
        attn["part2_genes_to_genes"] = attention_stats(
            split_heads(qq, cfg.heads)[0], split_heads(kk, cfg.heads)[0],
            1.0 / math.sqrt(d), self_aligned=True,
        )

        # ---------------- silent-gene analysis ----------------------------
        silent = row_norm[0] <= 1e-12
        sq = control.grad.detach().pow(2).sum(-1)                        # (B, G)
        v_sq = rec.captured["conc_emb"].grad.detach().pow(2).sum(-1)     # (B, G)
        qk_total = float(sq.sum().sqrt())
        qk_silent = float(sq[..., silent].sum().sqrt()) if silent.any() else 0.0
        qk_clean = float(sq[..., ~silent].sum().sqrt()) if (~silent).any() else 0.0
        v_clean = float(v_sq[..., ~silent].sum().sqrt()) if (~silent).any() else 0.0
        per_gene = sq.sqrt().reshape(cfg.batch, cfg.genes)

    # ---------------- parameter gradients ----------------------------------
    param_stats = {}
    for name, p in model.named_parameters():
        pn = float(p.norm())
        if p.grad is None:
            param_stats[name] = {"grad_norm": 0.0, "p_norm": pn, "rel_to_param": 0.0,
                                 "grad_rms": 0.0, "is_zero": True}
            continue
        gn = float(p.grad.norm())
        param_stats[name] = {
            "grad_norm": gn,
            "p_norm": pn,
            "grad_rms": float(p.grad.pow(2).mean().sqrt()),
            "rel_to_param": gn / max(pn, 1e-30),
            "is_zero": gn == 0.0,
        }

    groups: dict[str, dict] = {}
    for name, st in param_stats.items():
        g = "layers" if name.startswith("layers.") else name.split(".")[0]
        agg = groups.setdefault(g, {"grad_norm": 0.0, "p_norm": 0.0, "n": 0, "n_zero": 0})
        agg["grad_norm"] += st["grad_norm"] ** 2
        agg["p_norm"] += st["p_norm"] ** 2
        agg["n"] += 1
        agg["n_zero"] += int(st["is_zero"])
    for agg in groups.values():
        agg["grad_norm"] = math.sqrt(agg["grad_norm"])
        agg["p_norm"] = math.sqrt(agg["p_norm"])
        agg["rel"] = agg["grad_norm"] / max(agg["p_norm"], 1e-30)

    emb_grad = rec.captured["gene_embedding"].grad
    total_param_grad = math.sqrt(
        sum(float(p.grad.norm()) ** 2 for p in model.parameters() if p.grad is not None)
    )
    top_param, top_abs = max(
        ((n, float(p.grad.abs().max())) for n, p in model.named_parameters() if p.grad is not None),
        key=lambda t: t[1],
    )
    result = {
        "config": asdict(cfg),
        "loss": float(loss),
        "activations": activations,
        "backward_ratios": rec.bwd,
        "gelu": rec.gelu,
        "attention": attn,
        "attention_grad_split": attn_grad_split,
        "param_grad_oneness": {
            "max_abs_over_norm": top_abs / max(total_param_grad, 1e-30),
            "top_parameter": top_param,
            "top_abs": top_abs,
            "total_norm": total_param_grad,
        },
        "params": param_stats,
        "param_groups": groups,
        "input_grads": {
            "control": {"grad_norm": float(control.grad.norm()), "rms": rms(control.grad)},
            "chem": {"grad_norm": float(chem.grad.norm()), "rms": rms(chem.grad)},
            "coords": {"grad_norm": float(coords.grad.norm()), "rms": rms(coords.grad)},
        },
        "gene_coverage": {
            "genes_total": int(cfg.genes),
            "silent_genes": int(silent.sum()),
            "genes_with_zero_input_grad": int((per_gene <= 1e-20).sum()),
            "input_grad_per_gene_min": float(per_gene.min()),
            "input_grad_per_gene_median": float(per_gene.median()),
            "input_grad_per_gene_max": float(per_gene.max()),
            "input_grad_max_over_median": float(
                per_gene.max() / per_gene.median().clamp_min(1e-30)
            ),
            "embedding_rows_with_zero_grad": (
                None if emb_grad is None else int((emb_grad.detach().norm(dim=-1) <= 1e-20).sum())
            ),
        },
        "key_ratios": {
            "part1_qk_grad_norm": qk_total,
            "part1_value_grad_norm": float(rec.captured["conc_emb"].grad.norm()),
            "part1_qk_over_value": qk_total / max(float(rec.captured["conc_emb"].grad.norm()), 1e-30),
            "silent_gene_share_of_qk_grad": qk_silent / max(qk_total, 1e-30),
            "part1_qk_over_value_nonsilent": qk_clean / max(v_clean, 1e-30),
            "part1_out_grad_norm": float(rec.captured["part1_sdpa_out"].grad.norm()),
        },
    }

    rec.close()
    del model
    free(device)
    return result


# --------------------------------------------------------------------------- #
# phase 3: training dynamics
# --------------------------------------------------------------------------- #
def training_run(cfg: Config, device: torch.device, optimizer: str = "adam") -> dict:
    free(device)
    model = build_model(cfg, device)
    model.train()

    lr = cfg.lr if optimizer == "adam" else cfg.sgd_lr
    opt = (
        torch.optim.Adam(model.parameters(), lr=lr)
        if optimizer == "adam"
        else torch.optim.SGD(model.parameters(), lr=lr)
    )
    gen = torch.Generator().manual_seed(cfg.seed + 101)
    params = dict(model.named_parameters())
    pool = build_pool(cfg, device, gen, min(cfg.steps, 8))

    watch = {
        "gene_embedding": "gene_embedding.weight",
        "conc_emb": "conc_emb.weight",
        "L0.q_linear": "layers.0.q_linear.weight",
        "L0.kqv": "layers.0.kqv.weight",
        "coordi_layer.0": "coordi_layer.0.weight",
    }

    steps = []
    with torch.no_grad():
        loss_before = float(perturbation_loss(
            model(pool[0]["control"], pool[0]["chem"], pool[0]["coords"], pool[0]["mask"]),
            pool[0]["target"], 0.5,
        ))

    for step in range(cfg.steps):
        batch = pool[step % len(pool)]
        control = batch["control"].clone().requires_grad_(True)

        opt.zero_grad(set_to_none=True)
        loss = perturbation_loss(
            model(control, batch["chem"], batch["coords"], batch["mask"]),
            batch["target"], 0.5,
        )
        loss.backward()

        total = math.sqrt(
            sum(float(p.grad.norm()) ** 2 for p in model.parameters() if p.grad is not None)
        )
        gin = input_grad_stats(control, control.grad)
        oneness = max(
            (float(p.grad.abs().max()) for p in model.parameters() if p.grad is not None),
            default=0.0,
        ) / max(total, 1e-30)
        record = {
            "step": step,
            "loss": float(loss),
            "total_grad_norm": total,
            "grad_oneness": oneness,
            "input_grad_norm": gin["total"],
            "input_grad_nonsilent": gin["nonsilent"],
            "silent_share": gin["silent_share"],
            "watched": {k: float(params[v].grad.norm()) for k, v in watch.items()},
            "has_nan_grad": any(
                (not torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None
            ),
        }

        if step == 0:
            before = {n: p.detach().clone() for n, p in params.items()}
        opt.step()
        if step == 0:
            record["update_ratio"] = {
                n: float((params[n].detach() - before[n]).norm()) / max(float(before[n].norm()), 1e-30)
                for n in before
            }

        steps.append(record)
        print(
            f"  [{optimizer}] step {step:3d}  loss {record['loss']:.6f}  "
            f"|g| {total:.4e}  |g_in| {gin['total']:.3e}  "
            f"|g_in|_clean {gin['nonsilent']:.3e}"
            f"{'  NONFINITE!' if record['has_nan_grad'] else ''}",
            flush=True,
        )

    # cross-batch gradient noise: per-group cosine + "oneness" (largest single
    # coordinate / norm). A gradient with oneness ~1 carries one scalar of
    # information no matter how many parameters it has.
    def grads_by_group(b):
        opt.zero_grad(set_to_none=True)
        perturbation_loss(
            model(b["control"], b["chem"], b["coords"], b["mask"]), b["target"], 0.5
        ).backward()
        out: dict[str, list] = {}
        for n, p in model.named_parameters():
            if p.grad is None:
                continue
            grp = "layers" if n.startswith("layers.") else n.split(".")[0]
            out.setdefault(grp, []).append(p.grad.reshape(-1).clone())
        return {g: torch.cat(v) for g, v in out.items()}

    G1 = grads_by_group(pool[0])
    G2 = grads_by_group(pool[1 % len(pool)])
    gnames = sorted(G1)
    cos_group = {g: float(F.cosine_similarity(G1[g], G2[g], dim=0)) for g in gnames}
    all1 = torch.cat([G1[g] for g in gnames])
    all2 = torch.cat([G2[g] for g in gnames])

    with torch.no_grad():
        loss_after = float(perturbation_loss(
            model(pool[0]["control"], pool[0]["chem"], pool[0]["coords"], pool[0]["mask"]),
            pool[0]["target"], 0.5,
        ))

    result = {
        "optimizer": optimizer,
        "lr": lr,
        "steps": steps,
        "eval_loss_before": loss_before,
        "eval_loss_after": loss_after,
        "eval_loss_drop": loss_before - loss_after,
        "final_loss": steps[-1]["loss"],
        "grad_norm_first": steps[0]["total_grad_norm"],
        "grad_norm_last": steps[-1]["total_grad_norm"],
        "diverged": (not math.isfinite(loss_after)) or loss_after > 50 * loss_before,
        "grad_noise": {
            "cross_batch_grad_cosine": float(F.cosine_similarity(all1, all2, dim=0)),
            "cross_batch_grad_cosine_by_group": cos_group,
            "grad_oneness": float(all1.abs().max() / all1.norm().clamp_min(1e-30)),
            "g1_norm": float(all1.norm()),
            "g2_norm": float(all2.norm()),
        },
    }
    del model
    free(device)
    return result


# --------------------------------------------------------------------------- #
# phase 4: depth sweep
# --------------------------------------------------------------------------- #
def depth_sweep(cfg: Config, device: torch.device) -> list[dict]:
    rows = []
    for layers in [int(x) for x in cfg.depth_sweep.split(",") if x.strip()]:
        free(device)
        model = build_model(cfg, device, layers=layers)
        model.eval()
        gen = torch.Generator().manual_seed(cfg.seed + 11)
        batch = make_batch(cfg, gen, device)
        control = batch["control"].clone().requires_grad_(True)

        loss = perturbation_loss(
            model(control, batch["chem"], batch["coords"], batch["mask"]),
            batch["target"], 0.5,
        )
        loss.backward()

        total = math.sqrt(
            sum(float(p.grad.norm()) ** 2 for p in model.parameters() if p.grad is not None)
        )
        gin = input_grad_stats(control, control.grad)
        rows.append({
            "num_layers": layers,
            "loss": float(loss),
            "total_grad_norm": total,
            "input_grad_norm": gin["total"],
            "input_grad_nonsilent": gin["nonsilent"],
            "part1_grad_norm": float(model.merger.weight.grad.norm()),
            "last_layer_grad_norm": float(model.layers[-1].gene_ffn[0].weight.grad.norm()),
        })
        print(
            f"  layers={layers:2d}  loss {loss:9.4f}  |g| {total:.4e}  "
            f"|g_in| {gin['total']:.4e}  |g_in|_clean {gin['nonsilent']:.4e}",
            flush=True,
        )
        del model
    free(device)
    return rows


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #
def fmt(x, spec=".4e"):
    if x is None:
        return "–"
    if isinstance(x, bool):
        return str(x)
    try:
        return format(float(x), spec)
    except (TypeError, ValueError):
        return str(x)


def pct(x):
    return "–" if x is None else f"{100 * x:6.2f}%"


def write_report(cfg, structure, trains, depth, out_dir, tag):
    L = []
    L.append("# Gradient-flow report — `models/model.py`")
    L.append("")
    L.append(f"- device `{cfg.device}` · float32 · seed {cfg.seed}")
    L.append(f"- genes **{cfg.genes:,}** · cells {cfg.cells} · batch {cfg.batch} · "
             f"chemicals {cfg.chems} · dim {cfg.dim} · conc_dim {cfg.conc_dim} · "
             f"heads {cfg.heads} · layers {cfg.layers}")
    L.append(f"- data `{cfg.data}` (zero-inflation {cfg.sparsity}) · dropout off during analysis")
    L.append("")

    kr = structure["key_ratios"]
    gc = structure["gene_coverage"]
    s1 = structure["attention"]["part1_cosine_kernel"]

    L.append("## 1. Headline")
    L.append("")
    L.append("| quantity | value |")
    L.append("|---|---|")
    L.append(f"| loss | {fmt(structure['loss'])} |")
    L.append(f"| ‖∂L/∂control‖ — Q/K path (all genes) | {fmt(kr['part1_qk_grad_norm'])} |")
    L.append(f"| ‖∂L/∂concise_emb‖ — V path | {fmt(kr['part1_value_grad_norm'])} |")
    L.append(f"| **Q/K ÷ V, all genes** | **{fmt(kr['part1_qk_over_value'])}** |")
    L.append(f"| **Q/K ÷ V, silent genes excluded** | **{fmt(kr['part1_qk_over_value_nonsilent'])}** |")
    L.append(f"| share of Q/K grad coming from silent genes | {pct(kr['silent_gene_share_of_qk_grad'])} |")
    L.append(f"| silent genes | {gc['silent_genes']} / {gc['genes_total']} |")
    L.append(f"| per-gene input-grad max ÷ median | {fmt(gc['input_grad_max_over_median'])} |")
    on = structure.get("param_grad_oneness", {})
    if on:
        L.append(f"| gradient oneness max\\|g\\|/‖g‖ | {fmt(on['max_abs_over_norm'], '.4f')} |")
        L.append(f"| largest gradient coordinate | `{on['top_parameter']}` |")
    L.append("")

    L.append("## 2. Part-1 cosine kernel")
    L.append("")
    L.append("| metric | value |")
    L.append("|---|---|")
    L.append(f"| cosine mean / std | {fmt(s1['cos_mean'], '+.3f')} / {fmt(s1['cos_std'], '.3f')} |")
    L.append(f"| cosine p50 / p90 / p99 | {fmt(s1['cos_p50'], '+.3f')} / {fmt(s1['cos_p90'], '+.3f')} / {fmt(s1['cos_p99'], '+.3f')} |")
    L.append(f"| softmax entropy | {fmt(s1['entropy_bits'], '.3f')} bits (uniform {fmt(s1['uniform_entropy_bits'], '.2f')}) |")
    L.append(f"| mean max prob | {fmt(s1['max_prob_mean'], '.4f')} |")
    L.append(f"| mean self prob (diagonal) | {fmt(s1.get('self_prob_mean'), '.4f')} |")
    L.append(f"| effective support | {fmt(s1['effective_support'], '.1f')} genes ({pct(s1['support_fraction'])}) |")
    L.append(f"| rebuilt attn@V vs real SDPA — max abs err | {fmt(s1['reconstruction_max_abs_err'])} |")
    L.append(f"| rebuilt attn@V vs real SDPA — rel err | {fmt(s1['reconstruction_rel_err'])} |")
    L.append("")

    L.append("## 3. Part-2 attention")
    L.append("")
    L.append("| SDPA call | entropy bits | uniform | logit std | max prob | self prob | eff. support | support % |")
    L.append("|---|---|---|---|---|---|---|---|")
    for key, label in [("part2_genes_to_chem", "genes → chemicals"),
                       ("part2_genes_to_genes", "genes ↔ genes")]:
        a = structure["attention"][key]
        L.append(
            f"| {label} | {fmt(a['entropy_bits'], '.3f')} | {fmt(a['uniform_entropy_bits'], '.2f')} | "
            f"{fmt(a['logit_std'], '.4f')} | {fmt(a['max_prob_mean'], '.4f')} | "
            f"{fmt(a.get('self_prob_mean'), '.4f')} | {fmt(a['effective_support'], '.1f')} | "
            f"{pct(a['support_fraction'])} |"
        )
    L.append("")

    splits = structure.get("attention_grad_split", {})
    if splits:
        L.append("## 3b. Attention gradient split (RMS of ∂L/∂q, ∂L/∂k, ∂L/∂v)")
        L.append("")
        L.append("A uniform softmax has a near-zero Jacobian, so it starves q and k.")
        L.append("")
        L.append("| layer.stage | q RMS | k RMS | v RMS | q/v |")
        L.append("|---|---|---|---|---|")
        for name, st in splits.items():
            qv = st["q_rms"] / st["v_rms"] if st["v_rms"] > 0 else float("nan")
            L.append(f"| `{name}` | {fmt(st['q_rms'])} | {fmt(st['k_rms'])} | "
                     f"{fmt(st['v_rms'])} | {fmt(qv, '.3e')} |")
        L.append("")

    L.append("## 4. Stage-by-stage gradient norms")
    L.append("")
    L.append("| stage | activation RMS | ‖∂L/∂stage‖ | grad RMS |")
    L.append("|---|---|---|---|")
    order = ["gene_embedding", "chem_embedding", "conc_emb", "part1_sdpa_out",
             "conc_ffn", "merger"]
    for i in range(cfg.layers):
        order += [f"L{i}.q", f"L{i}.kv", f"L{i}.chem_ffn", f"L{i}.kqv",
                  f"L{i}.gene_ffn", f"L{i}.out"]
    for name in order:
        a = structure["activations"].get(name)
        if a is None:
            continue
        L.append(f"| `{name}` | {fmt(a['rms'])} | {fmt(a['grad_norm'])} | {fmt(a['grad_rms'])} |")
    L.append("")

    L.append("## 5. Local transmission ratios (‖grad_in‖ / ‖grad_out‖)")
    L.append("")
    L.append("Ratio ≪ 1 = attenuating block; ratio > 1 = amplifying block.")
    L.append("")
    L.append("| block | ‖grad_in‖ | ‖grad_out‖ | ratio |")
    L.append("|---|---|---|---|")
    for name, st in structure["backward_ratios"].items():
        L.append(f"| `{name}` | {fmt(st['grad_in'])} | {fmt(st['grad_out'])} | {fmt(st['ratio'], '.4f')} |")
    L.append("")

    L.append("## 6. Parameter gradient norms")
    L.append("")
    L.append("| group | ‖g‖ | ‖p‖ | ‖g‖/‖p‖ | zero-grad |")
    L.append("|---|---|---|---|---|")
    for name, st in structure["param_groups"].items():
        L.append(f"| `{name}` | {fmt(st['grad_norm'])} | {fmt(st['p_norm'])} | "
                 f"{fmt(st['rel'], '.3e')} | {st['n_zero']}/{st['n']} |")
    L.append("")

    L.append("## 7. Gene / unit coverage")
    L.append("")
    L.append("| metric | value |")
    L.append("|---|---|")
    L.append(f"| genes with zero ∂L/∂input | {gc['genes_with_zero_input_grad']} / {gc['genes_total']} |")
    L.append(f"| embedding rows with zero grad | {gc['embedding_rows_with_zero_grad']} |")
    L.append(f"| per-gene input grad min / median / max | {fmt(gc['input_grad_per_gene_min'])} / "
             f"{fmt(gc['input_grad_per_gene_median'])} / {fmt(gc['input_grad_per_gene_max'])} |")
    L.append("")

    if structure["gelu"]:
        L.append("| GELU site | fraction of negative inputs |")
        L.append("|---|---|")
        for name, st in structure["gelu"].items():
            L.append(f"| `{name}` | {pct(st['frac_negative'])} |")
        L.append("")

    L.append("## 8. Training dynamics")
    L.append("")
    for run in trains:
        L.append(f"### {run['optimizer']} (lr {run['lr']})")
        L.append("")
        L.append(f"- held-out well loss {fmt(run['eval_loss_before'], '.6f')} → "
                 f"{fmt(run['eval_loss_after'], '.6f')} (drop {fmt(run['eval_loss_drop'], '.6f')})")
        L.append(f"- total grad norm {fmt(run['grad_norm_first'])} → {fmt(run['grad_norm_last'])}")
        L.append(f"- **diverged: {run['diverged']}**")
        gn = run["grad_noise"]
        L.append(f"- cross-batch grad cosine {fmt(gn['cross_batch_grad_cosine'], '.4f')} "
                 f"(1 = consistent, 0 = noise, <0 = opposite)")
        L.append(f"- gradient oneness max|g|/‖g‖ = {fmt(gn['grad_oneness'], '.4f')} "
                 f"(≈1 ⇒ the gradient is one scalar, not a direction)")
        L.append("")
        L.append("| parameter group | cross-batch grad cosine |")
        L.append("|---|---|")
        for gname, c in gn["cross_batch_grad_cosine_by_group"].items():
            L.append(f"| `{gname}` | {fmt(c, '+.4f')} |")
        L.append("")
        L.append("| step | loss | ‖g‖ | oneness | ‖∂L/∂control‖ | ‖∂L/∂control‖ (expressed genes) |")
        L.append("|---|---|---|---|---|---|")
        for st in run["steps"]:
            L.append(f"| {st['step']} | {fmt(st['loss'], '.6f')} | {fmt(st['total_grad_norm'])} | "
                     f"{fmt(st.get('grad_oneness'), '.3f')} | {fmt(st['input_grad_norm'])} | "
                     f"{fmt(st.get('input_grad_nonsilent'))} |")
        L.append("")
        if "update_ratio" in run["steps"][0]:
            L.append("| parameter | ‖Δp‖/‖p‖ (step 0) |")
            L.append("|---|---|")
            for n, v in run["steps"][0]["update_ratio"].items():
                L.append(f"| `{n}` | {fmt(v, '.3e')} |")
            L.append("")

    if depth:
        L.append("## 9. Depth sweep")
        L.append("")
        L.append("| layers | loss | ‖g‖ | ‖∂L/∂control‖ | expressed-gene ‖∂L/∂control‖ | ‖∂L/∂merger.W‖ | newest layer gene_ffn.W |")
        L.append("|---|---|---|---|---|---|---|")
        for r in depth:
            L.append(f"| {r['num_layers']} | {fmt(r['loss'], '.4f')} | {fmt(r['total_grad_norm'])} | "
                     f"{fmt(r['input_grad_norm'])} | {fmt(r.get('input_grad_nonsilent'))} | "
                     f"{fmt(r['part1_grad_norm'])} | {fmt(r['last_layer_grad_norm'])} |")
        L.append("")

    path = os.path.join(out_dir, f"gradient_flow_report{tag}.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L))
    return path


def write_plots(cfg, structure, trains, depth, out_dir, tag):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return []

    made = []

    order = ["gene_embedding", "conc_emb", "part1_sdpa_out", "conc_ffn", "merger"]
    for i in range(cfg.layers):
        order += [f"L{i}.q", f"L{i}.kqv", f"L{i}.out"]
    names = [n for n in order if n in structure["activations"]]
    vals = [structure["activations"][n]["grad_norm"] or 0.0 for n in names]

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.barh(range(len(names)), vals)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("‖∂L/∂activation‖ (log)")
    ax.set_title("Backward signal through the graph (eval)")
    ax.invert_yaxis()
    fig.tight_layout()
    p = os.path.join(out_dir, f"grad_by_stage{tag}.png")
    fig.savefig(p, dpi=130)
    plt.close(fig)
    made.append(p)

    groups = structure["param_groups"]
    fig, ax = plt.subplots(figsize=(8, 4))
    gnames = list(groups)
    ax.barh(range(len(gnames)), [groups[g]["grad_norm"] for g in gnames])
    ax.set_yticks(range(len(gnames)))
    ax.set_yticklabels(gnames, fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("‖g‖ (log)")
    ax.set_title("Parameter gradient norms")
    ax.invert_yaxis()
    fig.tight_layout()
    p = os.path.join(out_dir, f"grad_by_param_group{tag}.png")
    fig.savefig(p, dpi=130)
    plt.close(fig)
    made.append(p)

    for run in trains:
        st = run["steps"]
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
        axes[0].plot([s["step"] for s in st], [s["loss"] for s in st])
        axes[0].set_title(f"loss ({run['optimizer']})")
        axes[1].plot([s["step"] for s in st], [s["total_grad_norm"] for s in st])
        axes[1].set_yscale("log")
        axes[1].set_title("‖g‖ total")
        axes[2].plot([s["step"] for s in st], [s["input_grad_norm"] for s in st])
        axes[2].set_yscale("log")
        axes[2].set_title("‖∂L/∂control‖")
        for a in axes:
            a.set_xlabel("step")
        fig.tight_layout()
        p = os.path.join(out_dir, f"training_curves_{run['optimizer']}{tag}.png")
        fig.savefig(p, dpi=130)
        plt.close(fig)
        made.append(p)

    if depth:
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
        ls = [r["num_layers"] for r in depth]
        axes[0].plot(ls, [r["input_grad_norm"] for r in depth], marker="o")
        axes[0].set_title("‖∂L/∂control‖ vs depth")
        axes[1].plot(ls, [r["total_grad_norm"] for r in depth], marker="o")
        axes[1].set_title("‖g‖ total vs depth")
        axes[2].plot(ls, [r["loss"] for r in depth], marker="o")
        axes[2].set_title("loss vs depth")
        for a in axes:
            a.set_xlabel("num_layers")
        fig.tight_layout()
        p = os.path.join(out_dir, f"depth_sweep{tag}.png")
        fig.savefig(p, dpi=130)
        plt.close(fig)
        made.append(p)

    return made


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description="Gradient-flow diagnostics for models/model.py")
    ap.add_argument("--genes", type=int, default=20000)
    ap.add_argument("--cells", type=int, default=32)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--chems", type=int, default=10)
    ap.add_argument("--chem-dim", type=int, default=128)
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--conc-dim", type=int, default=32)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--sparsity", type=float, default=0.70)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--sgd-lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--data", default="structured", choices=["structured", "noise", "both"])
    ap.add_argument("--depth-sweep", default="1,2,4,8")
    ap.add_argument("--out-dir", default=HERE)
    ap.add_argument("--tag", default="")
    ap.add_argument("--quick", action="store_true", help="structure phase only")
    ap.add_argument("--no-sgd", action="store_true", help="skip the SGD divergence probe")
    args = ap.parse_args()

    cfg = Config(
        genes=args.genes, cells=args.cells, batch=args.batch, chems=args.chems,
        chem_dim=args.chem_dim, dim=args.dim, conc_dim=args.conc_dim, heads=args.heads,
        layers=args.layers, dropout=args.dropout, sparsity=args.sparsity, steps=args.steps,
        lr=args.lr, sgd_lr=args.sgd_lr, seed=args.seed, device=args.device, data=args.data,
        depth_sweep=args.depth_sweep, out_dir=args.out_dir, tag=args.tag,
    )
    device = resolve_device(cfg.device)
    cfg.device = str(device)
    os.makedirs(cfg.out_dir, exist_ok=True)

    t0 = time.time()
    print(f"[device] {device}  genes={cfg.genes:,} cells={cfg.cells} batch={cfg.batch}")

    modes = ["structured", "noise"] if cfg.data == "both" else [cfg.data]
    structures = {}
    for mode in modes:
        cfg_mode = Config(**{**asdict(cfg), "data": mode})
        print(f"\n[1/4] structural flow ({mode}) — eval, 1 fwd/bwd")
        structures[mode] = structural_flow(cfg_mode, device)
        kr = structures[mode]["key_ratios"]
        print(f"      loss {structures[mode]['loss']:.4f}")
        print(f"      Q/K÷V all genes      = {fmt(kr['part1_qk_over_value'])}")
        print(f"      Q/K÷V non-silent     = {fmt(kr['part1_qk_over_value_nonsilent'])}")
        print(f"      silent share of Q/K  = {pct(kr['silent_gene_share_of_qk_grad'])}")
        s1 = structures[mode]["attention"]["part1_cosine_kernel"]
        print(f"      part1 cos mean={s1['cos_mean']:+.3f}  self-prob={s1['self_prob_mean']:.4f}  "
              f"entropy={s1['entropy_bits']:.2f}/{s1['uniform_entropy_bits']:.2f} bits")

    cfg.data = modes[0]
    trains = []
    depth = []
    if not args.quick:
        print(f"\n[2/4] training {cfg.steps} steps with Adam (lr={cfg.lr})")
        trains.append(training_run(cfg, device, "adam"))
        if not args.no_sgd:
            print(f"\n[2b/4] training {cfg.steps} steps with SGD (lr={cfg.sgd_lr})")
            trains.append(training_run(cfg, device, "sgd"))
        print(f"\n[3/4] depth sweep {cfg.depth_sweep}")
        depth = depth_sweep(cfg, device)
    else:
        print("\n[2/4] training skipped (--quick)")
        print("[3/4] depth sweep skipped (--quick)")

    print("\n[4/4] writing report")
    primary = structures[modes[0]]
    md = write_report(cfg, primary, trains, depth, cfg.out_dir, cfg.tag)
    plots = write_plots(cfg, primary, trains, depth, cfg.out_dir, cfg.tag)

    payload = {
        "config": asdict(cfg),
        "structures": structures,
        "trains": trains,
        "depth": depth,
        "elapsed_s": time.time() - t0,
    }
    js = os.path.join(cfg.out_dir, f"gradient_flow_report{cfg.tag}.json")
    with open(js, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=str)

    print(f"      report : {md}")
    print(f"      json   : {js}")
    for p in plots:
        print(f"      plot   : {p}")
    print(f"\ndone in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
