"""What does one atom vector do that one pooled drug vector cannot?

Runs stage 1 (genes query the chemical) of the *same* trained-shape model twice:
once with the drug as a single mean-pooled vector, once with the drug as its set
of per-atom vectors. Everything else -- real control cells, real gene states,
real weights -- is identical.

    python models/testing/atom_vs_pooled.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from models.model import PerturbationModel
from models.molformer_atoms import AtomEmbeddingStore, dose_to_scalar
from models.tahoe_data import BackedFile, build_example, build_file_index


def stage1(model, layer, gene_states, chem, mask):
    """Copy of ExternalPerturbAttention stage 1 -> (B, G, dim)."""
    key_value = layer.kv_linear(chem)
    keys, values = key_value.chunk(2, dim=-1)
    queries = layer.q_linear(gene_states)

    q = layer.split_heads(queries)
    k = layer.split_heads(keys)
    v = layer.split_heads(values)

    context = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0)
    context = layer.merge_heads(context)
    return layer.chem_ffn(context), q, k


def attention_weights(q, k):
    scores = q @ k.transpose(-2, -1) / (q.shape[-1] ** 0.5)
    return scores.softmax(-1)


def main() -> None:
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    index = build_file_index()
    backed = BackedFile(index[0].path, 128)
    store = AtomEmbeddingStore()

    rng = np.random.default_rng(0)
    example = build_example(
        backed, backed.control_wells[0], backed.drug_wells(128)[0], 256, 128, rng
    )
    backed.close()

    n_genes = example.control.shape[0]
    model = PerturbationModel(
        dim=512, conc_dim=32, num_heads=8, num_layers=1,
        num_of_genes=n_genes, chem_dim=store.dim + 1, cell_coordi_dim=2,
    ).to(device).eval()

    # ---- gene states: the real part-1 + merger output --------------------
    with torch.no_grad():
        control = example.control.unsqueeze(0).to(device)
        normalised = control / torch.linalg.norm(control, dim=-1, keepdim=True).clamp_min(1e-12)
        gene_embeddings = (
            model.gene_embedding(torch.arange(n_genes, device=device)).unsqueeze(0)
        )
        concise = model.conc_emb(gene_embeddings)
        # part 1, with the same memory-safe chunking the trainer uses
        from models.model import _chunked_sdpa

        ctx = _chunked_sdpa(
            normalised, normalised, concise, scale=10, dropout_p=0, query_chunk=4096
        )
        ctx = model.conc_ffn(ctx)
        gene_states = model.merger(torch.cat([gene_embeddings, ctx], dim=-1))

    atom_vectors = store.get(example.drug)          # (A, 768)
    dose = dose_to_scalar(example.concentration)
    n_atoms = atom_vectors.shape[0]

    variants = {
        "pooled (1 token)": atom_vectors.mean(0, keepdim=True),
        f"atoms ({n_atoms} tokens)": atom_vectors,
    }

    print(f"drug: {example.drug}   atoms: {n_atoms}   dose: {example.concentration:g} "
          f"(log10 {dose:.3f})   genes: {n_genes}")

    for label, vectors in variants.items():
        n_tokens = vectors.shape[0]
        chem = torch.cat(
            [vectors, torch.full((n_tokens, 1), dose)], dim=-1
        ).unsqueeze(0).to(device)
        mask = torch.ones(1, 1, 1, n_tokens, dtype=torch.bool, device=device)

        with torch.no_grad():
            chem_embeddings = model.chem_embedding(chem)
            context, q, k = stage1(model, model.layers[0], gene_states, chem_embeddings, mask)
            weights = attention_weights(q, k)  # (1, heads, G, tokens)

        per_gene_spread = context[0] - context[0].mean(0, keepdim=True)
        print(f"\n--- {label} ---")
        print(f"  context across the {n_genes} genes:")
        print(f"    max |deviation from the mean gene|: "
              f"{per_gene_spread.abs().max().item():.3e}")
        print(f"    distinct rows                   : "
              f"{len(torch.unique(context[0].round(decimals=5), dim=0))}")
        if n_tokens > 1:
            top = weights.amax(-1)          # each gene's strongest atom
            print(f"  attention over atoms, per gene:")
            print(f"    entropy {(-(weights * weights.clamp_min(1e-12).log()).sum(-1)).mean().item():.3f} "
                  f"nats (max {np.log(n_tokens):.3f} = uniform)")
            print(f"    strongest-atom weight  mean {top.mean().item():.4f} "
                  f"min {top.min().item():.4f} max {top.max().item():.4f}")
            head0 = weights.argmax(-1)[0, 0]  # strongest atom per gene, head 0
            counts = torch.bincount(head0, minlength=n_tokens)
            print(f"    genes sharing the same strongest atom: "
                  f"most popular atom chosen by {counts.max().item()} of {n_genes} genes, "
                  f"atoms never chosen: {(counts == 0).sum().item()}")

    print("\nstage-1 cost per layer:")
    print(f"  pooled : G x 1   = {n_genes * 1:,} pairs")
    print(f"  atoms  : G x {n_atoms}  = {n_genes * n_atoms:,} pairs")


if __name__ == "__main__":
    main()
