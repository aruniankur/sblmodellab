"""DMSO-only reconstruction model: control expression and coordinates in,
per-cell gene expression out. There are no chemical tokens or perturbation blocks.

B: wells, G: genes, C_in: context cells, C_out: queried cells.
Context shape: (B, G, C_in); coordinates: (B, C_out, 2).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def _chunked_sdpa(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    scale: float | None = None,
    query_chunk: int | None = None,
) -> torch.Tensor:
    """``scaled_dot_product_attention`` with the query axis optionally split.

    Splitting the query axis is mathematically exact: every query row attends to
    the same full key/value set, so only the peak size of the
    ``(B, heads, L, S)`` score tensor changes. It exists because with
    ``num_of_genes = 50266`` the built-in kernels fall back to the math backend
    for a wide query/key embedding (``head_dim = num_control_cells``), which
    materialises ~40 GB of scores. Each chunk is re-computed in the backward
    pass, so memory stays at one chunk instead of one per chunk.

    ``query_chunk=None`` (the default) is the original unmodified code path.
    """
    if query_chunk is None or query.shape[-2] <= query_chunk:
        return F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            scale=scale,
        )

    def _attend(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, scale=scale
        )

    needs_grad = torch.is_grad_enabled() and any(
        t.requires_grad for t in (query, key, value)
    )

    pieces = []
    for start in range(0, query.shape[-2], query_chunk):
        query_piece = query[..., start : start + query_chunk, :]
        if needs_grad:
            pieces.append(checkpoint(_attend, query_piece, key, value, use_reentrant=False))
        else:
            pieces.append(_attend(query_piece, key, value))

    return torch.cat(pieces, dim=-2)

class ControlPerturbationModel(nn.Module):
    """Reconstruct DMSO expression at supplied DMSO cell coordinates.

    First mix learned gene embeddings using cosine attention over control-cell
    expression profiles. Then combine each gene state with each query coordinate
    and predict expression through the readout MLP. Context and target cells may
    overlap or come from different DMSO wells of the same plate and cell line.

    num_heads, dropout and num_layers are compatibility arguments only; this
    architecture has no ExternalPerturbAttention stack and does not use them.
    """

    def __init__(
        self,
        dim: int = 1024,
        conc_dim: int = 128,
        num_heads: int = 8,
        dropout: float = 0.1,
        num_layers: int = 4,
        num_of_genes: int = 1000,
        cell_coordi_dim: int = 2,
    ):
        super().__init__()

        self.num_genes = num_of_genes

        self.gene_embedding = nn.Embedding(num_of_genes, dim)

        self.conc_emb = nn.Linear(dim, conc_dim)

        self.conc_ffn = nn.Sequential(
            nn.Linear(conc_dim, dim ),
            nn.GELU(),
            nn.Linear(dim , conc_dim),
        )

        self.merger = nn.Linear(dim + conc_dim, dim)

        self.coordi_layer = nn.Sequential(
            nn.Linear(dim + cell_coordi_dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, dim // 4),
            nn.GELU(),
            nn.Linear(dim // 4, 1),
        )

    def forward(
        self,
        control_cell_line: torch.Tensor,
        cell_coordi: torch.Tensor,
        query_chunk: int | None = None,
        readout_chunk: int | None = None,
    ) -> torch.Tensor:
        """Accept (B, G, C_in) context and (B, C_out, 2) coordinates.

        Return (B, C_out, G) expression. Optional query/readout chunks bound
        memory through tiling and activation recomputation.
        """
        # --- unit-normalise each gene's expression profile over the cell axis ---
        mean_value = control_cell_line.mean(axis=1)
        gene_norms = torch.linalg.norm(control_cell_line, dim=-1, keepdim=True)
        control_cell_line = control_cell_line / gene_norms.clamp_min(1e-12)
        
        batch_size = control_cell_line.shape[0]

        gene_embeddings = (
            self.gene_embedding(
                torch.arange(self.num_genes, device=control_cell_line.device)
            )
            .unsqueeze(0)
            .expand(batch_size, -1, -1)
        )

        # --- part 1: co-expression context via cosine kernel ---
        # Queries/keys live in the control-cell expression space; the values
        # come from the learned embeddings, so the kernel measures co-expression.
        concise_gene_embeddings = self.conc_emb(gene_embeddings)

        coexpression_context = _chunked_sdpa(
            control_cell_line,
            control_cell_line,
            concise_gene_embeddings,
            scale=10,
            dropout_p=0,
            query_chunk=query_chunk,
        )

        coexpression_context = self.conc_ffn(coexpression_context)

        gene_states = self.merger(
            torch.cat([gene_embeddings, coexpression_context], dim=-1)
        )

        # --- part 2: coordinate-conditioned readout ---
        if readout_chunk is None:
            coords_expanded = cell_coordi.unsqueeze(2).expand(
                -1, -1, gene_states.shape[1], -1
            )
            states_expanded = gene_states.unsqueeze(1).expand(
                -1, cell_coordi.shape[1], -1, -1
            )

            readout_features = torch.cat([coords_expanded, states_expanded], dim=-1)

            predicted_expression = self.coordi_layer(readout_features).squeeze(-1)
        else:
            def _readout(coords: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
                coords_expanded = coords.unsqueeze(2).expand(
                    -1, -1, states.shape[1], -1
                )
                states_expanded = states.unsqueeze(1).expand(
                    -1, coords.shape[1], -1, -1
                )
                features = torch.cat([coords_expanded, states_expanded], dim=-1)
                return self.coordi_layer(features).squeeze(-1)

            needs_grad = torch.is_grad_enabled() and gene_states.requires_grad

            pieces = []
            for start in range(0, cell_coordi.shape[1], readout_chunk):
                coords_piece = cell_coordi[:, start : start + readout_chunk]
                if needs_grad:
                    pieces.append(
                        checkpoint(_readout, coords_piece, gene_states, use_reentrant=False)
                    )
                else:
                    pieces.append(_readout(coords_piece, gene_states))

            predicted_expression = torch.cat(pieces, dim=1)

        return predicted_expression


def perturbation_loss(
    predicted_expression: torch.Tensor,
    target_expression: torch.Tensor,
    scale_exponent: float = 0.5,
) -> torch.Tensor:
    """Sum-over-genes squared error, meaned over cells and batch, then scaled.

    ``scale_exponent`` (``z`` in the notebook) divides the loss by
    ``num_genes ** z`` so the magnitude does not grow with the panel size.
    """
    num_genes = predicted_expression.shape[-1]

    loss = (target_expression - predicted_expression) ** 2
    loss = loss.sum(dim=-1)  # sum across genes
    loss = loss.mean(dim=-1)  # mean across cells
    loss = loss.mean()  # mean across batch

    return loss / (num_genes**scale_exponent)


# Backwards-compatible alias (the notebook uses the old, misspelled class name).
pertubationmodel = ControlPerturbationModel


if __name__ == "__main__":
    # Smoke test using DMSO context and queried DMSO coordinates only.
    torch.manual_seed(0)

    batch, num_genes, num_cells = 2, 1000, 5
    control_cell_line = torch.rand(batch, num_genes, 300)

    cell_coordi = torch.randn(batch, num_cells, 2)
    target_expression = torch.randn(batch, num_cells, num_genes)

    model = ControlPerturbationModel(
        dim=256,
        conc_dim=32,
        num_heads=4,
        num_layers=2,
        num_of_genes=num_genes,
        cell_coordi_dim=2,
    )

    predicted_expression = model(control_cell_line, cell_coordi)
    loss = perturbation_loss(predicted_expression, target_expression, 0.75)

    print("predicted_expression:", tuple(predicted_expression.shape))
    print("loss:", loss.item())
