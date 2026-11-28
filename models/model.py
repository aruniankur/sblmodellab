"""Chemical perturbation model for coordinate-conditioned gene expression.

Combines control-derived co-expression context, chemical cross-attention,
gene self-attention and a coordinate-conditioned expression readout.

Shapes used throughout
----------------------
B   : batch size (one well = one plate x drug x dose)
G   : number of genes
C   : number of cells
Nc  : number of chemicals (drug features per well)
F   : chemical feature width (``chem_dim``)
dim : model width
conc_dim : concentration / co-expression bottleneck width
d   : per-head width (``dim // num_heads``)
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

if __package__ in (None, ""):
    from expression_features import NUM_GENE_STATS, gene_expression_statistics
else:
    from .expression_features import NUM_GENE_STATS, gene_expression_statistics


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


class ExternalPerturbAttention(nn.Module):
    """Two-stage attention block: genes query chemicals, then genes query genes.

    Stage 1 (chemical conditioning)
        q <- genes, k/v <- chemicals  (cross-attention, masked to valid chemicals)

    Stage 2 (gene self-attention)
        q/k/v <- cat([genes, chemical context])  (no mask)

    New blocks normalize only the chemical and gene FFN updates.
    Chemical states are normalised once by the parent model before the stack.
    Projected Q/K/V go directly to head splitting without normalization.
    Chemical attention has an optional residual around its FFN, followed by
    normalization. An optional per-layer sigmoid gate adds chemical context
    directly to the gene update before post-residual normalization.
    """

    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1,
                 normalization: bool = True, attention_normalization: bool = True,
                 chem_ffn_residual: bool = True, ffn_multiplier: int = 2,
                 joint_normalization: bool = False, chem_skip_gate: bool = True,
                 chem_skip_init: float = 0.1, legacy_attention_norms: bool = False):
        super().__init__()

        assert dim % num_heads == 0, "dim must be divisible by num_heads"

        if isinstance(ffn_multiplier, bool) or not isinstance(ffn_multiplier, int) or ffn_multiplier < 1:
            raise ValueError("ffn_multiplier must be a positive integer")
        ffn_dim = dim * ffn_multiplier
        if not isinstance(chem_skip_gate, bool):
            raise ValueError("chem_skip_gate must be a boolean")
        if (isinstance(chem_skip_init, bool) or not isinstance(chem_skip_init, (int, float))
                or not 0 < chem_skip_init < 1):
            raise ValueError("chem_skip_init must be finite and strictly between 0 and 1")
        # One scalar per block. Sigmoid bounds the direct chemical contribution.
        if chem_skip_gate:
            self.chem_skip_gate_logit = nn.Parameter(torch.tensor(math.log(chem_skip_init / (1 - chem_skip_init))))
        else:
            self.register_parameter("chem_skip_gate_logit", None)

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.dropout = dropout
        self.use_chem_ffn_residual = chem_ffn_residual

        # FFN norms remain; projected Q/K/V norms exist only for old checkpoints.
        norm = lambda: nn.LayerNorm(dim) if normalization and attention_normalization else nn.Identity()
        self.legacy_attention_norms = legacy_attention_norms
        if legacy_attention_norms:
            # Preserve historical registration order for saved AdamW slots.
            self.chem_q_norm = norm()
            self.chem_k_norm = norm()
            self.chem_v_norm = norm()
        self.chem_ffn_norm = norm()
        if legacy_attention_norms:
            self.gene_q_norm = norm()
            self.gene_k_norm = norm()
            self.gene_v_norm = norm()
        self.gene_ffn_norm = norm()

        # --- stage 1: chemical -> gene cross attention ---
        self.kv_linear = nn.Linear(dim, dim * 2)
        self.q_linear = nn.Linear(dim, dim)
        self.query_norm = nn.LayerNorm(dim) if normalization and legacy_attention_norms else nn.Identity()

        self.chem_ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, dim),
        )

        # --- stage 2: gene -> gene self attention ---
        self.kqv = nn.Linear(dim * 2, dim * 3)
        self.joint_norm = nn.LayerNorm(dim * 2) if normalization and legacy_attention_norms and joint_normalization else nn.Identity()

        self.gene_ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, dim),
        )

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, dim) -> (B, num_heads, T, head_dim)."""
        batch, tokens, channels = x.shape
        x = x.view(batch, tokens, self.num_heads, self.head_dim)
        return x.transpose(1, 2)

    def merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """(B, num_heads, T, head_dim) -> (B, T, dim)."""
        batch, heads, tokens, head_dim = x.shape
        return x.transpose(1, 2).contiguous().view(batch, tokens, heads * head_dim)

    def forward(
        self,
        x: torch.Tensor,
        chem_embeddings: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """x: (B, G, dim) gene states, chem_embeddings: (B, Nc, dim)."""
        attn_dropout = self.dropout if self.training else 0.0

        # --- stage 1: genes attend over the chemicals ---
        key_value = self.kv_linear(chem_embeddings)
        keys, values = key_value.chunk(2, dim=-1)

        query_input = self.query_norm(x) if self.legacy_attention_norms else x
        queries = self.q_linear(query_input)
        if self.legacy_attention_norms:
            queries = self.chem_q_norm(queries)
            keys = self.chem_k_norm(keys)
            values = self.chem_v_norm(values)

        queries = self.split_heads(queries)
        keys = self.split_heads(keys)
        values = self.split_heads(values)

        chem_context = F.scaled_dot_product_attention(
            queries,
            keys,
            values,
            attn_mask=attn_mask,
            dropout_p=attn_dropout,
        )

        chem_context = self.merge_heads(chem_context)
        chem_update = self.chem_ffn(chem_context)
        if self.use_chem_ffn_residual:
            chem_update = chem_context + chem_update
        chem_context = self.chem_ffn_norm(chem_update)

        # --- stage 2: genes attend over genes (chemically conditioned) ---
        joint_input = torch.cat([x, chem_context], dim=-1)
        if self.legacy_attention_norms:
            joint_input = self.joint_norm(joint_input)
        qkv = self.kqv(joint_input)

        # NOTE: chunk order is (k, q, v) exactly as in the notebook.
        keys, queries, values = qkv.chunk(3, dim=-1)
        if self.legacy_attention_norms:
            keys = self.gene_k_norm(keys)
            queries = self.gene_q_norm(queries)
            values = self.gene_v_norm(values)

        keys = self.split_heads(keys)
        queries = self.split_heads(queries)
        values = self.split_heads(values)

        gene_context = F.scaled_dot_product_attention(
            queries,
            keys,
            values,
            dropout_p=attn_dropout,
        )

        gene_context = self.merge_heads(gene_context)
        gene_update = self.gene_ffn(gene_context) + x
        if self.chem_skip_gate_logit is not None:
            gate = self.chem_skip_gate_logit.sigmoid().to(chem_context.dtype)
            gene_update = gene_update + gate * chem_context
        gene_context = self.gene_ffn_norm(gene_update)

        return gene_context


class PerturbationModel(nn.Module):
    """Predicts per-(cell, gene) expression after a chemical perturbation.

    Pipeline
    --------
    1. Co-expression context (part 1)
       ``control_cell_line`` (B, G, C) is L2-normalised across the cell axis, so
       every gene becomes a unit vector over its expression profile. A cosine
       kernel (``scale=10``) between those unit vectors mixes the compressed
       gene embeddings (``concise_emb``) with the embeddings of co-expressed
       genes. Because the kernel is a plain Gram matrix over cells it is
       invariant to cell ordering and to the number of cells.

       Five per-gene statistics of the pre-L2 log-expression input are appended
       directly at the merger, preserving expression magnitude information.
       Add the original gene embeddings to its output before normalization.

    2. Chemical conditioning (part 2)
       The mixed gene states are refined by ``num_layers`` stacked
       :class:`ExternalPerturbAttention` blocks, which cross-attend to the
       chemical embeddings under ``chem_mask``.

    3. Coordinate-conditioned readout (part 3)
       Each cell's 2-D coordinate is concatenated with every gene state and fed
       through a small MLP, yielding (B, C, G) predicted expression.
    """

    def __init__(
        self,
        dim: int = 1024,
        conc_dim: int = 128,
        num_heads: int = 8,
        dropout: float = 0.1,
        num_layers: int = 4,
        num_of_genes: int = 1000,
        chem_dim: int = 128,
        cell_coordi_dim: int = 2,
        normalization: bool = True,
        gene_statistics: bool = True,
        gene_residual: bool = True,
        attention_normalization: bool = True,
        chem_ffn_residual: bool = True,
        ffn_multiplier: int = 2,
        joint_normalization: bool = False,
        chem_skip_gate: bool = True,
        chem_skip_init: float = 0.1,
        checkpoint_layers: bool = True,
        legacy_attention_norms: bool = False,
    ):
        super().__init__()

        self.legacy_attention_norms = legacy_attention_norms
        self.use_checkpoint_layers = checkpoint_layers
        self.num_genes = num_of_genes
        self.use_gene_statistics = gene_statistics
        self.use_gene_residual = gene_residual

        self.gene_embedding = nn.Embedding(num_of_genes, dim)

        self.chem_embedding = nn.Linear(chem_dim, dim)
        self.chem_norm = nn.LayerNorm(dim) if normalization else nn.Identity()

        self.layers = nn.ModuleList(
            [
                ExternalPerturbAttention(
                    dim=dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    normalization=normalization,
                    attention_normalization=attention_normalization,
                    legacy_attention_norms=legacy_attention_norms,
                    chem_ffn_residual=chem_ffn_residual,
                    ffn_multiplier=ffn_multiplier,
                    joint_normalization=joint_normalization,
                    chem_skip_gate=chem_skip_gate,
                    chem_skip_init=chem_skip_init,
                )
                for _ in range(num_layers)
            ]
        )

        self.conc_emb = nn.Linear(dim, conc_dim)

        self.conc_ffn = nn.Sequential(
            nn.Linear(conc_dim, dim ),
            nn.GELU(),
            nn.Linear(dim , conc_dim),
        )

        self.merger = nn.Linear(dim + conc_dim + (NUM_GENE_STATS if gene_statistics else 0), dim)
        self.merger_norm = nn.LayerNorm(dim) if normalization else nn.Identity()
        # Historical checkpoints include this affine LayerNorm. New runs use
        # the final block output directly; its gene FFN update is already normalized.
        if legacy_attention_norms:
            self.final_norm = nn.LayerNorm(dim) if normalization else nn.Identity()

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
        chem: torch.Tensor,
        cell_coordi: torch.Tensor,
        chem_mask: torch.Tensor | None = None,
        query_chunk: int | None = None,
        readout_chunk: int | None = None,
    ) -> torch.Tensor:
        """control_cell_line: (B, G, C), chem: (B, Nc, F), cell_coordi: (B, C, 2).

        ``Nc`` is "number of chemical tokens": either one pooled vector per drug
        (``Nc = number of drugs``) or one vector per atom (``Nc = number of atoms``,
        padded, with ``chem_mask`` marking the real atoms). Both are valid: the
        stage-1 cross-attention only requires a key/value set.

        ``query_chunk`` / ``readout_chunk`` are optional memory-only knobs and do
        not change the result (see ``_chunked_sdpa``); ``None`` is the original
        code path.

        Returns (B, C, G) predicted expression.
        """
        # --- unit-normalise each gene's expression profile over the cell axis ---
        # Preserve magnitude summaries before the cosine pathway removes scale.
        statistics = gene_expression_statistics(control_cell_line) if self.use_gene_statistics else None
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

        # Fixed chemical tokens are normalised once and reused by all blocks.
        chem_embeddings = self.chem_norm(self.chem_embedding(chem))

        # --- part 1: co-expression context via cosine kernel ---
        # Keys/values live in the raw (C-dim) expression space; only the values
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

        merger_inputs = [gene_embeddings, coexpression_context]
        if statistics is not None:
            merger_inputs.append(statistics)
        gene_states = self.merger(torch.cat(merger_inputs, dim=-1))

        # Identity path preserves gene identity independently of the learned fusion.
        if self.use_gene_residual:
            gene_states = gene_states + gene_embeddings
        gene_states = self.merger_norm(gene_states)

        # --- part 2: chemical-conditioned layer stack ---
        for layer in self.layers:
            if self.use_checkpoint_layers and self.training and torch.is_grad_enabled():
                gene_states = checkpoint(layer, gene_states, chem_embeddings, chem_mask, use_reentrant=False)
            else:
                gene_states = layer(gene_states, chem_embeddings, chem_mask)

        if self.legacy_attention_norms:
            gene_states = self.final_norm(gene_states)

        # --- part 3: coordinate-conditioned readout ---
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
pertubationmodel = PerturbationModel


if __name__ == "__main__":
    # Smoke test mirroring the notebook's demo cell.
    torch.manual_seed(0)

    batch, num_genes, num_cells = 2, 1000, 5
    num_chemicals, chem_features = 10, 128

    control_cell_line = torch.randn(batch, num_genes, 300)

    chem = torch.randn(batch, num_chemicals, chem_features)
    chem_conc = torch.randn(batch, 1).unsqueeze(1).expand(-1, num_chemicals, -1)
    chem = torch.cat([chem, chem_conc], dim=-1)  # (B, Nc, chem_features + 1)

    chem_mask = torch.tensor(
        [
            [1, 0, 1, 1, 0, 1, 0, 1, 1, 0],
            [1, 1, 0, 0, 1, 1, 0, 1, 1, 0],
        ],
        dtype=torch.bool,
    )[:, None, None, :]

    cell_coordi = torch.randn(batch, num_cells, 2)
    target_expression = torch.randn(batch, num_cells, num_genes)

    model = PerturbationModel(
        dim=256,
        conc_dim=32,
        num_heads=4,
        num_layers=2,
        num_of_genes=num_genes,
        chem_dim=chem_features + 1,
        cell_coordi_dim=2,
    )

    predicted_expression = model(control_cell_line, chem, cell_coordi, chem_mask)
    loss = perturbation_loss(predicted_expression, target_expression, 0.75)

    print("predicted_expression:", tuple(predicted_expression.shape))
    print("loss:", loss.item())
