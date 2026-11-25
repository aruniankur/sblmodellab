"""Cell-order-invariant summaries of the model's existing log1p(CP10K) input."""
from __future__ import annotations

import torch

GENE_STAT_NAMES = (
    "mean_log_expression",
    "std_log_expression",
    "detected_fraction",
    "positive_median_log_expression",
    "positive_p90_log_expression",
)
NUM_GENE_STATS = len(GENE_STAT_NAMES)


def gene_expression_statistics(control: torch.Tensor) -> torch.Tensor:
    """Return (B, G, 5) statistics across the control-cell axis, in FP32.

    Input is already log1p(CP10K); do not exponentiate, re-normalize or log it.
    Standard deviation uses correction=0. Positive quantiles use linear
    interpolation among strictly positive values only. All-zero genes have five
    zero features; a single positive observation supplies both positive quantiles.
    No learned projection is used. Only one sort is needed for both quantiles.
    """
    if control.ndim != 3 or control.shape[-1] == 0:
        raise ValueError("Expected nonempty control expression with shape (B, G, C)")
    with torch.autocast(device_type=control.device.type, enabled=False):
        x = control.float()
        std, mean = torch.std_mean(x, dim=-1, correction=0, keepdim=True)
        positive = x > 0
        count = positive.sum(dim=-1, keepdim=True)
        fraction = count.to(x.dtype) / x.shape[-1]
        ordered = x.masked_fill(~positive, float("inf")).sort(dim=-1).values
        # Avoid inf arithmetic for empty rows, including during backward.
        ordered = torch.where(count > 0, ordered, torch.zeros_like(ordered))

        def quantile(q):
            position = (count - 1).clamp_min(0).to(x.dtype) * q
            lo = position.floor().long()
            hi = position.ceil().long()
            low = ordered.gather(-1, lo)
            high = ordered.gather(-1, hi)
            return low + (high - low) * (position - lo.to(x.dtype))

        return torch.cat([mean, std, fraction, quantile(.5), quantile(.9)], dim=-1)
