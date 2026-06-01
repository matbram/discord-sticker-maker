"""Temporal / flicker term — the part of GIF quantization the eye actually catches.

GIF banding shimmers *between* frames in regions the source held still (dither
patterns and palette snapping change frame-to-frame). We measure how much the
candidate's inter-frame change diverges from the reference's, weighted toward
pixels the source kept static (``w_still``). Genuine motion — where the source
itself changes — is not penalized, so motion masking is respected.
"""
from __future__ import annotations

import numpy as np

_L = 255.0
DEFAULT_TAU = 8.0   # luma-delta scale (0..255) below which a pixel counts as "still"


def temporal_distance(ref: np.ndarray, cand: np.ndarray, *, tau: float = DEFAULT_TAU
                      ) -> tuple[float, list[float]]:
    """Flicker distance for two luma stacks ``(N, H, W)``.

    Returns ``(temporal, per_transition)`` where ``temporal`` is in ~[0, 1].
    Zero when the candidate's motion matches the reference's exactly.
    """
    n = ref.shape[0]
    if n < 2:
        return 0.0, []
    ref = ref.astype(np.float64, copy=False)
    cand = cand.astype(np.float64, copy=False)
    d_ref = ref[1:] - ref[:-1]
    d_cand = cand[1:] - cand[:-1]
    w_still = np.exp(-np.abs(d_ref) / tau)            # ~1 where source static
    err = w_still * (d_cand - d_ref) ** 2
    per_transition = np.sqrt(err.mean(axis=(1, 2))) / _L
    temporal = float(per_transition.mean())
    return temporal, per_transition.tolist()
