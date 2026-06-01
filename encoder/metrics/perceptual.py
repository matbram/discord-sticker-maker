"""The default Fovea judge: MS-SSIM (spatial) blended with a temporal flicker term.

This is a *reference* metric standing in until the learned, motion-aware judge
(milestone M2) is trained. It is deliberately simple, dependency-light, and
pluggable so M2 can replace it behind the same ``Metric`` interface. The
``invisible_threshold`` is a calibration target (see ``docs/metrics.md``), not a
physical constant — the M0 benchmark table is the instrument used to tune it.
"""
from __future__ import annotations

import numpy as np
from PIL import Image

from ..core.frames import Frames
from .base import DistanceResult, Metric
from .msssim import msssim_per_frame
from .temporal import temporal_distance

_LUMA = np.array([0.299, 0.587, 0.114], dtype=np.float64)


def composite_luma_stack(frames: list[np.ndarray]) -> np.ndarray:
    """RGBA frames -> luma stack ``(N, H, W)``, alpha composited over black.

    Compositing over a fixed background makes transparent regions identical
    between reference and candidate, so 1-bit GIF transparency does not pollute
    the score.
    """
    out = np.empty((len(frames),) + frames[0].shape[:2], dtype=np.float64)
    for i, fr in enumerate(frames):
        arr = fr.astype(np.float64)
        rgb = arr[..., :3]
        alpha = arr[..., 3:4] / 255.0 if arr.shape[-1] == 4 else 1.0
        comp = rgb * alpha
        out[i] = comp @ _LUMA
    return out


def _resize_luma_stack(stack: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    h, w = stack.shape[1:]
    th, tw = target_hw
    if (h, w) == (th, tw):
        return stack
    out = np.empty((stack.shape[0], th, tw), dtype=np.float64)
    for i, frame in enumerate(stack):
        im = Image.fromarray(frame.astype(np.float32), mode="F").resize((tw, th), Image.BILINEAR)
        out[i] = np.asarray(im, dtype=np.float64)
    return out


class PerceptualMetric(Metric):
    name = "msssim+temporal"

    def __init__(self, *, beta: float = 0.5, invisible_threshold: float = 0.005) -> None:
        self.beta = beta
        self.invisible_threshold = invisible_threshold

    def distance(self, reference: Frames, candidate: Frames) -> DistanceResult:
        ref_l = composite_luma_stack(reference.frames)
        cand_l = composite_luma_stack(candidate.frames)

        # All frames are kept, so counts should match; guard defensively.
        n = min(ref_l.shape[0], cand_l.shape[0])
        ref_l, cand_l = ref_l[:n], cand_l[:n]
        # The resolution lever can shrink the candidate; compare at reference size.
        cand_l = _resize_luma_stack(cand_l, ref_l.shape[1:])

        msssim_f = msssim_per_frame(ref_l, cand_l)
        per_frame = (1.0 - msssim_f)
        spatial = float(per_frame.mean())
        temporal, _ = temporal_distance(ref_l, cand_l)
        distance = spatial + self.beta * temporal
        worst = int(np.argmax(per_frame)) if per_frame.size else 0
        return DistanceResult(
            distance=distance,
            per_frame=per_frame.tolist(),
            spatial=spatial,
            temporal=temporal,
            worst_frame=worst,
            extra={
                "msssim_mean": float(msssim_f.mean()),
                "min_msssim": float(msssim_f.min()) if msssim_f.size else 1.0,
            },
        )
