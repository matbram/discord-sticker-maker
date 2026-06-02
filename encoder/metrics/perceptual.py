"""The default Fovea judge: MS-SSIM (spatial) blended with a temporal flicker term.

This is a *reference* metric standing in until the learned, motion-aware judge
(milestone M2) is trained. It is deliberately simple, dependency-light, and
pluggable so M2 can replace it behind the same ``Metric`` interface. The
``invisible_threshold`` is a calibration target (see ``docs/metrics.md``), not a
physical constant — the M0 benchmark table is the instrument used to tune it.
"""
from __future__ import annotations

import os

import numpy as np
from PIL import Image

from ..core.frames import Frames
from .banding import banding_per_frame
from .base import DistanceResult, Metric
from .msssim import msssim_per_frame
from .temporal import temporal_distance

_LUMA = np.array([0.299, 0.587, 0.114], dtype=np.float64)

# Perceptual scores are low-frequency: a full 512px frame is needless work (MS-SSIM
# + OKLab banding cost ~30s/score at that size). Downscale both sides to a common
# small size first — banding and structure survive it — for an ~10x speedup.
METRIC_MAX_DIM = int(os.getenv("FOVEA_METRIC_MAXDIM", "192"))


def _metric_frames(
    ref: list[np.ndarray], cand: list[np.ndarray], max_dim: int
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Downscale reference and candidate frames to a common small size (from the
    reference's aspect), so scoring is cheap and the two are directly comparable."""
    h, w = ref[0].shape[:2]
    m = max(h, w)
    s = min(1.0, max_dim / m) if m > 0 else 1.0
    nh, nw = max(1, round(h * s)), max(1, round(w * s))

    def rs(frames: list[np.ndarray]) -> list[np.ndarray]:
        out = []
        for f in frames:
            im = Image.fromarray(np.asarray(f, dtype=np.uint8)).convert("RGBA")
            if im.size != (nw, nh):
                im = im.resize((nw, nh), Image.BILINEAR)
            out.append(np.asarray(im, dtype=np.uint8))
        return out

    return rs(ref), rs(cand)


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
        ref_f, cand_f = _metric_frames(reference.frames, candidate.frames, METRIC_MAX_DIM)
        ref_l = composite_luma_stack(ref_f)
        cand_l = composite_luma_stack(cand_f)

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


class ColorAwareMetric(Metric):
    """MS-SSIM + temporal **+ a banding-aware OKLab color term** — the default judge.

    Luma MS-SSIM alone is blind to color banding and actually prefers it to dither
    (``docs/m2-judge.md``), which is what forced the bridge's color-floor bandaid.
    Adding the low-pass-OKLab ΔE term (``banding.py``) makes the distance fall when
    the result is *more* faithful in color — so the search, which already minimises
    distance under the byte budget, can finally be trusted to pick rich color over
    washout. ``gamma`` weights the color term; ``per_frame`` (and ``worst_frame``)
    fold it in so the honesty report still points at where loss is worst.
    """

    name = "msssim+temporal+color"

    def __init__(
        self, *, beta: float = 0.5, gamma: float = 3.0, invisible_threshold: float = 0.02
    ) -> None:
        self.beta = beta
        self.gamma = gamma
        self.invisible_threshold = invisible_threshold

    def distance(self, reference: Frames, candidate: Frames) -> DistanceResult:
        ref_f, cand_f = _metric_frames(reference.frames, candidate.frames, METRIC_MAX_DIM)
        ref_l = composite_luma_stack(ref_f)
        cand_l = composite_luma_stack(cand_f)
        n = min(ref_l.shape[0], cand_l.shape[0])
        ref_l, cand_l = ref_l[:n], cand_l[:n]
        cand_l = _resize_luma_stack(cand_l, ref_l.shape[1:])

        msssim_f = msssim_per_frame(ref_l, cand_l)
        struct = 1.0 - msssim_f                                  # per-frame structural loss
        band = banding_per_frame(ref_f, cand_f)                  # per-frame OKLab ΔE (small)
        band = band[:n]
        temporal, _ = temporal_distance(ref_l, cand_l)

        per_frame = struct + self.gamma * band
        spatial = float(per_frame.mean())
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
                "band_mean": float(band.mean()) if band.size else 0.0,
                "band_max": float(band.max()) if band.size else 0.0,
            },
        )
