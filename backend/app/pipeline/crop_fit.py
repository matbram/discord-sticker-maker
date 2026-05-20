"""Crop to the subject and fit to exactly 320x320 with transparent padding.

When the background was removed we crop to the union of the subject's alpha bbox
across all frames (so an animated subject doesn't jump), then square + resize.
Out-of-bounds crops are transparent-padded automatically by Pillow on RGBA.
"""
from __future__ import annotations

import numpy as np
from PIL import Image

from ..models import STICKER_SIZE
from ..observability import get_logger

log = get_logger("crop_fit")

ALPHA_THRESHOLD = 8


def downscale_max_side(frames: list[np.ndarray], max_side: int) -> list[np.ndarray]:
    """Shrink frames so their longest side is <= max_side (no-op if already smaller).

    Applied before background removal to bound peak memory and speed up matting.
    """
    h, w = frames[0].shape[:2]
    if max(h, w) <= max_side:
        return frames
    scale = max_side / float(max(h, w))
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    return [
        np.asarray(Image.fromarray(f, "RGBA").resize((nw, nh), Image.LANCZOS), dtype=np.uint8)
        for f in frames
    ]


def _alpha_union_bbox(frames: list[np.ndarray]) -> tuple[int, int, int, int]:
    h, w = frames[0].shape[:2]
    union = np.zeros((h, w), dtype=bool)
    for f in frames:
        union |= f[:, :, 3] > ALPHA_THRESHOLD
    if not union.any():
        return 0, 0, w, h
    ys, xs = np.where(union)
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def fit_square(frames: list[np.ndarray], params, has_alpha: bool, size: int = STICKER_SIZE) -> list[np.ndarray]:
    """Crop to a square (subject bbox / fit / fill) and resize to size x size."""
    h, w = frames[0].shape[:2]

    if params.auto_crop and has_alpha:
        x0, y0, x1, y1 = _alpha_union_bbox(frames)
    else:
        x0, y0, x1, y1 = 0, 0, w, h

    fit_mode = getattr(params, "fit_mode", "fit")
    fit_mode = fit_mode.value if hasattr(fit_mode, "value") else fit_mode
    if fit_mode == "fill":
        base = min(x1 - x0, y1 - y0)
    else:
        pad = int(round(max(x1 - x0, y1 - y0) * params.padding))
        x0, y0, x1, y1 = x0 - pad, y0 - pad, x1 + pad, y1 + pad
        base = max(x1 - x0, y1 - y0)

    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    side = base / max(params.zoom, 1e-3)
    cx += params.offset_x * side / 2.0
    cy += params.offset_y * side / 2.0
    half = side / 2.0
    box = (int(round(cx - half)), int(round(cy - half)), int(round(cx + half)), int(round(cy + half)))
    log.info("crop.box", box=box, source=(w, h), size=size)

    out: list[np.ndarray] = []
    for f in frames:
        im = Image.fromarray(f, "RGBA").crop(box).resize((size, size), Image.LANCZOS)
        out.append(np.asarray(im, dtype=np.uint8))
    return out


def fit_aspect(frames: list[np.ndarray], params, max_dim: int) -> list[np.ndarray]:
    """Keep aspect ratio; scale so the longest side == max_dim. Zoom does a centered
    crop (keeping aspect); offset pans. Used for GIFs (not forced square)."""
    h, w = frames[0].shape[:2]
    zoom = max(params.zoom, 1e-3)
    cw, ch = w / zoom, h / zoom
    cx = w / 2.0 + params.offset_x * cw / 2.0
    cy = h / 2.0 + params.offset_y * ch / 2.0
    box = (int(round(cx - cw / 2)), int(round(cy - ch / 2)), int(round(cx + cw / 2)), int(round(cy + ch / 2)))
    scale = max_dim / float(max(cw, ch))
    tw, th = max(1, int(round(cw * scale))), max(1, int(round(ch * scale)))
    # even dimensions keep ffmpeg/gif encoders happy
    tw -= tw % 2; th -= th % 2
    tw, th = max(2, tw), max(2, th)
    log.info("crop.aspect", box=box, source=(w, h), target=(tw, th))
    out: list[np.ndarray] = []
    for f in frames:
        im = Image.fromarray(f, "RGBA").crop(box).resize((tw, th), Image.LANCZOS)
        out.append(np.asarray(im, dtype=np.uint8))
    return out


# Back-compat alias (single-sticker callers).
def fit_frames(frames: list[np.ndarray], params, has_alpha: bool) -> list[np.ndarray]:
    return fit_square(frames, params, has_alpha, STICKER_SIZE)
