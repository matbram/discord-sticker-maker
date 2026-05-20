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


def _alpha_union_bbox(frames: list[np.ndarray]) -> tuple[int, int, int, int]:
    h, w = frames[0].shape[:2]
    union = np.zeros((h, w), dtype=bool)
    for f in frames:
        union |= f[:, :, 3] > ALPHA_THRESHOLD
    if not union.any():
        return 0, 0, w, h
    ys, xs = np.where(union)
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def fit_frames(frames: list[np.ndarray], params, has_alpha: bool) -> list[np.ndarray]:
    h, w = frames[0].shape[:2]

    if params.auto_crop and has_alpha:
        x0, y0, x1, y1 = _alpha_union_bbox(frames)
    else:
        x0, y0, x1, y1 = 0, 0, w, h

    bw, bh = x1 - x0, y1 - y0
    pad = int(round(max(bw, bh) * params.padding))
    x0, y0, x1, y1 = x0 - pad, y0 - pad, x1 + pad, y1 + pad

    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    side = max(x1 - x0, y1 - y0) / max(params.zoom, 1e-3)
    cx += params.offset_x * side / 2.0
    cy += params.offset_y * side / 2.0
    half = side / 2.0
    box = (int(round(cx - half)), int(round(cy - half)), int(round(cx + half)), int(round(cy + half)))

    log.info("crop.box", box=box, source=(w, h), padded_bbox=(x0, y0, x1, y1))

    out: list[np.ndarray] = []
    for f in frames:
        im = Image.fromarray(f, "RGBA").crop(box).resize((STICKER_SIZE, STICKER_SIZE), Image.LANCZOS)
        out.append(np.asarray(im, dtype=np.uint8))
    return out
