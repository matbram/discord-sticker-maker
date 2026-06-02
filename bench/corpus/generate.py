"""Generate a deterministic synthetic benchmark corpus.

Why: the "real clips only" manifest is empty on a fresh checkout, so nothing can
be measured — and the project's ship gate (spec §11) is "beat the best baseline at
equal size on the corpus." These procedural clips unblock that immediately: they
are reproducible (fixed seeds), need no licensing, and decode without ffmpeg
(written as **APNG** so the *source* keeps full color, unlike a pre-quantized GIF).

Each clip targets a different lever so regressions are localized:

  grad_pan         smooth panning gradient   -> banding / "keep the colors" torture
  screen_scroll    flat UI with a scroll      -> mostly static (delta reuse)
  subject_move     static bg + moving subject -> partial motion (the sweet spot)
  full_motion      plasma changing every px   -> hard case (little reuse)
  logo_flat        few flat colors, moving    -> sharp edges / LZW
  sticker_alpha    subject over transparency  -> the alpha (full-frame) path

Run:  python bench/corpus/generate.py [out_dir]   (default: bench/corpus/synthetic)
"""
from __future__ import annotations

import os
import sys

import numpy as np
from PIL import Image

DEFAULT_OUT = os.path.join("bench", "corpus", "synthetic")
W, H, N = 200, 150, 24


def _rgba(rgb: np.ndarray, alpha: np.ndarray | None = None) -> np.ndarray:
    h, w = rgb.shape[:2]
    a = (np.full((h, w, 1), 255, np.uint8) if alpha is None
         else alpha.reshape(h, w, 1).astype(np.uint8))
    return np.concatenate([np.clip(rgb, 0, 255).astype(np.uint8), a], axis=-1)


def grad_pan() -> list[np.ndarray]:
    xx = np.linspace(0, 1, W)[None, :]
    yy = np.linspace(0, 1, H)[:, None]
    out = []
    for k in range(N):
        ph = 2 * np.pi * k / N
        r = np.broadcast_to(128 + 120 * np.sin(2 * np.pi * xx + ph), (H, W))
        g = np.broadcast_to(128 + 120 * np.sin(2 * np.pi * yy + ph + 2.1), (H, W))
        b = np.broadcast_to(128 + 120 * np.sin(2 * np.pi * (xx + yy) + ph + 4.2), (H, W))
        out.append(_rgba(np.stack([r, g, b], -1)))
    return out


def screen_scroll() -> list[np.ndarray]:
    rng = np.random.default_rng(1)
    base = np.full((H * 2, W, 3), 245, np.float64)  # tall "page"
    base[:, :40] = (30, 32, 38)  # sidebar
    for y in range(0, H * 2, 14):  # text-like bars
        wbar = int(rng.integers(60, W - 50))
        base[y:y + 6, 50:50 + wbar] = rng.integers(40, 120, 3)
    out = []
    for k in range(N):
        off = (k * 5) % H
        view = base[off:off + H].copy()
        cx, cy = 50 + (k * 3) % (W - 60), 20 + (k * 2) % (H - 30)
        view[cy:cy + 10, cx:cx + 6] = (10, 110, 230)  # moving cursor
        out.append(_rgba(view))
    return out


def subject_move() -> list[np.ndarray]:
    yy = np.linspace(0, 1, H)[:, None]
    xx = np.linspace(0, 1, W)[None, :]
    bg = np.stack(np.broadcast_arrays(40 + 80 * yy, 60 + 120 * yy, 200 - 120 * yy), -1)
    gx, gy = np.meshgrid(np.arange(W), np.arange(H))
    out = []
    for k in range(N):
        cx = int(30 + (W - 60) * (0.5 + 0.4 * np.sin(2 * np.pi * k / N)))
        cy = int(H * 0.55)
        d = np.sqrt((gx - cx) ** 2 + (gy - cy) ** 2)
        m = np.clip(1 - d / 34, 0, 1)[..., None]
        subject = np.stack(np.broadcast_arrays(
            240 - 0.6 * d, 180 - 0.4 * d, 90 + 0.3 * d), -1)
        out.append(_rgba(bg * (1 - m) + subject * m))
    return out


def full_motion() -> list[np.ndarray]:
    xx = np.linspace(0, 6, W)[None, :]
    yy = np.linspace(0, 6, H)[:, None]
    out = []
    for k in range(N):
        t = k * 0.5
        v = (np.sin(xx + t) + np.sin(yy - t) + np.sin(xx + yy + t)
             + np.sin(np.sqrt(xx ** 2 + yy ** 2) * 2 + t))
        r = 128 + 90 * np.sin(v + 0)
        g = 128 + 90 * np.sin(v + 2)
        b = 128 + 90 * np.sin(v + 4)
        out.append(_rgba(np.stack(np.broadcast_arrays(r, g, b), -1)))
    return out


def logo_flat() -> list[np.ndarray]:
    palette = [(229, 57, 53), (30, 136, 229), (253, 216, 53), (67, 160, 71), (142, 36, 170)]
    gx, gy = np.meshgrid(np.arange(W), np.arange(H))
    out = []
    for k in range(N):
        img = np.full((H, W, 3), 250, np.float64)
        ang = 2 * np.pi * k / N
        for i, col in enumerate(palette):
            cx = int(W / 2 + 45 * np.cos(ang + i * 1.25))
            cy = int(H / 2 + 35 * np.sin(ang + i * 1.25))
            img[((gx - cx) ** 2 + (gy - cy) ** 2) < 22 ** 2] = col
        out.append(_rgba(img))
    return out


def sticker_alpha() -> list[np.ndarray]:
    gx, gy = np.meshgrid(np.arange(W), np.arange(H))
    out = []
    for k in range(N):
        cx = int(30 + (W - 60) * (0.5 + 0.4 * np.cos(2 * np.pi * k / N)))
        cy = int(H / 2 + 20 * np.sin(4 * np.pi * k / N))
        d = np.sqrt((gx - cx) ** 2 + (gy - cy) ** 2)
        inside = d < 36
        rgb = np.stack(np.broadcast_arrays(
            250 - 1.2 * d, 120 + 0.8 * d, 60 + 1.4 * d), -1)
        alpha = np.where(inside, 255, 0).astype(np.uint8)
        out.append(_rgba(rgb, alpha))
    return out


CLIPS = {
    "grad_pan": ("motion_graphics", grad_pan),
    "screen_scroll": ("screen_recording", screen_scroll),
    "subject_move": ("video_clip", subject_move),
    "full_motion": ("video_clip", full_motion),
    "logo_flat": ("motion_graphics", logo_flat),
    "sticker_alpha": ("video_clip", sticker_alpha),
}


def write_apng(frames: list[np.ndarray], path: str, fps: int = 20) -> None:
    # Disposal matters for a faithful round-trip through PIL's APNG frame optimizer:
    #   opaque clips -> dispose=NONE(0): unchanged regions keep the prior opaque pixel,
    #                   so they don't read back as spurious transparency.
    #   alpha clips  -> dispose=BACKGROUND(1): clear to transparent each frame so a
    #                   moving subject leaves no trail and the matte is preserved.
    has_alpha = any(bool((f[..., 3] < 255).any()) for f in frames)
    ims = [Image.fromarray(f, "RGBA") for f in frames]
    ims[0].save(
        path, save_all=True, append_images=ims[1:],
        duration=int(round(1000 / fps)), loop=0,
        disposal=(1 if has_alpha else 0), blend=0,
    )


def generate(out_dir: str = DEFAULT_OUT) -> list[tuple[str, str, str]]:
    os.makedirs(out_dir, exist_ok=True)
    made = []
    for cid, (cat, fn) in CLIPS.items():
        path = os.path.join(out_dir, f"{cid}.png")
        write_apng(fn(), path)
        made.append((cid, path, cat))
        print(f"  {cid:<14} {cat:<16} {os.path.getsize(path)//1024:>4} KB  {path}")
    return made


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_OUT
    print(f"generating synthetic corpus -> {out}")
    generate(out)
