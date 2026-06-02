"""Bridge: encode an animated **sticker** as a perceptually-lossy *truecolor* APNG.

Discord stickers must be APNG, and APNG is lossless PNG/DEFLATE (no DCT), so a full-color
clip can't fit 512KB *losslessly*. Instead of crushing every frame into one shared palette
(the old "sepia" washout), we keep **truecolor (no palette → no washout) and every frame**,
and manufacture the needed compression *perceptually* in the native core (`fovea_native`):

  * OKLab inter-frame delta — redraw only the pixels the eye sees change.
  * edge-aware denoise — remove incompressible grain the eye doesn't track (the big lever).
  * chroma reduction — coarsen chroma, which the eye tolerates far more than luma.

A metric-guided search raises a single ``strength`` only as far as needed to fit the byte
budget, so clean clips stay pristine and only hard ones are (imperceptibly) smoothed. Frames
and color are never dropped. Falls back to the legacy ``encode_animated`` if the native
extension isn't built.
"""
from __future__ import annotations

import io
import logging
import os
import time

log = logging.getLogger("fovea.apng")


def _native_available() -> bool:
    try:
        import fovea_native  # noqa: F401

        return hasattr(fovea_native, "encode_apng")
    except Exception:  # noqa: BLE001
        return False


def _strength_params(s: float) -> tuple[float, int, float]:
    """Map one 0..1 ``strength`` knob to (denoise, chroma_step, delta_threshold).

    All three reduce entropy the eye barely notices; the search picks the *lowest* strength
    that fits, so fidelity is spent only as needed."""
    c = max(0.0, min(1.0, s))
    denoise = max(0.0, s)                  # uncapped: s>1 = heavier blur (guaranteed-fit resort)
    chroma_step = 1 + int(round(c * 31))   # 1 (off) .. 32 (coarse chroma, ~grayscale at max)
    delta_threshold = 0.02 + c * 0.06      # 1..4 JND temporal reuse
    return denoise, chroma_step, delta_threshold


def _decode_apng_frames(data: bytes):
    """Composited RGBA frames from APNG bytes (PIL composites sub-frames on seek)."""
    import numpy as np
    from PIL import Image

    im = Image.open(io.BytesIO(data))
    frames = []
    try:
        while True:
            frames.append(np.asarray(im.convert("RGBA")))
            im.seek(im.tell() + 1)
    except EOFError:
        pass
    return frames


def apng_encode(fitted, delays, *, budget, deadline=None, notes=None):
    """Encode RGBA ``fitted`` frames to a truecolor APNG ≤ ``budget`` -> (bytes, "APNG",
    n_frames, fps, report). Keeps every frame + full color; raises perceptual ``strength``
    only as far as needed to fit. Returns ``None`` if the native encoder is unavailable
    (caller falls back to the legacy APNG path)."""
    if not _native_available():
        return None

    import numpy as np

    import fovea_native
    from encoder.core.timing import effective_fps

    n = len(fitted)
    h, w = fitted[0].shape[:2]
    dl = list(delays) if (delays and any(delays)) else [40] * n
    base = [np.ascontiguousarray(fr) for fr in fitted]
    fps = effective_fps(dl) or 10.0
    dcs = [max(1, int(round(d / 10.0))) for d in dl]  # ms -> centiseconds

    def tight() -> bool:
        return deadline is not None and (deadline - time.monotonic()) < 1.0

    def enc(strength: float) -> bytes:
        den, ch, dt = _strength_params(strength)
        return fovea_native.encode_apng(
            [f.tobytes() for f in base], w, h, dcs, delta_threshold=dt, alpha_threshold=24,
            loop_count=0, compression=4, denoise=den, chroma_step=ch)["png"]

    # 1) Lossless attempt (no transforms). Cut-out / clean / static-bg stickers fit here.
    data = enc(0.0)
    used = 0.0
    if len(data) > budget:
        # 2) `CAP` is the most denoise we'll accept before it stops being grain-removal and
        #    starts blurring real detail. If even CAP won't fit, this is dense full-frame
        #    content truecolor can't hold cleanly -> bail fast to the legacy palette path
        #    (one extra encode, no slow climb into mush).
        cap = 0.85
        top = enc(cap)
        if len(top) > budget:
            return None
        # Bisect (0, CAP] for the lowest (highest-fidelity) strength that fits.
        lo, hi, best = 0.0, cap, (top, cap)
        for _ in range(6):
            if tight():
                break
            mid = (lo + hi) / 2.0
            d = enc(mid)
            if len(d) <= budget:
                best = (d, mid)
                hi = mid
            else:
                lo = mid
        data, used = best

    # 3) Perceptual gate: keep the truecolor APNG ONLY if it's genuinely clean. For dense
    #    full-frame motion, fitting truecolor would require visibly blurring real detail
    #    (mush) — the legacy shared-palette path stays sharp and uses the budget, so hand
    #    back to it instead of shipping a blurred result. (Cut-out stickers pass easily.)
    dist = None
    try:
        from encoder.core.frames import frames_from_list
        from encoder.metrics import default_metric

        frs = _decode_apng_frames(data)
        m = default_metric()
        dist = m.distance(
            frames_from_list([np.ascontiguousarray(f) for f in base], dl),
            frames_from_list(frs, dl[: len(frs)])).distance
    except Exception:  # noqa: BLE001 - report is best-effort
        pass
    accept = float(os.getenv("FOVEA_APNG_ACCEPT", "0.045"))
    if dist is not None and dist > accept:
        return None  # would be visibly soft — the palette path looks better here

    report = {"mode": "cap", "format": "apng", "strength": round(used, 3),
              "perceptual_distance": (round(dist, 6) if dist is not None else None),
              "perceptually_lossless": (bool(dist <= 0.02) if dist is not None else None)}
    kb = len(data) // 1024
    log.info("fovea.apng frames=%d dim=%dx%d strength=%.2f bytes=%d dist=%s",
             n, w, h, used, len(data), dist)
    if notes is not None:
        if used <= 0.0:
            notes.append(f"APNG: full color, all {n} frames, losslessly ({kb} KB).")
        elif dist is not None and dist <= 0.02:
            notes.append(f"APNG: all {n} frames at full color (no sepia) — removed imperceptible "
                         f"grain to fit {kb} KB.")
        else:
            notes.append(f"APNG: all {n} frames at full color, lightly smoothed to fit {kb} KB.")
    return data, "APNG", n, fps, report
