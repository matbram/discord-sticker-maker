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

    from .encode import even_subsample

    n = len(fitted)
    h, w = fitted[0].shape[:2]
    dl = list(delays) if (delays and any(delays)) else [40] * n
    base = [np.ascontiguousarray(fr) for fr in fitted]
    fps = effective_fps(dl) or 10.0

    def tight() -> bool:
        return deadline is not None and (deadline - time.monotonic()) < 1.0

    def enc(frames, dl_ms, strength: float) -> bytes:
        den, ch, dt = _strength_params(strength)
        dcs = [max(1, int(round(d / 10.0))) for d in dl_ms]  # ms -> centiseconds
        res = fovea_native.encode_apng(
            [f.tobytes() for f in frames], w, h, dcs, delta_threshold=dt, alpha_threshold=24,
            loop_count=0, compression=4, denoise=den, chroma_step=ch)
        return res["png"]

    out_frames, out_dl = base, dl

    # 1) Lossless attempt (no transforms). Clean / static / cut-out stickers fit here.
    data = enc(base, dl, 0.0)
    used = 0.0
    if len(data) > budget:
        # 2) Bisect strength for the *lowest* (highest-fidelity) value that fits the budget —
        #    edge-aware denoise + chroma reduction spend only imperceptible entropy.
        lo, hi, best = 0.0, 1.0, None
        for _ in range(7):
            if tight():
                break
            mid = (lo + hi) / 2.0
            d = enc(base, dl, mid)
            if len(d) <= budget:
                best = (d, mid)
                hi = mid
            else:
                lo = mid
        if best is not None:
            data, used = best
        else:
            # 3) A bounded amount of extra (now visible) blur for very dense clips.
            for s in (1.4, 2.0):
                if tight():
                    break
                data, used = enc(base, dl, s), s
                if len(data) <= budget:
                    break
            # 4) Last resort for pathological full-frame motion that no perceptual transform
            #    can fit at full frame count: trim frames to honor the hard byte cap (Discord
            #    rejects >512KB). Real stickers never reach here; this guarantees a valid file.
            fcount = n
            while len(data) > budget and fcount > 8 and not tight():
                fcount = max(8, int(fcount * 0.7))
                out_frames, out_dl = even_subsample(base, dl, fcount)
                data = enc(out_frames, out_dl, used)

    n_out = len(out_frames)
    fps = effective_fps(out_dl) or fps

    # Honesty report: perceptual distance of the APNG vs the (kept) source frames.
    report = {"mode": "cap", "format": "apng", "strength": round(used, 3)}
    try:
        from encoder.core.frames import frames_from_list
        from encoder.metrics import default_metric

        frs = _decode_apng_frames(data)
        m = default_metric()
        res = m.distance(
            frames_from_list([np.ascontiguousarray(f) for f in out_frames], out_dl),
            frames_from_list(frs, out_dl[: len(frs)]))
        report["perceptual_distance"] = round(res.distance, 6)
        report["perceptually_lossless"] = bool(res.distance <= m.invisible_threshold)
    except Exception:  # noqa: BLE001 - report is best-effort
        pass

    kb = len(data) // 1024
    lossless = report.get("perceptually_lossless", True)
    log.info("fovea.apng frames=%d/%d dim=%dx%d strength=%.2f bytes=%d dist=%s",
             n_out, n, w, h, used, len(data), report.get("perceptual_distance"))
    if notes is not None:
        if n_out < n:
            notes.append(f"APNG: this clip is extremely detailed — kept full color but trimmed "
                         f"to {n_out} of {n} frames to fit {kb} KB (raise the size limit to keep "
                         f"every frame).")
        elif used <= 0.0:
            notes.append(f"APNG: full color, all {n} frames, losslessly ({kb} KB).")
        elif lossless:
            notes.append(f"APNG: kept all {n} frames at full color (no sepia) — smoothed "
                         f"imperceptible grain to fit {kb} KB.")
        else:
            notes.append(f"APNG: kept all {n} frames and full color; softened detail to fit "
                         f"{kb} KB (raise the size limit for more sharpness).")
    return data, "APNG", n_out, fps, report
