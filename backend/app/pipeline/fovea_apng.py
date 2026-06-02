"""Bridge: encode an animated **sticker** as a perceptually-lossy *truecolor* APNG.

Discord stickers must be APNG, and APNG is lossless PNG/DEFLATE (no DCT), so a full-color
clip can't fit 512KB *losslessly*. Instead of crushing every frame into one shared palette
(the old "sepia" washout), we keep **truecolor (no palette → no washout) and every frame**,
and manufacture the needed compression *perceptually* in the native core (`fovea_native`):

  * temporal stabilization (the dominant lever) — real clips have huge frame-to-frame
    redundancy a per-frame palette throws away. Static pixels collapse to a temporal constant
    (drawn once, reused forever), grain is averaged out, and the alpha matte is frozen so its
    per-frame shimmer (the "shader" edge artifact) stops churning. This is what a video codec
    exploits; APNG's inter-frame delta + an OVER blend that carries only changed pixels turns
    that redundancy into bytes saved.
  * OKLab inter-frame delta — redraw only the pixels the eye sees change.
  * spatial denoise / chroma reduction — a last-resort tail for pathological full-frame motion.

A metric-guided search raises a single ``strength`` only as far as needed to fit the byte
budget, so clean clips stay pristine and only hard ones are (imperceptibly) stabilized. Frames
and color are never dropped. Falls back to the legacy ``encode_animated`` if the native
extension isn't built (or for content too motion-dense for lossless truecolor).
"""
from __future__ import annotations

import io
import logging
import os
import time

log = logging.getLogger("fovea.apng")

_warned_unavailable = False


def _native_available() -> bool:
    global _warned_unavailable
    try:
        import fovea_native  # noqa: F401

        ok = hasattr(fovea_native, "encode_apng")
        if not ok and not _warned_unavailable:
            _warned_unavailable = True
            log.warning("fovea.apng native encode_apng UNAVAILABLE (fovea_native=%s) — stickers "
                        "fall back to the legacy palette. Likely a stale wheel: rebuild fovea-core.",
                        getattr(fovea_native, "__version__", "?"))
        return ok
    except Exception:  # noqa: BLE001
        if not _warned_unavailable:
            _warned_unavailable = True
            log.warning("fovea.apng fovea_native import failed — stickers use the legacy palette.")
        return False


def _strength_params(s: float) -> tuple[float, float, int, float]:
    """Map one 0..1 ``strength`` knob to (temporal, denoise, chroma_step, delta_threshold).

    Temporal stabilization is the primary lever — it exploits frame-to-frame redundancy
    (static regions, grain, small motion) the way a video codec does, with no spatial
    softening. The spatial tail (denoise/chroma) only engages near the top for pathological
    full-frame motion. The search picks the *lowest* strength that fits, so fidelity is spent
    only as needed."""
    c = max(0.0, min(1.0, s))
    temporal = c                                       # primary: temporal redundancy + matte freeze
    denoise = max(0.0, (s - 0.85) / 0.15)              # spatial blur: off until 0.85 (last resort)
    chroma_step = 1 + int(round(max(0.0, s - 0.9) * 70))  # 1..~8: only the very top coarsens chroma
    delta_threshold = 0.022 + c * 0.018                # ~1..2 JND temporal reuse
    return temporal, denoise, chroma_step, delta_threshold


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
    log.info("apng.start frames=%d dim=%dx%d budget=%d fps=%.2f native=%s",
             n, w, h, budget, fps, getattr(fovea_native, "__version__", "?"))

    def tight() -> bool:
        return deadline is not None and (deadline - time.monotonic()) < 1.0

    def enc(strength: float) -> bytes:
        temporal, den, ch, dt = _strength_params(strength)
        t0 = time.monotonic()
        out = fovea_native.encode_apng(
            [f.tobytes() for f in base], w, h, dcs, delta_threshold=dt, alpha_threshold=24,
            loop_count=0, compression=4, temporal=temporal, denoise=den, chroma_step=ch)
        png = out["png"]
        log.info("apng.enc strength=%.3f temporal=%.3f delta=%.3f denoise=%.2f chroma=%d "
                 "bytes=%d (%dKB) fit=%s changed_frames=%s t=%.1fs", strength, temporal, dt, den, ch,
                 len(png), len(png) // 1024, len(png) <= budget, out.get("changed_frames"),
                 time.monotonic() - t0)
        return png

    # 1) Lossless attempt (no transforms). Truly clean/static clips fit here and ship pristine.
    data = enc(0.0)
    used = 0.0
    if len(data) > budget:
        # 2) Climb the temporal lever. `CAP` is the strongest stabilization we'll accept; if
        #    even CAP won't fit, this is pathological full-frame motion truecolor can't hold ->
        #    bail to the legacy palette path (one extra encode, no slow climb). Most stickers —
        #    cut-out or not, static or handheld — fit far below CAP.
        cap = 0.85
        top = enc(cap)
        if len(top) > budget:
            log.warning("apng.bail_legacy reason=over_budget_at_cap cap_bytes=%d budget=%d",
                        len(top), budget)
            return None
        # Bisect (0, CAP] for the lowest (highest-fidelity) strength that fits.
        lo, hi, best = 0.0, cap, (top, cap)
        for _ in range(6):
            if tight():
                log.info("apng.bisect_stop reason=deadline")
                break
            mid = (lo + hi) / 2.0
            d = enc(mid)
            if len(d) <= budget:
                best = (d, mid)
                hi = mid
            else:
                lo = mid
        data, used = best
        log.info("apng.search_done strength=%.3f bytes=%d", used, len(data))

    # 3) Score the truecolor candidate. We DON'T hard-reject on a fixed threshold anymore —
    #    keeping all frames + full color is almost always better than the legacy palette path
    #    (which drops ~half the frames and quantizes to ~32 colors). Instead we return the
    #    candidate + its distance and let the orchestrator ship whichever *actually* scores
    #    better. We only bail (None -> legacy) if the truecolor is genuinely too soft to be
    #    worth comparing (a pathological full-frame-motion clip past the sanity cap).
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
        log.warning("apng.score_failed", exc_info=True)
    # <= accept: all-frames full-color truecolor is clearly better than frame-dropped palette,
    # ship it now (skip the costly legacy compare). Between accept and sanity: borderline, let
    # the orchestrator compare to legacy. > sanity: too soft even for all-frames truecolor.
    accept = float(os.getenv("FOVEA_APNG_ACCEPT", "0.06"))
    sanity = float(os.getenv("FOVEA_APNG_SANITY", "0.18"))
    log.info("apng.score dist=%s accept=%.3f sanity=%.3f", dist, accept, sanity)
    if dist is not None and dist > sanity:
        log.warning("apng.bail_legacy reason=too_soft dist=%.4f > sanity=%.3f", dist, sanity)
        return None
    fast_accept = dist is None or dist <= accept

    report = {"mode": "temporal", "format": "apng", "strength": round(used, 3),
              "fast_accept": fast_accept,
              "perceptual_distance": (round(dist, 6) if dist is not None else None),
              "perceptually_lossless": (bool(dist <= 0.02) if dist is not None else None)}
    kb = len(data) // 1024
    log.info("fovea.apng frames=%d dim=%dx%d strength=%.2f bytes=%d dist=%s fast_accept=%s",
             n, w, h, used, len(data), dist, fast_accept)
    if notes is not None:
        if used <= 0.0:
            notes.append(f"APNG: full color, all {n} frames, losslessly ({kb} KB).")
        elif dist is not None and dist <= 0.02:
            notes.append(f"APNG: all {n} frames at full color (no sepia) — temporally stabilized "
                         f"(removed grain the eye doesn't track) to fit {kb} KB.")
        else:
            notes.append(f"APNG: all {n} frames at full color, temporally smoothed to fit {kb} KB.")
    return data, "APNG", n, fps, report


def apng_distance(fitted, delays, data) -> float | None:
    """Perceptual distance of an encoded animation (`data`, any PIL-decodable animated
    format incl. APNG) vs the source ``fitted`` frames — used to compare the truecolor APNG
    against the legacy palette result and keep whichever looks better. None if it can't score."""
    try:
        import numpy as np

        from encoder.core.frames import frames_from_list
        from encoder.metrics import default_metric

        n = len(fitted)
        dl = list(delays) if (delays and any(delays)) else [40] * n
        frs = _decode_apng_frames(data)
        if not frs:
            return None
        m = default_metric()
        return m.distance(
            frames_from_list([np.ascontiguousarray(f) for f in fitted], dl),
            frames_from_list(frs, dl[: len(frs)])).distance
    except Exception:  # noqa: BLE001
        return None
