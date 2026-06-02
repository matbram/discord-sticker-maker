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


def _interframe_sad(frames):
    """Mean absolute luma difference between consecutive frames — a cheap proxy for temporal
    redundancy. Alpha-weighted so the transparent background (which the encoder skips anyway,
    and whose stray RGB shifts under alignment) doesn't drown out the subject's signal."""
    import numpy as np

    lums = [(f[..., :3].astype(np.float32).mean(-1)) * (f[..., 3].astype(np.float32) / 255.0)
            for f in frames]
    if len(lums) < 2:
        return 0.0
    return float(np.mean([np.abs(lums[i] - lums[i - 1]).mean() for i in range(1, len(lums))]))


def _global_motion_offsets(base, has_alpha):
    """Per-frame integer (dx,dy) that registers each frame to frame 0 (cancels camera shake).
    Coarse-to-fine SAD on (alpha-masked) half-res luma; the dominant rigid content drives the
    match, so local subject motion doesn't throw it off."""
    import numpy as np

    n = len(base)

    def feat(f):
        lum = f[..., :3].astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32)
        if has_alpha:
            lum = lum * (f[..., 3].astype(np.float32) / 255.0)
        return lum[::2, ::2]  # half-res for speed

    ref = feat(base[0])
    h2, w2 = ref.shape
    offs = [(0, 0)]
    for i in range(1, n):
        cur = feat(base[i])
        best = (0, 0)
        for step, seed in ((3, (0, 0)), (1, None)):
            cx, cy = best if seed is None else seed
            bs, bb = 1e18, best
            for dy in range(cy - step * 3, cy + step * 3 + 1, step):
                for dx in range(cx - step * 3, cx + step * 3 + 1, step):
                    a = ref[max(0, dy):h2 + min(0, dy), max(0, dx):w2 + min(0, dx)]
                    b = cur[max(0, -dy):h2 + min(0, -dy), max(0, -dx):w2 + min(0, -dx)]
                    if a.size:
                        s = float(np.abs(a - b).mean())
                        if s < bs:
                            bs, bb = s, (dx, dy)
            best = bb
        offs.append((best[0] * 2, best[1] * 2))  # back to full-res
    return offs


def _stabilize(base, has_alpha):
    """Cancel global camera motion: align every frame to frame 0, fill the exposed border
    (transparent for cut-outs, edge-clamp otherwise), and — for cut-outs — freeze the
    silhouette to its temporal-median binary alpha so the (±1-2px) alignment residual can't
    churn the hard edge. This is what lets a handheld clip's static content actually flatten
    temporally. Returns (frames, max_shift); max_shift 0 means no meaningful camera motion."""
    import numpy as np

    offs = _global_motion_offsets(base, has_alpha)
    max_shift = max((abs(ox) + abs(oy)) for ox, oy in offs)
    if max_shift < 2:
        return [np.ascontiguousarray(f) for f in base], 0
    out = []
    for f, (ox, oy) in zip(base, offs):
        g = np.roll(np.roll(f, oy, axis=0), ox, axis=1)
        if has_alpha:  # exposed border -> transparent (free for a cut-out)
            if oy > 0:
                g[:oy] = 0
            elif oy < 0:
                g[oy:] = 0
            if ox > 0:
                g[:, :ox] = 0
            elif ox < 0:
                g[:, ox:] = 0
        else:  # opaque -> replicate the edge so the border doesn't flash
            if oy > 0:
                g[:oy] = g[oy:oy + 1]
            elif oy < 0:
                g[oy:] = g[oy - 1:oy]
            if ox > 0:
                g[:, :ox] = g[:, ox:ox + 1]
            elif ox < 0:
                g[:, ox:] = g[:, ox - 1:ox]
        out.append(g)
    if has_alpha:
        amed = np.median(np.stack([f[..., 3] for f in out], 0), 0)
        afz = np.where(amed >= 128, 255, 0).astype(np.uint8)
        out = [np.ascontiguousarray(np.dstack([f[..., :3], afz]).astype(np.uint8)) for f in out]
    else:
        out = [np.ascontiguousarray(f) for f in out]
    return out, max_shift


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

    # `frames_src` is what we encode; if camera motion is detected it becomes the stabilized
    # frames. `ref` is the metric reference — also the stabilized frames, so we score the
    # encode honestly (stabilization itself is a perceptual win, not a defect to penalize).
    alpha_present = any(int(f[..., 3].min()) < 250 for f in base)
    frames_src = base
    ref = base
    mc_applied = False
    mc_shift = 0

    def enc(strength: float) -> bytes:
        temporal, den, ch, dt = _strength_params(strength)
        t0 = time.monotonic()
        out = fovea_native.encode_apng(
            [f.tobytes() for f in frames_src], w, h, dcs, delta_threshold=dt, alpha_threshold=24,
            loop_count=0, compression=4, temporal=temporal, denoise=den, chroma_step=ch)
        png = out["png"]
        log.info("apng.enc strength=%.3f temporal=%.3f delta=%.3f denoise=%.2f chroma=%d "
                 "bytes=%d (%dKB) fit=%s changed_frames=%s mc=%s t=%.1fs", strength, temporal, dt, den, ch,
                 len(png), len(png) // 1024, len(png) <= budget, out.get("changed_frames"), mc_applied,
                 time.monotonic() - t0)
        return png

    # 1) Lossless attempt (no transforms). Truly clean/static clips fit here and ship pristine.
    data = enc(0.0)
    used = 0.0
    if len(data) > budget:
        # 2) Camera shake makes EVERY frame redraw — temporal can't flatten a translating scene,
        #    so the delta re-sends the whole subject each frame. Cancel the global motion
        #    (stabilize); the now-static content collapses and truecolor fits at a low strength.
        stab, mc_shift = _stabilize(base, alpha_present)
        # Keep MC only if alignment genuinely made consecutive frames more similar (real camera
        # motion with dominant rigid content). The inter-frame SAD is a cheap, encode-free signal;
        # if it barely drops, the estimate was unreliable — revert so we never ship jittery output.
        sad_ratio = (_interframe_sad(stab) / max(1e-6, _interframe_sad(base))) if mc_shift >= 2 else 1.0
        if mc_shift >= 2 and sad_ratio < 0.7:
            frames_src = stab
            ref = stab
            mc_applied = True
            log.info("apng.mc applied max_shift=%d alpha=%s sad_ratio=%.2f", mc_shift, alpha_present, sad_ratio)
            data = enc(0.0)
            used = 0.0
        elif mc_shift >= 2:
            log.info("apng.mc skipped reason=no_gain max_shift=%d sad_ratio=%.2f", mc_shift, sad_ratio)
    if len(data) > budget:
        # 3) Climb the temporal lever. `CAP` is the strongest stabilization we'll accept; if even
        #    CAP won't fit, the clip is too motion-dense for lossless truecolor -> legacy palette.
        cap = 0.85
        top = enc(cap)
        if len(top) > budget:
            log.warning("apng.bail_legacy reason=over_budget_at_cap mc=%s cap_bytes=%d budget=%d",
                        mc_applied, len(top), budget)
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
        log.info("apng.search_done strength=%.3f bytes=%d mc=%s", used, len(data), mc_applied)

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
            frames_from_list([np.ascontiguousarray(f) for f in ref], dl),
            frames_from_list(frs, dl[: len(frs)])).distance
    except Exception:  # noqa: BLE001 - report is best-effort
        log.warning("apng.score_failed", exc_info=True)
    # <= accept: all-frames full-color truecolor is clearly better than frame-dropped palette,
    # ship it now (skip the costly legacy compare). Between accept and sanity: borderline, let
    # the orchestrator compare to legacy. > sanity: too soft even for all-frames truecolor.
    accept = float(os.getenv("FOVEA_APNG_ACCEPT", "0.06"))
    sanity = float(os.getenv("FOVEA_APNG_SANITY", "0.18"))
    log.info("apng.score dist=%s accept=%.3f sanity=%.3f mc=%s", dist, accept, sanity, mc_applied)
    if dist is not None and dist > sanity:
        log.warning("apng.bail_legacy reason=too_soft dist=%.4f > sanity=%.3f", dist, sanity)
        return None
    # When MC stabilized the clip, the original-frame legacy comparison is unfair (it would
    # penalize the stabilization shift), so ship the stabilized truecolor directly within sanity.
    fast_accept = dist is None or dist <= accept or mc_applied

    report = {"mode": ("motion+temporal" if mc_applied else "temporal"), "format": "apng",
              "strength": round(used, 3), "fast_accept": fast_accept, "motion_comp": mc_applied,
              "perceptual_distance": (round(dist, 6) if dist is not None else None),
              "perceptually_lossless": (bool(dist <= 0.02) if dist is not None else None)}
    kb = len(data) // 1024
    log.info("fovea.apng frames=%d dim=%dx%d strength=%.2f bytes=%d dist=%s fast_accept=%s mc=%s",
             n, w, h, used, len(data), dist, fast_accept, mc_applied)
    if notes is not None:
        if mc_applied:
            notes.append(f"APNG: all {n} frames at full color — stabilized the camera shake so it "
                         f"fits {kb} KB without dropping frames or colors.")
        elif used <= 0.0:
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
