"""Bridge: encode a GIF output through the Fovea encoder, with a safe fallback.

The orchestrator's GIF outputs (gif + emoji) route through here. We run Fovea's
``encode()`` to hit the byte budget while judging quality perceptually, with the
legacy ffmpeg path as an automatic fallback.

When the **native engine** is built (`FoveaNativeEngine`), per-frame local palettes
give rich color with *every* frame on easy clips, so it just trusts the metric-driven
encode. On HARD clips (full-frame motion) even per-frame palettes can only fit a few
colors across 72 frames, so the frames-vs-color ``priority`` still applies — it trims
frames so the per-frame palette gets richer. The **legacy ffmpeg fallback** keeps the
same priority split (its single global palette forces the tradeoff on every clip):

  * smooth   — frames first: keep every frame; color is whatever fits.
  * balanced — most frames whose palette still fills the budget, then top off
               with frames.
  * sharp    — color first: trim frames for the richest palette, then add frames
               back to use the budget.

Trimming never drops below an fps floor (no 2-fps slideshows). Either way we never
knowingly leave budget on the table: after choosing a palette we add frames back at
that palette+dither until the byte limit is used. Tunables:
  USE_FOVEA_GIF (on), FOVEA_AUTOBALANCE (on), FOVEA_BUDGET_USE (0.93),
  FOVEA_BUDGET_SECONDS (12), FOVEA_MAX_ATTEMPTS (12), FOVEA_COMPARE (on),
  FOVEA_MIN_FPS (8), FOVEA_MIN_FPS_SHARP (5).
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import tempfile
import threading
import time
from collections import OrderedDict

log = logging.getLogger("fovea.bridge")

MIN_FRAMES = 6      # absolute floor (fit-rescue may reach it); the user-facing rich floor is 24
GREAT_COLORS = 96   # a per-frame palette this rich is banding-free -> stop trimming, keep the frames
# Color-aware perceptual distance at/below which the result is "no longer washed". The
# metric's own perceptually-lossless threshold is ~0.02; the washout->clean knee sits just
# above it (measured: 8 colors=0.037 still washed, 20 colors=0.019 clean), so ~0.03 stops
# at the *highest* resolution whose per-frame palette is rich enough to kill banding.
# Tunable without a logic redeploy via FOVEA_GOOD_DIST.
GOOD_DIST = float(os.getenv("FOVEA_GOOD_DIST", "0.03"))


def _enabled() -> bool:
    return os.getenv("USE_FOVEA_GIF", "1").lower() not in ("0", "false", "no")


def compare_enabled() -> bool:
    return os.getenv("FOVEA_COMPARE", "1").lower() not in ("0", "false", "no")


def _autobalance_enabled() -> bool:
    return os.getenv("FOVEA_AUTOBALANCE", "1").lower() not in ("0", "false", "no")


def _color_floor_for(priority: str) -> int:
    """Target palette richness per mode. 0 = never trim frames (frames-first);
    higher = trim more frames for a richer palette (color-first)."""
    if not _autobalance_enabled():
        return 0
    return {"smooth": 0, "balanced": 64, "sharp": 160}.get(priority, 64)


def _native_available() -> bool:
    """True when the native per-frame-palette engine is built. It keeps every frame
    AND rich color, so the color-floor frame-trimming bandaid below is unnecessary on
    that path. ``FOVEA_FORCE_LEGACY_BRIDGE=1`` forces the legacy dance (escape hatch)."""
    if os.getenv("FOVEA_FORCE_LEGACY_BRIDGE", "").lower() in ("1", "true", "yes"):
        return False
    try:
        from encoder.core.engines import FoveaNativeEngine

        return FoveaNativeEngine.available()
    except Exception:  # noqa: BLE001
        return False


# --------------------------------------------------------------------------- #
# Legacy-baseline cache: the side-by-side's standard-encoder result depends only on
# the fitted frames + size/colors/fps — NOT on Fovea's priority/mode. Cache it so
# tweaking priority doesn't re-run the standard encoder on every regenerate.
# --------------------------------------------------------------------------- #
_LEGACY_TTL = 900.0
_LEGACY_MAX = 8
_legacy_lock = threading.Lock()
_legacy_cache: "OrderedDict[str, tuple]" = OrderedDict()  # sig -> (data, n_frames, ts)


def _legacy_sig(fitted, delays, budget: int, max_colors: int, fps_cap) -> str:
    """Cheap content signature of the legacy inputs (samples ~4 frames)."""
    h = hashlib.sha1()
    n = len(fitted)
    h.update(f"{n}|{fitted[0].shape}|{list(delays)}|{budget}|{max_colors}|{fps_cap}".encode())
    for i in range(0, n, max(1, n // 4)):
        h.update(fitted[i].tobytes())
    return h.hexdigest()


def _legacy_get(sig: str):
    with _legacy_lock:
        e = _legacy_cache.get(sig)
        if e is not None and (time.time() - e[2]) <= _LEGACY_TTL:
            _legacy_cache.move_to_end(sig)
            return e[0], e[1]
        if e is not None:
            _legacy_cache.pop(sig, None)
    return None


def _legacy_put(sig: str, data: bytes, n_frames: int) -> None:
    with _legacy_lock:
        _legacy_cache[sig] = (data, n_frames, time.time())
        _legacy_cache.move_to_end(sig)
        while len(_legacy_cache) > _LEGACY_MAX:
            _legacy_cache.popitem(last=False)


# --------------------------------------------------------------------------- #
# Encoding primitives
# --------------------------------------------------------------------------- #

def _encode_once(fitted, delays, budget: int, seconds: float, attempts: int, mode: str = "cap"):
    """One Fovea encode -> (bytes, output_fps, colors, report).

    ``report`` carries the honesty fields from the JSON sidecar (whether the result
    stayed perceptually lossless, where any loss landed, why the search stopped)."""
    from encoder import encode as fovea_encode

    td = tempfile.mkdtemp(prefix="fovea_run_")
    try:
        out = os.path.join(td, "o.gif")
        rep = out + ".json"
        res = fovea_encode(
            list(fitted), target_bytes=budget, mode=mode, delays_ms=list(delays),
            max_attempts=attempts, budget_seconds=seconds, out_path=out, report_path=rep,
        )
        with open(out, "rb") as fh:
            data = fh.read()
        colors, report = None, {"mode": mode}
        try:
            rj = json.load(open(rep))
            colors = (rj.get("lever_setting") or {}).get("colors")
            report = {
                "mode": mode,
                "dither": (rj.get("lever_setting") or {}).get("dither"),
                "perceptually_lossless": rj.get("perceptually_lossless"),
                "perceptual_distance": rj.get("perceptual_distance"),
                "under_target": rj.get("under_target"),
                "stopped_early": rj.get("stopped_early"),
                "stop_reason": rj.get("stop_reason"),
                "loss_locus": rj.get("loss_locus"),
            }
        except Exception:  # noqa: BLE001
            pass
        return data, res.output_fps, colors, report
    finally:
        shutil.rmtree(td, ignore_errors=True)


def _encode_fixed_colors(fitted, delays, colors: int, dither: str | None = "sierra2_4a"):
    """Encode at a FIXED palette + dither (so we can add frames while holding color).

    Matching the chosen candidate's dither matters: re-encoding at a different dither
    changes the byte size, which would break the frame-fill's size estimate."""
    from encoder.core.engines import FfmpegPaletteEngine, prepare_context
    from encoder.core.frames import frames_from_list
    from encoder.core.levers import LeverState

    td = tempfile.mkdtemp(prefix="fovea_fix_")
    try:
        ctx = prepare_context(frames_from_list(list(fitted), list(delays)), 1.0, td)
        out = os.path.join(td, "f.gif")
        eo = FfmpegPaletteEngine().encode(
            ctx, LeverState(colors=colors, dither=dither or "sierra2_4a"), out)
        with open(out, "rb") as fh:
            return fh.read(), eo.size_bytes, ctx.fps
    finally:
        shutil.rmtree(td, ignore_errors=True)


def _fill_frames_at_colors(fitted, delays, budget: int, base_frames: int, base_size: int,
                           colors: int, dither: str | None = None):
    """Add frames at a fixed palette to use the budget -> (data, n, fps, usage) or None.

    GIF size is ~linear in frame count at a fixed palette, so estimate the largest
    frame count that still fits and take the most frames that do.
    """
    from .encode import even_subsample

    total = len(fitted)
    if base_size <= 0:
        return None
    est = int(base_frames * budget / base_size)
    for f2 in sorted({min(total, est + 1), min(total, est), min(total, est - 1),
                      min(total, est - 2)}, reverse=True):
        if f2 <= base_frames:
            continue
        wf, wd = even_subsample(list(fitted), list(delays), f2)
        data, size, fps = _encode_fixed_colors(wf, wd, colors, dither)
        if size <= budget:
            return data, f2, fps, size / budget
    return None


def _seconds_left(deadline: float | None, default: float) -> float:
    """Per-encode wall-clock budget, shrunk to fit a job-level ``deadline``."""
    if deadline is None:
        return default
    return max(0.0, min(default, deadline - time.monotonic()))


def _run_fovea(fitted, delays, budget: int, priority: str = "balanced", mode: str = "cap",
               deadline: float | None = None, allow_descent: bool = True):
    """Encode per ``priority``/``mode`` -> (bytes, n_frames, fps, colors, report).

    ``cap`` fills the byte budget (3-phase); ``invisible`` skips the budget-fill and
    returns the smallest perceptually-lossless GIF under ``budget`` as a ceiling."""
    from .encode import even_subsample

    total = len(fitted)
    target = float(os.getenv("FOVEA_BUDGET_USE", "0.93"))
    per_encode = float(os.getenv("FOVEA_BUDGET_SECONDS", "12"))
    attempts = int(os.getenv("FOVEA_MAX_ATTEMPTS", "12"))

    # NATIVE PATH: the in-Rust byte-target search keeps every frame and guarantees a
    # <= budget result (lowering the per-frame color budget, then resolution if forced),
    # shrinking for invisible mode internally. On easy clips (screen recordings, partial
    # motion) that already yields rich color with every frame. But on HARD clips
    # (full-frame motion / film grain — little reuse, incompressible) even per-frame
    # palettes fit only ~2 colors at full res -> washout. So we honor `priority`: "more
    # frames" keeps them all (color is whatever fits); "balanced"/"richer color" chase a
    # clean (un-washed) palette with TWO levers, in order — (1) trim frames toward the
    # >=24-frame floor (keeps resolution; best when a clip just has more frames than the
    # budget can color), then (2) drop RESOLUTION keeping the frames (the decisive lever
    # on grainy/full-motion content, where trimming buys no color). Each lever stops as
    # soon as the result is no longer washed.
    if _native_available():
        secs = lambda: max(1.0, _seconds_left(deadline, per_encode))  # noqa: E731
        data, fps, colors, report = _encode_once(fitted, delays, budget, secs(), attempts, mode=mode)
        chosen = (data, total, fps, colors, report)
        log.info("fovea.native mode=%s frames=%d colors=%s bytes=%d usage=%.2f",
                 mode, total, colors, len(data), (len(data) / budget) if budget else 1.0)

        def _rich(c, rep):
            # "no longer washed" = the judge sees no visible loss, the color-aware distance
            # is at/below the washout->clean knee, or the palette is already objectively rich.
            if bool(rep.get("perceptually_lossless")):
                return True
            d = rep.get("perceptual_distance")
            if d is not None and d <= GOOD_DIST:
                return True
            return (c or 0) >= GREAT_COLORS

        # "more frames" (smooth) keeps every frame. "balanced"/"richer color" first try to win
        # color by trimming frames toward (never below) the >=24-frame floor — this keeps full
        # RESOLUTION, the better trade when a clip simply has more frames than the budget can
        # color richly. We STOP at the LARGEST frame count that's no longer washed. But on
        # full-motion / grainy content (the John Wick case) the per-frame entropy is so high
        # that even few frames can't hold color at full res, so trimming buys nothing — we
        # detect that (a trim step that fails to add color) and bail immediately to the
        # resolution lever below instead of burning the whole deadline on dead-end trims.
        if mode != "invisible" and _color_floor_for(priority) and not _rich(colors, report):
            duration_s = (sum(delays) / 1000.0) if delays and any(delays) else (total / 10.0)
            min_fps = float(os.getenv("FOVEA_MIN_FPS_SHARP", "5") if priority == "sharp"
                            else os.getenv("FOVEA_MIN_FPS", "8"))
            rich_min = int(os.getenv("FOVEA_RICH_MIN_FRAMES", "24"))
            fps_floor_n = int(round(min_fps * duration_s)) if duration_s > 0 else MIN_FRAMES
            min_n = min(total, max(MIN_FRAMES, rich_min, fps_floor_n))
            f = total
            best_c = colors or 0
            while f > min_n:
                if deadline is not None and (deadline - time.monotonic()) < 1.5:
                    break  # out of time — keep the best candidate so far
                f = max(min_n, int(f * 0.6))
                wf, wd = even_subsample(list(fitted), list(delays), f)
                d, fp, c, rep = _encode_once(wf, wd, budget, secs(), attempts, mode="cap")
                log.info("fovea.native_trim mode=%s frames=%d colors=%s bytes=%d lossless=%s",
                         priority, f, c, len(d), rep.get("perceptually_lossless"))
                if _rich(c, rep):
                    chosen = (d, f, fp, c, rep)   # largest frame count that's clean -> done
                    break
                if (c or 0) > best_c:
                    chosen = (d, f, fp, c, rep)   # improving -> keep the richest as a fallback
                    best_c = c or 0
                else:
                    break  # trimming isn't buying color (grain-dominated) -> use resolution

        # RESOLUTION-for-color (the decisive lever for detailed/grainy clips). If color is
        # still washed at the >=24-frame floor, the user pinned frames + color + budget on a
        # high-entropy clip — which over-constrains it — so the only thing left to give is
        # pixels/frame. Downscaling also averages out incompressible grain, so each per-frame
        # palette suddenly fits far more real color. Shrink ONLY until no longer washed (the
        # highest such resolution), KEEPING the frames, and report the new size honestly.
        data, n, fps, colors, report = chosen
        if allow_descent and mode != "invisible" and priority != "smooth" and not _rich(colors, report):
            import numpy as _np
            from PIL import Image as _Image

            bf, bd = (even_subsample(list(fitted), list(delays), n) if n < total
                      else (list(fitted), list(delays)))
            h0, w0 = bf[0].shape[:2]
            best = chosen
            for sc in (0.85, 0.72, 0.6, 0.5):
                if deadline is not None and (deadline - time.monotonic()) < 1.5:
                    break
                # Scale the longest side by `sc` and derive the other from the source
                # ratio, so the aspect ratio is held exactly (no rounding drift) at every
                # resolution the descent visits — "Source" keeps the source's shape.
                if w0 >= h0:
                    nw = max(16, round(w0 * sc)); nh = max(16, round(nw * h0 / w0))
                else:
                    nh = max(16, round(h0 * sc)); nw = max(16, round(nh * w0 / h0))
                rf = [_np.asarray(_Image.fromarray(fr).resize((nw, nh), _Image.LANCZOS)) for fr in bf]
                d2, fp2, c2, rep2 = _encode_once(rf, bd, budget, secs(), attempts, mode="cap")
                log.info("fovea.native_scale mode=%s frames=%d dim=%dx%d colors=%s bytes=%d dist=%s",
                         priority, n, nw, nh, c2, len(d2), rep2.get("perceptual_distance"))
                rep2 = {**rep2, "scaled_dim": f"{nw}x{nh}"}
                # Prefer a candidate that FITS the budget over one that doesn't; among fitting
                # ones the richest color (lower res buys color); among non-fitting ones the
                # smallest (closest to fitting). Lower res tends to both fit and add color, so
                # the descent walks toward "fits AND no longer washed" and never returns a
                # larger-than-budget result when a fitting one was seen.
                cf, best_fits = len(d2) <= budget, len(best[0]) <= budget
                better = (cf and not best_fits) or (cf == best_fits and (
                    (c2 or 0) > (best[3] or 0) if cf else len(d2) < len(best[0])))
                if better:
                    best = (d2, n, fp2, c2, rep2)
                if cf and _rich(c2, rep2):
                    break  # highest resolution that both fits and is no longer washed -> done
            chosen = best
        return chosen

    # ---- LEGACY ffmpeg fallback: invisible handling + the 3-phase frames-vs-color dance.
    # Invisible: smallest perceptually-lossless GIF; if a clip can't be made lossless under
    # the ceiling, fall THROUGH to cap budget-fill (never worse than cap).
    if mode == "invisible":
        data, fps, colors, report = _encode_once(
            fitted, delays, budget, max(1.0, _seconds_left(deadline, per_encode)), attempts,
            mode="invisible")
        if report.get("perceptually_lossless"):
            log.info("fovea.invisible frames=%d colors=%s bytes=%d lossless=True",
                     total, colors, len(data))
            return data, total, fps, colors, report
        log.info("fovea.invisible_fallback frames=%d colors=%s bytes=%d lossless=False; "
                 "using cap budget-fill", total, colors, len(data))

    floor = _color_floor_for(priority)
    # fps floor: never trim into a slideshow. 'sharp' may trim further for richer color
    # than the smoother modes, but both stay above a watchable frame rate. Duration is
    # preserved across subsampling, so frames / duration == output fps.
    duration_s = (sum(delays) / 1000.0) if delays and any(delays) else (total / 10.0)
    min_fps = float(os.getenv("FOVEA_MIN_FPS_SHARP", "5") if priority == "sharp"
                    else os.getenv("FOVEA_MIN_FPS", "8"))
    min_n = (max(MIN_FRAMES, min(total, int(round(min_fps * duration_s))))
             if duration_s > 0 else MIN_FRAMES)

    # 1. All frames at the richest palette the budget allows.
    data, fps, colors, report = _encode_once(
        fitted, delays, budget, max(1.0, _seconds_left(deadline, per_encode)), attempts)
    usage = (len(data) / budget) if budget else 1.0
    chosen = (data, total, fps, colors, usage, report)
    log.info("fovea.fill mode=%s frames=%d colors=%s bytes=%d usage=%.2f",
             priority, total, colors, len(data), usage)

    # 2. Color-seeking trim (balanced/sharp): drop frames toward the palette floor, but
    #    never below the fps floor (no slideshows). Skipped for 'smooth' or when the
    #    palette is already rich enough.
    # 2a. Color-seeking trim (balanced/sharp): drop frames toward the palette floor,
    #     clamped at the fps floor so we never produce a slideshow.
    if floor and (colors or 0) < floor and (colors or 0) < 256 and total > min_n:
        f = total
        for _ in range(6):
            if deadline is not None and (deadline - time.monotonic()) < 1.0:
                break  # out of time — keep the best candidate so far
            f = max(min_n, int(f * 0.72))   # clamp: never step past the fps floor
            wf, wd = even_subsample(list(fitted), list(delays), f)
            d, fp, c, rep = _encode_once(
                wf, wd, budget, _seconds_left(deadline, per_encode), attempts)
            log.info("fovea.fill mode=%s frames=%d colors=%s bytes=%d usage=%.2f",
                     priority, f, c, len(d), (len(d) / budget) if budget else 1.0)
            chosen = (d, f, fp, c, (len(d) / budget) if budget else 1.0, rep)
            if f <= min_n or (len(d) <= budget and ((c or 0) >= floor or (c or 0) >= 256)):
                break

    # 2b. Fit rescue: if even the fps-floor result overshoots the budget, keep trimming
    #     below the floor toward MIN_FRAMES — fitting the hard byte limit is mandatory;
    #     the fps floor is only a preference.
    data, n, fps, colors, usage, report = chosen
    if len(data) > budget and n > MIN_FRAMES:
        f = n
        for _ in range(4):
            if deadline is not None and (deadline - time.monotonic()) < 1.0:
                break
            f = max(MIN_FRAMES, int(f * 0.72))
            wf, wd = even_subsample(list(fitted), list(delays), f)
            d, fp, c, rep = _encode_once(
                wf, wd, budget, _seconds_left(deadline, per_encode), attempts)
            log.info("fovea.fitrescue mode=%s frames=%d colors=%s bytes=%d usage=%.2f",
                     priority, f, c, len(d), (len(d) / budget) if budget else 1.0)
            chosen = (d, f, fp, c, (len(d) / budget) if budget else 1.0, rep)
            if len(d) <= budget or f <= MIN_FRAMES:
                break
        data, n, fps, colors, usage, report = chosen
    # 3. Frame-fill: top off the budget by adding frames back at the chosen palette+dither
    #    (more frames = smoother, no extra washout). Never leave budget on the table.
    if (priority != "smooth" and usage < target and n < total and colors
            and (deadline is None or (deadline - time.monotonic()) > 1.0)):
        filled = _fill_frames_at_colors(fitted, delays, budget, n, len(data), int(colors),
                                        report.get("dither"))
        if filled is not None and filled[3] > usage:
            data, n, fps, usage = filled
            log.info("fovea.framefill mode=%s frames=%d colors=%s bytes=%d usage=%.2f",
                     priority, n, colors, len(data), usage)
    return data, n, fps, colors, report


def _fovea_note(total: int, kept: int, colors) -> str:
    if kept < total:
        c = f"{colors} colors" if colors else "more color"
        return f"Fovea kept {kept} of {total} frames to hold {c} (avoids washed-out color)."
    if colors:
        return f"Fovea kept all {total} frames at {colors} colors."
    return f"Fovea kept all {total} frames."


# --------------------------------------------------------------------------- #
# Public entry points
# --------------------------------------------------------------------------- #

def gif_encode(fitted, delays, *, budget, max_colors=256, fps_cap=24, priority="balanced",
               mode="cap", deadline=None, notes=None, allow_descent=True):
    """Return ``(bytes, "GIF", n_frames, fps, report)`` for the fitted frames under ``budget``.

    ``allow_descent`` lets the encoder trade resolution for color (Auto GIF); set False to
    lock the given dimensions (Source / Custom W×H / square emoji) and fit via color only."""
    if _enabled():
        try:
            data, n, fps, colors, report = _run_fovea(
                fitted, delays, int(budget), priority, mode, deadline, allow_descent)
            if notes is not None:
                notes.append(_fovea_note(len(fitted), n, colors))
                if isinstance(report, dict) and report.get("scaled_dim"):
                    notes.append(f"Scaled down (keeping your source aspect ratio) so all {n} frames "
                                 f"could hold rich color within the size limit — raise the limit or "
                                 f"pick “More frames” to keep the full resolution.")
                if mode == "invisible":
                    kb = len(data) // 1024
                    if report.get("mode") == "invisible":          # true smallest-lossless result
                        notes.append(f"Shrunk to the smallest perceptually-lossless size ({kb} KB).")
                    elif report.get("perceptually_lossless"):      # fell back, but still lossless
                        notes.append(f"Couldn't shrink further without visible loss; kept a "
                                     f"perceptually-lossless {kb} KB fit.")
                    else:                                          # fell back, not losslessly possible
                        notes.append("Couldn't reach a no-visible-loss size for this clip; "
                                     "kept the best-looking fit instead.")
            return data, "GIF", n, fps, report
        except Exception as exc:  # noqa: BLE001 - never let Fovea break the pipeline
            log.warning("fovea.bridge_failed; falling back to legacy: %s", str(exc)[:200])
            if notes is not None:
                notes.append("Fovea encode failed; used the standard encoder.")
    from .encode import encode_gif as legacy

    data, fmt, n, fps = legacy(fitted, delays, budget=budget, max_colors=max_colors, fps_cap=fps_cap)
    return data, fmt, n, fps, None


def _aligned_distance(metric, fitted, delays, gif_path, n_frames: int) -> float:
    """Distance between a GIF and the SAME frames it kept (subsample the source to
    match), so a frame-trimmed candidate is judged on color/spatial fidelity rather
    than on misaligned frames. Frame *count* is reported separately for motion."""
    from encoder.core.frames import frames_from_list, load_gif

    if n_frames < len(fitted):
        from .encode import even_subsample

        sf, sd = even_subsample(list(fitted), list(delays), n_frames)
    else:
        sf, sd = list(fitted), list(delays)
    return metric.distance(frames_from_list(sf, sd), load_gif(gif_path)).distance


def gif_encode_compare(fitted, delays, *, budget, max_colors=256, fps_cap=24,
                       priority="balanced", deadline=None, notes=None, allow_descent=True):
    """Encode with BOTH Fovea and the legacy encoder for a side-by-side (cap mode only).

    Returns ``(fovea_bytes, "GIF", n_frames, fps, comparison, legacy_bytes, report)``.
    Each side's perceptual distance is measured against the source subsampled to that
    side's frame count, so the comparison is fair when frame counts differ.
    """
    from encoder.metrics import default_metric

    from .encode import encode_gif as legacy_encode

    td = tempfile.mkdtemp(prefix="fovea_cmp_")
    try:
        fovea_data, fovea_n, fovea_fps, fovea_colors, report = _run_fovea(
            fitted, delays, int(budget), priority, "cap", deadline, allow_descent)

        sig = _legacy_sig(fitted, delays, budget, max_colors, fps_cap)
        cached = _legacy_get(sig)
        if cached is not None:
            legacy_data, legacy_n = cached
        else:
            legacy_data, _, legacy_n, _ = legacy_encode(
                fitted, delays, budget=budget, max_colors=max_colors, fps_cap=fps_cap)
            _legacy_put(sig, legacy_data, legacy_n)
        lpath = os.path.join(td, "legacy.gif")
        with open(lpath, "wb") as fh:
            fh.write(legacy_data)

        # Single source of truth for the Fovea side: the encoder's own report — the same
        # numbers the honesty line shows — so the comparison badge can never contradict
        # it. Only the legacy baseline needs its own score (it has no Fovea report).
        metric = default_metric()
        ldist = _aligned_distance(metric, fitted, delays, lpath, legacy_n)
        fdist = report.get("perceptual_distance")
        comparison = {
            "metric": metric.name,
            "fovea": {"bytes": len(fovea_data), "frames": fovea_n, "colors": fovea_colors,
                      "distance": (round(float(fdist), 5) if fdist is not None else None),
                      "perceptually_lossless": bool(report.get("perceptually_lossless"))},
            "legacy": {"bytes": len(legacy_data), "frames": int(legacy_n),
                       "distance": round(float(ldist), 5)},
        }
        if notes is not None:
            notes.append(_fovea_note(len(fitted), fovea_n, fovea_colors))
        return fovea_data, "GIF", fovea_n, fovea_fps, comparison, legacy_data, report
    finally:
        shutil.rmtree(td, ignore_errors=True)
