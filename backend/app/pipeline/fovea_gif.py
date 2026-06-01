"""Bridge: encode a GIF output through the Fovea encoder, with a safe fallback.

The orchestrator's GIF outputs (gif + emoji) route through here. We run Fovea's
``encode()`` to hit the byte budget while judging quality perceptually, with the
legacy ffmpeg path as an automatic fallback.

The budget is split between frames (smoothness) and colors (richness) per a
``priority`` (reusing the smooth/balanced/sharp control):

  * smooth   — frames first: keep every frame; color is whatever fits.
  * balanced — most frames whose palette still fills the budget, then top off
               with frames.
  * sharp    — color first: trim frames for the richest palette, then add frames
               back to use the budget.

Either way we never knowingly leave budget on the table: after choosing a palette
we add frames back at that palette until the byte limit is used. Tunables:
  USE_FOVEA_GIF (on), FOVEA_AUTOBALANCE (on), FOVEA_BUDGET_USE (0.93),
  FOVEA_BUDGET_SECONDS (12), FOVEA_MAX_ATTEMPTS (12), FOVEA_COMPARE (on).
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
import time

log = logging.getLogger("fovea.bridge")

MIN_FRAMES = 6  # never trim a clip below this many frames


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


def _encode_fixed_colors(fitted, delays, colors: int):
    """Encode at a FIXED palette size (so we can add frames while holding color)."""
    from encoder.core.engines import FfmpegPaletteEngine, prepare_context
    from encoder.core.frames import frames_from_list
    from encoder.core.levers import LeverState

    td = tempfile.mkdtemp(prefix="fovea_fix_")
    try:
        ctx = prepare_context(frames_from_list(list(fitted), list(delays)), 1.0, td)
        out = os.path.join(td, "f.gif")
        eo = FfmpegPaletteEngine().encode(ctx, LeverState(colors=colors, dither="sierra2_4a"), out)
        with open(out, "rb") as fh:
            return fh.read(), eo.size_bytes, ctx.fps
    finally:
        shutil.rmtree(td, ignore_errors=True)


def _fill_frames_at_colors(fitted, delays, budget: int, base_frames: int, base_size: int,
                           colors: int):
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
        data, size, fps = _encode_fixed_colors(wf, wd, colors)
        if size <= budget:
            return data, f2, fps, size / budget
    return None


def _seconds_left(deadline: float | None, default: float) -> float:
    """Per-encode wall-clock budget, shrunk to fit a job-level ``deadline``."""
    if deadline is None:
        return default
    return max(0.0, min(default, deadline - time.monotonic()))


def _run_fovea(fitted, delays, budget: int, priority: str = "balanced", mode: str = "cap",
               deadline: float | None = None):
    """Encode per ``priority``/``mode`` -> (bytes, n_frames, fps, colors, report).

    ``cap`` fills the byte budget (3-phase); ``invisible`` skips the budget-fill and
    returns the smallest perceptually-lossless GIF under ``budget`` as a ceiling."""
    from .encode import even_subsample

    total = len(fitted)
    target = float(os.getenv("FOVEA_BUDGET_USE", "0.93"))
    per_encode = float(os.getenv("FOVEA_BUDGET_SECONDS", "12"))
    attempts = int(os.getenv("FOVEA_MAX_ATTEMPTS", "12"))

    # Invisible: aim for the smallest perceptually-lossless GIF, every frame kept.
    # If the clip CAN'T be made lossless under the ceiling, fall THROUGH to cap-mode
    # budget-fill: the encoder never trims frames, so a stuck-low-palette invisible
    # result is washed out and worse than the default. Falling back guarantees
    # invisible is never worse than cap (see tests/test_invisible_fallback.py).
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

    # 1. All frames at the richest palette the budget allows.
    data, fps, colors, report = _encode_once(
        fitted, delays, budget, max(1.0, _seconds_left(deadline, per_encode)), attempts)
    usage = (len(data) / budget) if budget else 1.0
    chosen = (data, total, fps, colors, usage, report)
    log.info("fovea.fill mode=%s frames=%d colors=%s bytes=%d usage=%.2f",
             priority, total, colors, len(data), usage)

    # 2. Color-seeking trim (balanced/sharp): drop frames until the palette reaches the
    #    mode's richness floor. Skipped for 'smooth' or when all frames are already rich.
    if floor and (colors or 0) < floor and (colors or 0) < 256:
        f = total
        for _ in range(5):
            if deadline is not None and (deadline - time.monotonic()) < 1.0:
                break  # out of time — keep the best candidate so far
            f = max(MIN_FRAMES, int(f * 0.72))
            wf, wd = even_subsample(list(fitted), list(delays), f)
            d, fp, c, rep = _encode_once(
                wf, wd, budget, _seconds_left(deadline, per_encode), attempts)
            log.info("fovea.fill mode=%s frames=%d colors=%s bytes=%d usage=%.2f",
                     priority, f, c, len(d), (len(d) / budget) if budget else 1.0)
            chosen = (d, f, fp, c, (len(d) / budget) if budget else 1.0, rep)
            if (c or 0) >= floor or (c or 0) >= 256 or f <= MIN_FRAMES:
                break

    data, n, fps, colors, usage, report = chosen
    # 3. Frame-fill: top off the budget by adding frames back at the chosen palette
    #    (more frames = smoother, no extra washout). Never leave budget on the table.
    if (priority != "smooth" and usage < target and n < total and colors
            and (deadline is None or (deadline - time.monotonic()) > 1.0)):
        filled = _fill_frames_at_colors(fitted, delays, budget, n, len(data), int(colors))
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
               mode="cap", deadline=None, notes=None):
    """Return ``(bytes, "GIF", n_frames, fps, report)`` for the fitted frames under ``budget``."""
    if _enabled():
        try:
            data, n, fps, colors, report = _run_fovea(
                fitted, delays, int(budget), priority, mode, deadline)
            if notes is not None:
                notes.append(_fovea_note(len(fitted), n, colors))
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
                       priority="balanced", deadline=None, notes=None):
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
            fitted, delays, int(budget), priority, "cap", deadline)
        fpath = os.path.join(td, "fovea.gif")
        with open(fpath, "wb") as fh:
            fh.write(fovea_data)

        legacy_data, _, legacy_n, _ = legacy_encode(
            fitted, delays, budget=budget, max_colors=max_colors, fps_cap=fps_cap)
        lpath = os.path.join(td, "legacy.gif")
        with open(lpath, "wb") as fh:
            fh.write(legacy_data)

        metric = default_metric()
        fdist = _aligned_distance(metric, fitted, delays, fpath, fovea_n)
        ldist = _aligned_distance(metric, fitted, delays, lpath, legacy_n)
        lossless = fdist <= metric.invisible_threshold
        comparison = {
            "metric": metric.name,
            "fovea": {"bytes": len(fovea_data), "frames": fovea_n, "colors": fovea_colors,
                      "distance": round(float(fdist), 5), "perceptually_lossless": bool(lossless)},
            "legacy": {"bytes": len(legacy_data), "frames": int(legacy_n),
                       "distance": round(float(ldist), 5)},
        }
        if notes is not None:
            notes.append(_fovea_note(len(fitted), fovea_n, fovea_colors))
        return fovea_data, "GIF", fovea_n, fovea_fps, comparison, legacy_data, report
    finally:
        shutil.rmtree(td, ignore_errors=True)
