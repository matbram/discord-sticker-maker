"""Bridge: encode a GIF output through the Fovea encoder, with a safe fallback.

The orchestrator's GIF outputs (gif + emoji) route through here. We run Fovea's
``encode()`` to hit the byte budget while judging quality perceptually, with the
legacy ffmpeg path as an automatic fallback.

Auto-balance (default on): keeping *every* frame at a tight size forces a tiny
palette (washed-out color) and, because palette size jumps in lumps, leaves the
budget unused. So we iterate: encode, and if we're using well under the budget,
trim frames a little (which lets a richer palette fit) and re-encode — until the
budget is actually used. The result is the most frames whose palette fills the
budget: richer color AND the byte limit put to work. Tunables:
  USE_FOVEA_GIF (on), FOVEA_AUTOBALANCE (on), FOVEA_COLOR_FLOOR (48),
  FOVEA_BUDGET_USE (0.90), FOVEA_BUDGET_SECONDS (12), FOVEA_MAX_ATTEMPTS (12),
  FOVEA_COMPARE (on).
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile

log = logging.getLogger("fovea.bridge")

MIN_FRAMES = 6  # never trim a clip below this many frames for color


def _enabled() -> bool:
    return os.getenv("USE_FOVEA_GIF", "1").lower() not in ("0", "false", "no")


def compare_enabled() -> bool:
    return os.getenv("FOVEA_COMPARE", "1").lower() not in ("0", "false", "no")


def _autobalance_enabled() -> bool:
    return os.getenv("FOVEA_AUTOBALANCE", "1").lower() not in ("0", "false", "no")


def _color_floor() -> int:
    try:
        return max(2, int(os.getenv("FOVEA_COLOR_FLOOR", "48")))
    except ValueError:
        return 48


# --------------------------------------------------------------------------- #
# Auto-balance: choose how many frames to keep so the palette stays rich.
# --------------------------------------------------------------------------- #

def _probe_size_at_colors(fitted, delays, colors: int) -> int | None:
    """Measure the GIF size of ALL frames at a fixed color count (one encode)."""
    try:
        from encoder.core.engines import FfmpegPaletteEngine, prepare_context
        from encoder.core.frames import frames_from_list
        from encoder.core.levers import LeverState

        td = tempfile.mkdtemp(prefix="fovea_probe_")
        try:
            ctx = prepare_context(frames_from_list(list(fitted), list(delays)), 1.0, td)
            out = FfmpegPaletteEngine().encode(
                ctx, LeverState(colors=colors, dither="sierra2_4a"), os.path.join(td, "p.gif"))
            return out.size_bytes
        finally:
            shutil.rmtree(td, ignore_errors=True)
    except Exception as exc:  # noqa: BLE001
        log.warning("fovea.probe_failed: %s", str(exc)[:160])
        return None


def _autobalanced_frames(fitted, delays, budget: int):
    """Return the frames Fovea should encode: all of them if they fit the color
    floor, else an evenly-spread subset sized so the floor fits the budget."""
    total = len(fitted)
    if not _autobalance_enabled() or total <= MIN_FRAMES:
        return fitted, delays
    floor = _color_floor()
    size_at_floor = _probe_size_at_colors(fitted, delays, floor)
    if size_at_floor is None or size_at_floor <= budget:
        return fitted, delays  # all frames already hold the color floor
    # GIF size is ~linear in frame count; size for the budget at the floor.
    keep = max(MIN_FRAMES, int(total * budget / size_at_floor * 0.90))
    if keep >= total:
        return fitted, delays
    from .encode import even_subsample

    work_f, work_d = even_subsample(list(fitted), list(delays), keep)
    log.info("fovea.autobalance total=%d keep=%d floor=%d size_at_floor=%d budget=%d",
             total, len(work_f), floor, size_at_floor, budget)
    return work_f, work_d


def _run_fovea(fitted, delays, budget: int):
    """Budget-filling Fovea encode -> (bytes, n_frames, output_fps, colors).

    Keeping every frame at a tight size caps the palette (washout) and leaves the
    budget unused because palette size jumps in lumps. So we iterate: encode, and
    if we're using well under the budget, trim frames a little (which lets a richer
    palette fit) and re-encode — until the budget is actually used or we hit the
    frame floor. The result is the most frames whose palette fills the budget.
    """
    from .encode import even_subsample

    total = len(fitted)
    target_use = float(os.getenv("FOVEA_BUDGET_USE", "0.90"))
    seconds = float(os.getenv("FOVEA_BUDGET_SECONDS", "12"))
    attempts = int(os.getenv("FOVEA_MAX_ATTEMPTS", "12"))

    # Start from the auto-balance estimate (leaves room for the color floor).
    work_f, work_d = _autobalanced_frames(fitted, delays, budget) if _autobalance_enabled() else (fitted, delays)
    best = None  # (data, n_frames, fps, colors, usage)
    for i in range(5):
        data, fps, colors = _encode_once(work_f, work_d, budget, seconds, attempts)
        usage = (len(data) / budget) if budget else 1.0
        if best is None or usage > best[4]:
            best = (data, len(work_f), fps, colors, usage)
        log.info("fovea.budgetfill iter=%d frames=%d colors=%s bytes=%d usage=%.2f",
                 i, len(work_f), colors, len(data), usage)
        if (not _autobalance_enabled() or usage >= target_use
                or (colors and colors >= 256) or len(work_f) <= MIN_FRAMES):
            break
        nf = max(MIN_FRAMES, int(len(work_f) * 0.8))
        if nf >= len(work_f):
            break
        work_f, work_d = even_subsample(list(fitted), list(delays), nf)
    return best[0], best[1], best[2], best[3]


def _encode_once(fitted, delays, budget: int, seconds: float, attempts: int):
    """One Fovea encode -> (bytes, output_fps, colors)."""
    from encoder import encode as fovea_encode

    td = tempfile.mkdtemp(prefix="fovea_run_")
    try:
        out = os.path.join(td, "o.gif")
        rep = out + ".json"
        res = fovea_encode(
            list(fitted), target_bytes=budget, mode="cap", delays_ms=list(delays),
            max_attempts=attempts, budget_seconds=seconds, out_path=out, report_path=rep,
        )
        with open(out, "rb") as fh:
            data = fh.read()
        colors = None
        try:
            colors = json.load(open(rep)).get("lever_setting", {}).get("colors")
        except Exception:  # noqa: BLE001
            pass
        return data, res.output_fps, colors
    finally:
        shutil.rmtree(td, ignore_errors=True)


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

def gif_encode(fitted, delays, *, budget, max_colors=256, fps_cap=24, notes=None):
    """Return ``(bytes, "GIF", n_frames, fps)`` for the fitted frames under ``budget``."""
    if _enabled():
        try:
            data, n, fps, colors = _run_fovea(fitted, delays, int(budget))
            if notes is not None:
                notes.append(_fovea_note(len(fitted), n, colors))
            return data, "GIF", n, fps
        except Exception as exc:  # noqa: BLE001 - never let Fovea break the pipeline
            log.warning("fovea.bridge_failed; falling back to legacy: %s", str(exc)[:200])
            if notes is not None:
                notes.append("Fovea encode failed; used the standard encoder.")
    from .encode import encode_gif as legacy

    return legacy(fitted, delays, budget=budget, max_colors=max_colors, fps_cap=fps_cap)


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


def gif_encode_compare(fitted, delays, *, budget, max_colors=256, fps_cap=24, notes=None):
    """Encode with BOTH Fovea (auto-balanced) and the legacy encoder for a side-by-side.

    Returns ``(fovea_bytes, "GIF", n_frames, fps, comparison, legacy_bytes)``. Each
    side's perceptual distance is measured against the source subsampled to that
    side's frame count, so the comparison is fair when frame counts differ.
    """
    from encoder.metrics import default_metric

    from .encode import encode_gif as legacy_encode

    td = tempfile.mkdtemp(prefix="fovea_cmp_")
    try:
        fovea_data, fovea_n, fovea_fps, fovea_colors = _run_fovea(fitted, delays, int(budget))
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
        return fovea_data, "GIF", fovea_n, fovea_fps, comparison, legacy_data
    finally:
        shutil.rmtree(td, ignore_errors=True)
