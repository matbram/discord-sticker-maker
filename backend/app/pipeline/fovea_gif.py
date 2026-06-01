"""Bridge: encode a GIF output through the Fovea encoder, with a safe fallback.

The orchestrator produces GIF outputs (emoji + gif) by calling ``encode_gif``.
This module routes those through Fovea's ``encode()`` — which hits the byte
budget while keeping every frame and judging quality perceptually — and falls
back to the legacy ffmpeg path if Fovea is disabled, unavailable, or errors.

Toggle with ``USE_FOVEA_GIF`` (default on). Per-encode time budget via
``FOVEA_BUDGET_SECONDS`` (default 25s); the search is anytime, so it returns the
best result found within the budget.
"""
from __future__ import annotations

import logging
import os
import shutil
import tempfile

log = logging.getLogger("fovea.bridge")


def _enabled() -> bool:
    return os.getenv("USE_FOVEA_GIF", "1").lower() not in ("0", "false", "no")


def gif_encode(fitted, delays, *, budget, max_colors=256, fps_cap=24, notes=None):
    """Return ``(bytes, "GIF", n_frames, fps)`` for the fitted frames under ``budget``.

    Tries Fovea first (keeping all frames); falls back to the legacy ffmpeg
    palette encoder on any failure so a single bad encode never breaks a job.
    """
    if _enabled():
        try:
            return _fovea_encode(fitted, delays, int(budget), notes)
        except Exception as exc:  # noqa: BLE001 - never let Fovea break the pipeline
            log.warning("fovea.bridge_failed; falling back to legacy: %s", str(exc)[:200])
            if notes is not None:
                notes.append("Fovea encode failed; used the standard encoder.")
    # Lazy import keeps this module importable without the backend's heavier deps.
    from .encode import encode_gif as legacy

    return legacy(fitted, delays, budget=budget, max_colors=max_colors, fps_cap=fps_cap)


def _fovea_encode(fitted, delays, budget, notes):
    from encoder import encode as fovea_encode

    td = tempfile.mkdtemp(prefix="fovea_gif_")
    out = os.path.join(td, "o.gif")
    try:
        res = fovea_encode(
            list(fitted), target_bytes=budget, mode="cap", delays_ms=list(delays),
            max_attempts=int(os.getenv("FOVEA_MAX_ATTEMPTS", "16")),
            budget_seconds=float(os.getenv("FOVEA_BUDGET_SECONDS", "25")),
            out_path=out,
        )
        with open(out, "rb") as fh:
            data = fh.read()
    finally:
        shutil.rmtree(td, ignore_errors=True)
    if notes is not None:
        if res.perceptually_lossless:
            notes.append("Fovea: visually identical to the source at this size.")
        else:
            notes.append("Fovea: slight visible softening to fit the size limit.")
    return data, "GIF", len(fitted), res.output_fps
