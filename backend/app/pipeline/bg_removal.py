"""Background removal via rembg (ONNX Runtime) — free, self-hosted, no API keys.

Sessions are cached per model and reused (loaded once, kept warm). Multi-frame
inputs are matted in parallel since ONNX Runtime releases the GIL during inference.
``rembg`` is imported lazily so the rest of the app runs even if it isn't installed.
"""
from __future__ import annotations

import importlib.util
from typing import Callable

import numpy as np
from PIL import Image

from ..observability import get_logger

log = get_logger("bg_removal")

DEFAULT_MODEL = "birefnet-general"
ANIME_MODEL = "isnet-anime"
# Used when something heavier (birefnet) is requested for multi-frame input:
# good quality, but far lighter on memory so it won't OOM the container.
SAFE_ANIM_MODEL = "isnet-general-use"

_sessions: dict[str, object] = {}
_remove_fn = None


def available() -> bool:
    return importlib.util.find_spec("rembg") is not None


def _load():
    global _remove_fn
    if _remove_fn is None:
        from rembg import new_session, remove  # lazy import

        _remove_fn = (new_session, remove)
    return _remove_fn


def _session(model: str):
    if model not in _sessions:
        new_session, _ = _load()
        log.info("bg.load_model", model=model)
        _sessions[model] = new_session(model)
    return _sessions[model]


def _looks_like_illustration(arr: np.ndarray) -> bool:
    """Cheap heuristic: flat-color art (anime/illustration) has few unique colors."""
    small = arr[::8, ::8, :3].reshape(-1, 3)
    if small.size == 0:
        return False
    quantized = (small // 16).astype(np.uint32)
    keys = quantized[:, 0] * 4096 + quantized[:, 1] * 64 + quantized[:, 2]
    unique_ratio = len(np.unique(keys)) / max(len(keys), 1)
    return unique_ratio < 0.18


def pick_model(requested: str, frames: list[np.ndarray]) -> str:
    multi = len(frames) > 1
    if requested and requested != "auto":
        # birefnet is too memory-heavy to run per-frame on a small instance.
        if multi and requested.startswith("birefnet"):
            return SAFE_ANIM_MODEL
        return requested
    illustration = bool(frames) and _looks_like_illustration(frames[0])
    if multi:
        # Animated: favor speed/memory. Anime art still benefits from isnet-anime;
        # everything else uses the fast u2net (birefnet is too slow per-frame).
        return ANIME_MODEL if illustration else "u2net"
    # Single still: one inference, so use the highest-quality model.
    return ANIME_MODEL if illustration else DEFAULT_MODEL


def warmup(models: list[str] | None = None) -> None:
    """Pre-load sessions at startup so the first request isn't slow."""
    if not available():
        log.warning("bg.unavailable")
        return
    for model in models or [DEFAULT_MODEL]:
        try:
            _session(model)
        except Exception:  # noqa: BLE001
            log.error("bg.warmup_failed", model=model, exc_info=True)


def remove_bg(
    frames: list[np.ndarray],
    model: str,
    progress: Callable[[int], None] | None = None,
) -> list[np.ndarray]:
    """Matte frames serially. ONNX Runtime already parallelizes a single
    inference across all cores, so running frames serially avoids the CPU
    oversubscription (and memory blow-up) of N concurrent inferences."""
    _, remove = _load()
    session = _session(model)

    results: list[np.ndarray] = []
    for i, arr in enumerate(frames):
        out = remove(Image.fromarray(arr, "RGBA"), session=session)
        results.append(np.asarray(out.convert("RGBA"), dtype=np.uint8))
        if progress:
            progress(i + 1)
    return results
