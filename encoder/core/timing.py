"""Map source frame timing onto the GIF centisecond (1/100 s) delay grid.

GIF stores each frame's display time in centiseconds. Naively rounding, e.g.,
24 fps (4.1667 cs/frame) down to 4 cs accumulates a ~4% speed-up over a clip, so
we carry the rounding remainder forward (a Bresenham-style accumulator) and the
cumulative duration tracks the source to within a single centisecond.

Players historically clamp very short delays, so we floor each delay at
``MIN_DELAY_CS`` (= 2 cs, GIF's practical ~50 fps ceiling). Flooring inflates the
duration of sub-2cs frames — an unavoidable format limit, surfaced in the report.
"""
from __future__ import annotations

MIN_DELAY_CS = 2          # GIF practical ceiling ~= 50 fps; players clamp below this
DEFAULT_DELAY_CS = 10     # fallback when a source gives no usable delay
CLAMP_WARN_FPS = 20.0     # above this effective fps, many players still clamp playback


def _diffuse(per_frame_cs: list[float], min_cs: int) -> list[int]:
    """Round a list of fractional centisecond delays, carrying the remainder."""
    out: list[int] = []
    carry = 0.0
    for exact in per_frame_cs:
        v = exact + carry
        cs = int(round(v))
        carry = v - cs               # carry the rounding error to the next frame
        out.append(max(min_cs, cs))  # floor (does not feed back into carry by design)
    return out


def ms_to_centiseconds(delays_ms: list[int], *, min_cs: int = MIN_DELAY_CS) -> list[int]:
    """Convert per-frame millisecond delays to centiseconds, duration-preserving."""
    return _diffuse([max(0, int(d)) / 10.0 for d in delays_ms], min_cs)


def fps_to_centiseconds(fps: float, n: int, *, min_cs: int = MIN_DELAY_CS) -> list[int]:
    """Constant-fps source -> ``n`` per-frame centisecond delays, duration-preserving."""
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")
    if n <= 0:
        return []
    per = 100.0 / fps
    return _diffuse([per] * n, min_cs)


def effective_fps(delays_ms: list[int]) -> float | None:
    """Mean frames-per-second implied by a list of millisecond delays."""
    if not delays_ms:
        return None
    mean_ms = sum(delays_ms) / len(delays_ms)
    return round(1000.0 / mean_ms, 2) if mean_ms > 0 else None


def centiseconds_to_ms(delays_cs: list[int]) -> list[int]:
    """Inverse helper for reporting: cs grid back to milliseconds."""
    return [int(c) * 10 for c in delays_cs]
