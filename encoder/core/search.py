"""The rate-targeting search: the heart of M1, reused by the benchmark harness.

Two pieces, both decoupled from real encoding via ``measure``/``score`` callbacks
so they are unit-testable without any external binary:

  * ``size_target_search`` — bisection over a monotone-ish lever ladder for the
    largest setting whose *measured* size fits the byte target. Tracks the best
    fitting and best overshoot so weak non-monotonicity can't return an overshoot
    when a real fit exists.
  * ``guided_search`` — the anytime constrained search: hit the target via the
    size search (descending the resolution lever only as a last resort), then a
    quality phase over secondary levers; ``cap`` picks the least-noticeable fit,
    ``invisible`` shrinks to the smallest size that stays under the perceptual
    threshold.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from ..metrics.base import DistanceResult
from .budget import Budget
from .levers import LeverState
from .sizes import Tolerance


@dataclass
class Candidate:
    idx: int
    size_bytes: int
    state: LeverState
    out_path: str
    scale: float = 1.0
    result: DistanceResult | None = None

    @property
    def distance(self) -> float | None:
        return self.result.distance if self.result is not None else None


@dataclass
class SearchResult:
    best_fit: Candidate | None       # largest size <= target among probed
    best_over: Candidate | None      # smallest size > target among probed
    in_window: bool = False

    @property
    def over_target(self) -> bool:
        return self.best_fit is None

    @property
    def chosen(self) -> Candidate | None:
        return self.best_fit or self.best_over


@dataclass
class SearchOutcome:
    chosen: Candidate | None
    over_target: bool
    scale: float
    attempts: int
    stopped_early: bool
    stop_reason: str | None


def _better_fit(a: Candidate | None, b: Candidate | None) -> Candidate | None:
    if a is None:
        return b
    if b is None:
        return a
    return a if a.size_bytes >= b.size_bytes else b   # prefer larger (closer to target)


def _better_over(a: Candidate | None, b: Candidate | None) -> Candidate | None:
    if a is None:
        return b
    if b is None:
        return a
    return a if a.size_bytes <= b.size_bytes else b   # prefer smaller overshoot


def size_target_search(
    measure: Callable[[int], Candidate],
    idx_range: tuple[int, int],
    target_bytes: int,
    tol: Tolerance,
    budget: Budget,
    cache: dict[int, Candidate] | None = None,
) -> SearchResult:
    """Bisection over ``[lo, hi]`` for the largest lever index whose size fits.

    ``measure`` runs the engine and returns a measured ``Candidate``; every fresh
    measurement ticks the budget. Returns the best fitting / best overshoot seen.
    """
    lo, hi = idx_range
    cache = cache if cache is not None else {}
    best_fit: Candidate | None = None
    best_over: Candidate | None = None
    in_window = False

    def probe(idx: int) -> Candidate | None:
        nonlocal best_fit, best_over
        if idx in cache:
            c = cache[idx]
        else:
            if budget.expired():
                return None
            c = measure(idx)
            budget.tick()
            cache[idx] = c
        if c.size_bytes <= target_bytes:
            best_fit = _better_fit(best_fit, c)
        else:
            best_over = _better_over(best_over, c)
        return c

    a, b = lo, hi
    while a <= b and not budget.expired():
        mid = (a + b) // 2
        c = probe(mid)
        if c is None:
            break
        if c.size_bytes <= target_bytes:
            if tol.in_window(c.size_bytes, target_bytes):
                in_window = True
                break
            a = mid + 1          # room to grow toward the target
        else:
            b = mid - 1          # too big; shrink
    return SearchResult(best_fit, best_over, in_window)


def guided_search(
    *,
    primary_n: int,
    scales: list[float],
    measure: Callable[[float, int], Candidate],
    score: Callable[[Candidate], float],
    target_bytes: int,
    tol: Tolerance,
    budget: Budget,
    mode: str = "cap",
    invisible_threshold: float = 0.005,
    explore: Callable[[Candidate, Budget], list[Candidate]] | None = None,
) -> SearchOutcome:
    """Anytime constrained search. See module docstring for the algorithm."""
    anchor: Candidate | None = None
    used_scale: float = scales[0]
    size_cache: dict[int, Candidate] = {}
    last: SearchResult | None = None

    # SIZE PHASE — least scaling that fits. scales[0] == 1.0, so the first scale
    # that yields a fit is the least-degraded; resolution drop is genuinely last.
    for scale in scales:
        if budget.expired():
            break
        cache: dict[int, Candidate] = {}
        last = size_target_search(
            lambda i, s=scale: measure(s, i), (0, primary_n - 1),
            target_bytes, tol, budget, cache,
        )
        if last.best_fit is not None:
            anchor, used_scale, size_cache = last.best_fit, scale, cache
            break

    if anchor is None:
        # Nothing fits even at the smallest resolution: return the smallest overshoot.
        chosen = last.best_over if last else None
        if chosen is not None:
            score(chosen)
        return SearchOutcome(
            chosen=chosen, over_target=True, scale=used_scale,
            attempts=budget.attempts, stopped_early=budget.stopped_early,
            stop_reason=budget.stop_reason or "exhausted",
        )

    # QUALITY PHASE — score the anchor and any fitting secondary-lever neighbours.
    score(anchor)
    best = anchor
    if explore is not None and not budget.expired():
        for c in explore(anchor, budget):
            if c.size_bytes <= target_bytes:
                score(c)
                if c.distance is not None and best.distance is not None and c.distance < best.distance:
                    best = c

    # MODE EXIT
    if mode == "invisible":
        best = _shrink_while_invisible(
            best, used_scale, primary_n, measure, score, target_bytes,
            budget, invisible_threshold, size_cache,
        )

    return SearchOutcome(
        chosen=best, over_target=False, scale=used_scale,
        attempts=budget.attempts, stopped_early=budget.stopped_early,
        stop_reason=budget.stop_reason or "converged",
    )


def _shrink_while_invisible(
    best: Candidate,
    scale: float,
    primary_n: int,
    measure: Callable[[float, int], Candidate],
    score: Callable[[Candidate], float],
    target_bytes: int,
    budget: Budget,
    threshold: float,
    size_cache: dict[int, Candidate],
) -> Candidate:
    """Walk the primary lever down while the candidate stays under ``threshold``."""
    if best.result is None:
        score(best)
    if best.distance is None or best.distance > threshold:
        return best        # cannot be made invisible at this size; report best quality
    cur = best
    idx = best.idx
    while idx - 1 >= 0 and not budget.expired():
        idx -= 1
        if idx in size_cache:
            c = size_cache[idx]
        else:
            c = measure(scale, idx)
            budget.tick()
            size_cache[idx] = c
        score(c)
        if c.size_bytes <= target_bytes and c.distance is not None and c.distance <= threshold:
            cur = c        # smaller and still invisible
        else:
            break          # crossed the threshold (or overshot) — stop
    return cur
