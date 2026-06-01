"""Search tests against deterministic mock encoders/metrics — no binaries."""
from __future__ import annotations

from encoder.core.budget import Budget
from encoder.core.levers import LeverState
from encoder.core.search import Candidate, guided_search, size_target_search
from encoder.core.sizes import Tolerance
from encoder.metrics.base import DistanceResult

KB = 1024


def _cand(idx, size, scale=1.0):
    return Candidate(idx=idx, size_bytes=size, state=LeverState(colors=idx),
                     out_path=f"/x/{scale}_{idx}.gif", scale=scale)


def test_size_search_picks_largest_fitting():
    sizes = [50, 120, 260, 540, 1100]  # KB, monotone increasing
    measure = lambda i: _cand(i, sizes[i] * KB)
    sr = size_target_search(measure, (0, 4), 600 * KB, Tolerance(0.05), Budget(max_attempts=100))
    assert sr.best_fit.idx == 3                       # 540KB is the largest <= 600KB
    assert not sr.over_target


def test_size_search_smaller_target():
    sizes = [50, 120, 260, 540, 1100]
    measure = lambda i: _cand(i, sizes[i] * KB)
    sr = size_target_search(measure, (0, 4), 300 * KB, Tolerance(0.05), Budget(max_attempts=100))
    assert sr.best_fit.idx == 2                       # 260KB


def test_size_search_all_overshoot():
    sizes = [50, 120, 260, 540, 1100]
    measure = lambda i: _cand(i, sizes[i] * KB)
    sr = size_target_search(measure, (0, 4), 40 * KB, Tolerance(0.05), Budget(max_attempts=100))
    assert sr.over_target
    assert sr.best_over.idx == 0                       # smallest overshoot


def test_size_search_handles_non_monotone_wiggle():
    sizes = [50, 120, 115, 540, 1100]  # idx2 dips below idx1
    measure = lambda i: _cand(i, sizes[i] * KB)
    sr = size_target_search(measure, (0, 4), 600 * KB, Tolerance(0.05), Budget(max_attempts=100))
    assert sr.best_fit.size_bytes <= 600 * KB
    assert sr.best_fit.idx == 3                        # never returns an overshoot


def test_budget_caps_measurements():
    calls = {"n": 0}

    def measure(i):
        calls["n"] += 1
        return _cand(i, (i + 1) * 100 * KB)

    budget = Budget(max_attempts=2)
    size_target_search(measure, (0, 9), 600 * KB, Tolerance(0.05), budget)
    assert calls["n"] <= 2
    assert budget.stop_reason == "max_attempts"


# --- guided_search ---------------------------------------------------------- #

SIZES = {0: 100, 1: 200, 2: 300, 3: 500, 4: 900}  # KB; index up => bigger file


def _make_callbacks(distances):
    def measure(scale, idx):
        return _cand(idx, SIZES[idx] * KB, scale)

    def score(cand):
        d = distances[cand.idx]
        cand.result = DistanceResult(d, [d], d, 0.0, 0)
        return d

    return measure, score


def test_guided_cap_picks_least_noticeable_fit():
    # larger size => lower distance (realistic); fitting idx are 0..3 (<=600KB)
    distances = {0: 0.05, 1: 0.03, 2: 0.02, 3: 0.008, 4: 0.001}
    measure, score = _make_callbacks(distances)
    out = guided_search(
        primary_n=5, scales=[1.0], measure=measure, score=score,
        target_bytes=600 * KB, tol=Tolerance(0.05), budget=Budget(max_attempts=50),
        mode="cap", invisible_threshold=0.005,
    )
    assert not out.over_target
    assert out.chosen.idx == 3                          # 500KB, lowest distance that fits
    assert out.stop_reason == "converged"


def test_guided_invisible_shrinks_to_smallest_lossless():
    distances = {0: 0.05, 1: 0.02, 2: 0.009, 3: 0.005, 4: 0.001}
    measure, score = _make_callbacks(distances)
    out = guided_search(
        primary_n=5, scales=[1.0], measure=measure, score=score,
        target_bytes=600 * KB, tol=Tolerance(0.05), budget=Budget(max_attempts=50),
        mode="invisible", invisible_threshold=0.01,
    )
    # idx2 (300KB, 0.009) is the smallest size still under threshold 0.01
    assert out.chosen.idx == 2


def test_guided_no_fit_returns_smallest_overshoot():
    distances = {i: 0.01 for i in range(5)}
    measure, score = _make_callbacks(distances)
    out = guided_search(
        primary_n=5, scales=[1.0], measure=measure, score=score,
        target_bytes=50 * KB, tol=Tolerance(0.05), budget=Budget(max_attempts=50),
        mode="cap", invisible_threshold=0.005,
    )
    assert out.over_target
    assert out.chosen.idx == 0                           # smallest possible


def test_guided_is_deterministic():
    distances = {0: 0.05, 1: 0.03, 2: 0.02, 3: 0.008, 4: 0.001}
    results = []
    for _ in range(3):
        measure, score = _make_callbacks(distances)
        out = guided_search(
            primary_n=5, scales=[1.0], measure=measure, score=score,
            target_bytes=600 * KB, tol=Tolerance(0.05), budget=Budget(max_attempts=50),
            mode="cap", invisible_threshold=0.005,
        )
        results.append(out.chosen.idx)
    assert results == [results[0]] * 3
