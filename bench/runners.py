"""Baseline runners: encode one clip at one target with one engine, then score it.

Each ``(clip, target, engine)`` cell decodes the clip once, runs the shared
``size_target_search`` over the engine's primary lever to land at/under the
target, scores the winner against the source with the judge, and returns one
``ResultRecord``. The ``fovea`` runner instead drives the full guided-search
encoder so the harness can compare it head-to-head with the baselines.
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
import time

from encoder.core.budget import Budget
from encoder.core.engines import Engine, prepare_context
from encoder.core.frames import frames_from_source, load_gif
from encoder.core.search import Candidate, size_target_search
from encoder.core.sizes import Tolerance
from encoder.core.timing import effective_fps
from encoder.metrics.base import Metric

from .manifest import ClipEntry, resolve_clip_path
from .records import ResultRecord


def _skip(clip: ClipEntry, target: int, engine_name: str, reason: str, metric_name: str,
          version: str) -> ResultRecord:
    return ResultRecord(
        clip_id=clip.id, category=clip.category, engine=engine_name, target_bytes=target,
        skipped_reason=reason, metric_name=metric_name, fovea_version=version,
    )


def run_clip_target(
    clip: ClipEntry, target_bytes: int, engine: Engine, judge: Metric, corpus_dir: str,
    *, max_attempts: int = 12, fps: float | None = None, max_fps: float = 50.0,
    tol: Tolerance | None = None, version: str = "",
) -> ResultRecord:
    tol = tol or Tolerance(0.05)
    path = resolve_clip_path(clip, corpus_dir)
    t0 = time.monotonic()
    try:
        frames = frames_from_source(path, fps=fps, max_fps=max_fps)
    except Exception as exc:  # noqa: BLE001
        return _skip(clip, target_bytes, engine.name, f"decode_error:{type(exc).__name__}",
                     judge.name, version)

    workdir = tempfile.mkdtemp(prefix="fbench_")
    budget = Budget(max_attempts=max_attempts)
    try:
        ctx = prepare_context(frames, 1.0, workdir)   # bench compares engines at full resolution

        def measure(idx: int) -> Candidate:
            state = engine.state_for_primary(idx)
            out = os.path.join(workdir, f"{engine.name}_{idx}.gif")
            eo = engine.encode(ctx, state, out)
            return Candidate(idx, eo.size_bytes, eo.state, out)

        sr = size_target_search(measure, (0, len(engine.primary_values()) - 1),
                                target_bytes, tol, budget)
        chosen = sr.chosen
        if chosen is None:
            return _skip(clip, target_bytes, engine.name, "no_output", judge.name, version)
        res = judge.distance(frames, load_gif(chosen.out_path))
        return ResultRecord(
            clip_id=clip.id, category=clip.category, engine=engine.name,
            target_bytes=target_bytes, achieved_bytes=chosen.size_bytes,
            under_target=chosen.size_bytes <= target_bytes,
            lever_setting=chosen.state.as_dict(),
            distance=round(res.distance, 6),
            msssim=round(res.extra.get("msssim_mean", 0.0), 6),
            temporal=round(res.temporal, 6),
            worst_frame=res.worst_frame,
            fps=effective_fps(frames.delays_ms),
            n_frames=frames.n,
            encode_ms=round((time.monotonic() - t0) * 1000, 1),
            attempts=budget.attempts, stopped_early=budget.stopped_early,
            metric_name=judge.name, fovea_version=version,
        )
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def run_clip_target_fovea(
    clip: ClipEntry, target_bytes: int, judge: Metric, corpus_dir: str,
    *, max_attempts: int = 24, fps: float | None = None, max_fps: float = 50.0,
    version: str = "",
) -> ResultRecord:
    """Run the full Fovea encoder on a cell, reading its report into a record."""
    from encoder.core.encode import encode

    path = resolve_clip_path(clip, corpus_dir)
    t0 = time.monotonic()
    workdir = tempfile.mkdtemp(prefix="fbench_fovea_")
    out = os.path.join(workdir, "out.gif")
    rep = out + ".json"
    try:
        encode(path, target_bytes, "cap", fps=fps, max_fps=max_fps, metric=judge,
               budget_seconds=120.0, max_attempts=max_attempts,
               out_path=out, report_path=rep)
        with open(rep) as fh:
            r = json.load(fh)
        return ResultRecord(
            clip_id=clip.id, category=clip.category, engine="fovea",
            target_bytes=target_bytes, achieved_bytes=r["achieved_bytes"],
            under_target=r["under_target"], lever_setting=r["lever_setting"],
            distance=r["perceptual_distance"], worst_frame=(r.get("loss_locus") or {}).get("worst_frame"),
            fps=r["output_fps"], n_frames=r["n_frames"],
            encode_ms=round((time.monotonic() - t0) * 1000, 1),
            attempts=r["attempts"], stopped_early=r["stopped_early"],
            metric_name=judge.name, fovea_version=version,
        )
    except Exception as exc:  # noqa: BLE001
        return _skip(clip, target_bytes, "fovea", f"encode_error:{type(exc).__name__}",
                     judge.name, version)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
