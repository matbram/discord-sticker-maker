"""`bench run` orchestration: iterate clips x targets x engines into a results table.

Count-based budgets (``max_attempts``) keep the table reproducible (spec §13.6).
Missing clips and missing engine binaries are recorded as skips rather than
errors, so a fresh checkout (no clips, no binaries) still produces a clean,
zero-exit summary.
"""
from __future__ import annotations

import os

from encoder.core.engines import ALL_ENGINES, get_engine
from encoder.metrics import get_metric

from .manifest import Manifest, clip_present, load_manifest, resolved_targets
from .records import ResultRecord, env_meta, summary_table, write_csv, write_json
from .runners import run_clip_target, run_clip_target_fovea

BASELINE_ENGINES = [c.name for c in ALL_ENGINES]   # ffmpeg-palette, gifski, gifsicle-lossy


def run_bench(
    manifest_path: str,
    corpus_dir: str,
    *,
    engine_names: list[str] | None = None,
    out_dir: str = "bench/out",
    max_attempts: int = 12,
    metric_name: str = "auto",
    fps: float | None = None,
) -> tuple[list[ResultRecord], dict]:
    manifest: Manifest = load_manifest(manifest_path)
    judge = get_metric(metric_name)
    version = env_meta()["fovea_version"]
    requested = engine_names or BASELINE_ENGINES

    # Resolve availability once so unavailable engines become skip rows (not errors).
    availability: dict[str, bool] = {}
    for name in requested:
        if name == "fovea":
            availability[name] = True
        else:
            try:
                availability[name] = get_engine(name).available()
            except ValueError:
                availability[name] = False

    records: list[ResultRecord] = []
    for clip in manifest.clips:
        present = clip_present(clip, corpus_dir)
        targets = resolved_targets(clip, manifest)
        for target in targets:
            for name in requested:
                if not present:
                    records.append(_skip_record(clip, target, name, "clip_missing",
                                                judge.name, version))
                    continue
                if not availability.get(name, False):
                    records.append(_skip_record(clip, target, name, f"binary_missing:{name}",
                                                judge.name, version))
                    continue
                if name == "fovea":
                    records.append(run_clip_target_fovea(
                        clip, target, judge, corpus_dir,
                        max_attempts=max(max_attempts, 24), fps=fps, version=version))
                else:
                    records.append(run_clip_target(
                        clip, target, get_engine(name), judge, corpus_dir,
                        max_attempts=max_attempts, fps=fps, version=version))

    meta = env_meta()
    meta["manifest"] = manifest_path
    meta["engines_requested"] = requested
    meta["engines_available"] = [n for n, ok in availability.items() if ok]
    os.makedirs(out_dir, exist_ok=True)
    write_csv(records, os.path.join(out_dir, "results.csv"))
    write_json(records, os.path.join(out_dir, "results.json"), meta)
    return records, meta


def _skip_record(clip, target, engine_name, reason, metric_name, version) -> ResultRecord:
    return ResultRecord(
        clip_id=clip.id, category=clip.category, engine=engine_name, target_bytes=target,
        skipped_reason=reason, metric_name=metric_name, fovea_version=version,
    )


def format_run_summary(records: list[ResultRecord], meta: dict) -> str:
    runnable = [r for r in records if not r.skipped_reason]
    if not runnable:
        hint = ("corpus empty — drop real clips into the corpus dir per the manifest, "
                "and install ffmpeg/gifsicle/gifski")
        return summary_table(records) + f"\n\n0 runnable cells. {hint}"
    return summary_table(records)
