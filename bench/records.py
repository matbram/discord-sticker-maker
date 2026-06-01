"""Benchmark result records + CSV/JSON writers + a human-readable summary table."""
from __future__ import annotations

import csv
import json
import platform
from pydantic import BaseModel, Field


class ResultRecord(BaseModel):
    clip_id: str
    category: str
    engine: str                       # "ffmpeg-palette" | "gifski" | "gifsicle-lossy" | "fovea"
    target_bytes: int
    achieved_bytes: int | None = None
    under_target: bool = False
    lever_setting: dict = Field(default_factory=dict)
    distance: float | None = None     # perceptual judge scalar (lower is better)
    msssim: float | None = None
    temporal: float | None = None
    worst_frame: int | None = None
    fps: float | None = None
    n_frames: int | None = None
    encode_ms: float | None = None
    attempts: int = 0
    stopped_early: bool = False
    skipped_reason: str | None = None  # "clip_missing" | "binary_missing:<n>" | "decode_error:..."
    metric_name: str = ""
    fovea_version: str = ""


_FIELDS = list(ResultRecord.model_fields.keys())


def env_meta() -> dict:
    from encoder import __version__

    return {
        "fovea_version": __version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
    }


def write_csv(records: list[ResultRecord], path: str) -> None:
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=_FIELDS)
        writer.writeheader()
        for r in records:
            row = r.model_dump()
            row["lever_setting"] = json.dumps(row["lever_setting"], separators=(",", ":"))
            writer.writerow(row)


def write_json(records: list[ResultRecord], path: str, meta: dict | None = None) -> None:
    payload = {"meta": meta or env_meta(), "records": [r.model_dump() for r in records]}
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2)


def read_json(path: str) -> list[ResultRecord]:
    with open(path) as fh:
        data = json.load(fh)
    return [ResultRecord.model_validate(r) for r in data["records"]]


def _fmt_bytes(n: int | None) -> str:
    if n is None:
        return "-"
    f = float(n)
    for unit in ("B", "KB", "MB"):
        if f < 1024 or unit == "MB":
            return f"{f:.1f}{unit}" if unit != "B" else f"{int(f)}B"
        f /= 1024
    return f"{f:.1f}MB"


def summary_table(records: list[ResultRecord]) -> str:
    """Group records by clip and render a fixed-width comparison table."""
    if not records:
        return "(no records)"
    by_clip: dict[str, list[ResultRecord]] = {}
    for r in records:
        by_clip.setdefault(r.clip_id, []).append(r)

    lines: list[str] = []
    header = f"  {'engine':<16}{'target':>9}{'achieved':>10}{'fit':>5}{'MS-SSIM':>9}{'dist':>8}{'fps':>7}"
    skips = 0
    for clip_id, rows in by_clip.items():
        cat = rows[0].category
        lines.append(f"\n{clip_id}  [{cat}]")
        lines.append(header)
        for r in sorted(rows, key=lambda x: (x.target_bytes, x.engine)):
            if r.skipped_reason:
                skips += 1
                lines.append(f"  {r.engine:<16}{_fmt_bytes(r.target_bytes):>9}"
                             f"{'  skipped: ' + r.skipped_reason}")
                continue
            fit = "yes" if r.under_target else "OVER"
            ms = f"{r.msssim:.3f}" if r.msssim is not None else "-"
            dist = f"{r.distance:.4f}" if r.distance is not None else "-"
            fps = f"{r.fps:.1f}" if r.fps is not None else "-"
            lines.append(
                f"  {r.engine:<16}{_fmt_bytes(r.target_bytes):>9}{_fmt_bytes(r.achieved_bytes):>10}"
                f"{fit:>5}{ms:>9}{dist:>8}{fps:>7}"
            )
    runnable = len(records) - skips
    lines.append(f"\n{runnable} runnable cell(s), {skips} skipped.")
    return "\n".join(lines)
