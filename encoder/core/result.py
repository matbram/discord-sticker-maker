"""Result and honesty-report data shapes returned by ``encode``."""
from __future__ import annotations

from dataclasses import dataclass, field

from pydantic import BaseModel, Field


def _fovea_version() -> str:
    from .. import __version__

    return __version__


@dataclass
class EncodeResult:
    """The library return value (spec §13.5)."""

    path: str
    size_bytes: int
    perceptually_lossless: bool
    output_fps: float | None
    notes: list[str] = field(default_factory=list)


class LossLocus(BaseModel):
    """Where any visible loss landed — populated only when not lossless."""

    worst_frame: int
    worst_frame_distance: float
    region_hint: str | None = None


class EncodeReport(BaseModel):
    """The JSON honesty report written alongside the GIF."""

    input_path: str
    input_kind: str                 # "gif" | "image" | "video" | "frames"
    mode: str                       # "invisible" | "cap"
    target_bytes: int | None
    achieved_bytes: int
    under_target: bool
    perceptually_lossless: bool
    perceptual_distance: float
    metric_name: str
    invisible_threshold: float
    output_fps: float | None
    n_frames: int
    duration_ms: int
    engine_used: str
    lever_setting: dict = Field(default_factory=dict)
    loss_locus: LossLocus | None = None
    stopped_early: bool = False
    stop_reason: str | None = None
    attempts: int = 0
    elapsed_ms: float = 0.0
    tool_versions: dict = Field(default_factory=dict)
    fovea_version: str = Field(default_factory=_fovea_version)
    warnings: list[str] = Field(default_factory=list)
