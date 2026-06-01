"""Metric interface shared by the encoder's judge and the benchmark harness."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # avoid importing numpy/Pillow-heavy module at type-check time
    from ..core.frames import Frames


@dataclass
class DistanceResult:
    """A perceptual distance between a reference and a candidate animation.

    ``distance`` is a non-negative scalar where 0 means identical and larger is
    worse. ``per_frame`` holds the spatial distance of each frame so callers can
    report *where* loss landed; ``worst_frame`` indexes the max.
    """

    distance: float
    per_frame: list[float]
    spatial: float
    temporal: float
    worst_frame: int
    extra: dict[str, float] = field(default_factory=dict)


class Metric(ABC):
    """A perceptual judge. Lower distance = closer to the source."""

    name: str = "metric"
    invisible_threshold: float = 0.0   # distance <= this => "perceptually lossless"

    @abstractmethod
    def distance(self, reference: "Frames", candidate: "Frames") -> DistanceResult:
        ...
