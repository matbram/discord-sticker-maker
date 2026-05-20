from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

DISCORD_MAX_BYTES = 512 * 1024
TARGET_BYTES = 500 * 1024
STICKER_SIZE = 320
# Hard cap on frames an animated sticker keeps. A 320x320 APNG under 512KB can't
# hold many frames anyway, so capping early makes every later stage far cheaper.
MAX_ANIM_FRAMES = 48


class BgModel(str, Enum):
    auto = "auto"
    birefnet_general = "birefnet-general"
    isnet_anime = "isnet-anime"
    birefnet_portrait = "birefnet-portrait"
    u2net = "u2net"


class Priority(str, Enum):
    # How to spend the 512 KB budget for animated stickers:
    #   smooth   -> keep the most frames, drop colors hard (down to 16)
    #   balanced -> keep frames but stop at 32 colors before trimming frames
    #   sharp    -> keep colors high, drop frames instead
    smooth = "smooth"
    balanced = "balanced"
    sharp = "sharp"


class ProcessParams(BaseModel):
    """Everything the pipeline needs. All optional — full-auto defaults."""

    remove_bg: bool = False
    bg_model: BgModel = BgModel.auto

    # crop / fit
    auto_crop: bool = True
    zoom: float = Field(1.0, ge=0.1, le=5.0)
    offset_x: float = Field(0.0, ge=-1.0, le=1.0)
    offset_y: float = Field(0.0, ge=-1.0, le=1.0)
    padding: float = Field(0.06, ge=0.0, le=0.5)  # fraction of subject size

    # animation
    max_fps: int = Field(18, ge=1, le=60)
    max_duration_s: float = Field(4.0, ge=0.1, le=30.0)
    trim_start_s: float = Field(0.0, ge=0.0)
    priority: Priority = Priority.balanced

    # encoding / size budget
    max_bytes: int = Field(TARGET_BYTES, ge=10 * 1024, le=DISCORD_MAX_BYTES)
    max_colors: int = Field(256, ge=2, le=256)


class StickerMeta(BaseModel):
    width: int
    height: int
    bytes: int
    frames: int
    fps: Optional[float] = None
    requested_fps: Optional[float] = None
    animated: bool
    format: str  # "PNG" | "APNG"
    under_limit: bool
    checklist: dict
    notes: list[str] = Field(default_factory=list)
