from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

DISCORD_MAX_BYTES = 512 * 1024
TARGET_BYTES = 500 * 1024
STICKER_SIZE = 320
# Max frames an animated sticker keeps. Inter-frame (apngasm) compression lets us
# keep more frames under 512KB, so this is generous; encode trims further if needed.
MAX_ANIM_FRAMES = 72


class FitMode(str, Enum):
    fit = "fit"    # show the whole image / subject, transparent-pad to square
    fill = "fill"  # cover the square, cropping the long edge


class BgModel(str, Enum):
    auto = "auto"
    birefnet_general = "birefnet-general"
    isnet_anime = "isnet-anime"
    birefnet_portrait = "birefnet-portrait"
    u2net = "u2net"


class Priority(str, Enum):
    # How to spend the byte budget for animated output:
    #   smooth   -> keep the most frames, drop colors hard
    #   balanced -> keep frames but stop mid-way before trimming frames
    #   sharp    -> keep colors high, drop frames instead
    smooth = "smooth"
    balanced = "balanced"
    sharp = "sharp"


class OutputType(str, Enum):
    sticker = "sticker"
    emoji = "emoji"
    gif = "gif"


class GifQuality(str, Enum):
    small = "small"
    balanced = "balanced"
    high = "high"


class GifAspect(str, Enum):
    # Shape of the GIF canvas. Sticker/emoji are always square; the GIF can match the
    # square framing, keep the source's aspect, or be a 16:9 wide crop.
    square = "square"
    source = "source"
    wide = "16:9"


# Per-output Discord profiles. Square types resize to (size, size); gif keeps
# aspect within max_dim. budget = target bytes (hard Discord limit noted too).
GIF_PROFILES = {
    "small": {"max_dim": 240, "fps_cap": 15, "budget": 2 * 1024 * 1024},
    "balanced": {"max_dim": 360, "fps_cap": 20, "budget": 5 * 1024 * 1024},
    "high": {"max_dim": 480, "fps_cap": 24, "budget": 8 * 1024 * 1024},
}


def profile_for(output_type: str, gif_quality: str = "balanced") -> dict:
    if output_type == "sticker":
        # 320x320 under 512KB rarely fits >~25 frames, so a lower cap avoids
        # encoding frames we'd only trim away (big speedup, same output).
        return {"square": True, "size": 320, "animated_format": "APNG", "static_format": "PNG",
                "budget": TARGET_BYTES, "hard_limit": DISCORD_MAX_BYTES, "frame_cap": 48}
    if output_type == "emoji":
        return {"square": True, "size": 128, "animated_format": "GIF", "static_format": "PNG",
                "budget": 256 * 1024, "hard_limit": 256 * 1024, "frame_cap": 48}
    g = GIF_PROFILES.get(gif_quality, GIF_PROFILES["balanced"])
    return {"square": False, "max_dim": g["max_dim"], "animated_format": "GIF", "static_format": "GIF",
            "budget": g["budget"], "hard_limit": g["budget"], "fps_cap": g["fps_cap"], "frame_cap": 96}


def resolve_aspect(aspect, src_w: int, src_h: int) -> tuple[int, int]:
    """Map a GifAspect (or its string value) to a target (w, h) ratio. ``source``
    uses the source dims so the GIF keeps the input's shape; only the ratio matters."""
    a = aspect.value if hasattr(aspect, "value") else aspect
    if a == "square":
        return (1, 1)
    if a == "16:9":
        return (16, 9)
    return (max(1, int(src_w)), max(1, int(src_h)))


class OutputSpec(BaseModel):
    """One requested output. Per-output knobs fall back to ProcessParams defaults."""
    type: OutputType = OutputType.sticker
    priority: Optional[Priority] = None
    max_colors: Optional[int] = Field(None, ge=2, le=256)
    gif_quality: GifQuality = GifQuality.balanced
    aspect: GifAspect = GifAspect.source  # GIF-only; sticker/emoji ignore it (always square)
    # Per-output framing — falls back to the shared ProcessParams values when unset,
    # so panning/zooming one format never moves the others.
    zoom: Optional[float] = Field(None, ge=0.1, le=5.0)
    offset_x: Optional[float] = Field(None, ge=-1.0, le=1.0)
    offset_y: Optional[float] = Field(None, ge=-1.0, le=1.0)
    fit_mode: Optional[FitMode] = None


class ProcessParams(BaseModel):
    """Shared edit (source-level) + the list of outputs to produce."""

    remove_bg: bool = False
    bg_model: BgModel = BgModel.auto

    # crop / fit
    auto_crop: bool = True
    fit_mode: FitMode = FitMode.fit
    zoom: float = Field(1.0, ge=0.1, le=5.0)
    offset_x: float = Field(0.0, ge=-1.0, le=1.0)
    offset_y: float = Field(0.0, ge=-1.0, le=1.0)
    padding: float = Field(0.06, ge=0.0, le=0.5)  # fraction of subject size

    # animation (shared)
    max_fps: int = Field(18, ge=1, le=60)
    max_duration_s: float = Field(4.0, ge=0.1, le=30.0)
    trim_start_s: float = Field(0.0, ge=0.0)
    priority: Priority = Priority.balanced

    # encoding defaults (per-output may override)
    max_bytes: int = Field(TARGET_BYTES, ge=10 * 1024, le=DISCORD_MAX_BYTES)
    max_colors: int = Field(256, ge=2, le=256)

    # which outputs to make (default: one sticker, backward-compatible)
    outputs: list[OutputSpec] = Field(default_factory=lambda: [OutputSpec()])


class StickerMeta(BaseModel):
    output_type: str = "sticker"
    width: int
    height: int
    bytes: int
    frames: int
    fps: Optional[float] = None
    requested_fps: Optional[float] = None
    animated: bool
    format: str  # "PNG" | "APNG" | "GIF"
    under_limit: bool
    checklist: dict
    notes: list[str] = Field(default_factory=list)
