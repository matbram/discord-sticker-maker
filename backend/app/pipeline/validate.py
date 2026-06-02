"""Build the Discord upload checklist per output type."""
from __future__ import annotations

from ..models import DISCORD_MAX_BYTES, STICKER_SIZE


def build_checklist(data: bytes, width: int, height: int, fmt: str, has_alpha: bool) -> dict:
    """Sticker checklist (back-compat)."""
    return {
        "dimensions_320": width == STICKER_SIZE and height == STICKER_SIZE,
        "under_512kb": len(data) <= DISCORD_MAX_BYTES,
        "format_ok": fmt in ("PNG", "APNG"),
        "transparent": has_alpha,
    }


def _kb(n: int) -> int:
    return (n + 1023) // 1024


def build_checklist_for(output_type: str, data: bytes, width: int, height: int,
                        fmt: str, has_alpha: bool, profile: dict) -> dict:
    size = len(data)
    if output_type == "sticker":
        return {
            f"{profile['size']}×{profile['size']} px": width == profile["size"] and height == profile["size"],
            "Under 512 KB": size <= DISCORD_MAX_BYTES,
            "PNG / APNG": fmt in ("PNG", "APNG"),
            "Transparent background": has_alpha,
        }
    if output_type == "emoji":
        return {
            f"{profile['size']}×{profile['size']} px": width == profile["size"] and height == profile["size"],
            "Under 256 KB": size <= profile["hard_limit"],
            "PNG / GIF": fmt in ("PNG", "GIF"),
            "Transparent background": has_alpha,
        }
    # gif — dimensions are user-chosen (source resolution or an explicit W×H), so there's
    # no fixed size to assert; show the actual output dims and gate on format + byte limit.
    return {
        f"{width}×{height} px": True,
        f"Under {_kb(profile['budget']) // 1024} MB": size <= profile["hard_limit"],
        "GIF format": fmt == "GIF",
    }
