"""Build the Discord upload checklist for the result."""
from __future__ import annotations

from ..models import DISCORD_MAX_BYTES, STICKER_SIZE


def build_checklist(data: bytes, width: int, height: int, fmt: str, has_alpha: bool) -> dict:
    return {
        "dimensions_320": width == STICKER_SIZE and height == STICKER_SIZE,
        "under_512kb": len(data) <= DISCORD_MAX_BYTES,
        "format_ok": fmt in ("PNG", "APNG"),
        "transparent": has_alpha,
    }
