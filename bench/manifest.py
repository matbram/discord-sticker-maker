"""Benchmark corpus manifest schema + loading.

The corpus is *real clips only*: the committed manifest describes the intended
clips but no media is committed, so ``clip_present`` is False on a fresh checkout
and the harness skips cleanly until real clips are dropped into the corpus dir.
"""
from __future__ import annotations

import os
from typing import Literal

import yaml
from pydantic import BaseModel, Field

from encoder.core.sizes import parse_size_str

Category = Literal["screen_recording", "video_clip", "motion_graphics"]


class ClipEntry(BaseModel):
    id: str
    path: str                          # relative to the corpus dir (or absolute)
    category: Category
    duration_s: float | None = None
    note: str | None = None
    license: str | None = None
    target_sizes: list[str] | None = None   # overrides defaults when set


class ManifestDefaults(BaseModel):
    target_sizes: list[str] = Field(default_factory=lambda: ["256KB", "512KB", "1MB", "2MB", "8MB"])
    max_fps: float = 50.0
    fps: float | None = None


class Manifest(BaseModel):
    version: int = 1
    defaults: ManifestDefaults = Field(default_factory=ManifestDefaults)
    clips: list[ClipEntry] = Field(default_factory=list)


def load_manifest(path: str) -> Manifest:
    with open(path) as fh:
        data = yaml.safe_load(fh) or {}
    return Manifest.model_validate(data)


def resolve_clip_path(clip: ClipEntry, corpus_dir: str) -> str:
    if os.path.isabs(clip.path):
        return clip.path
    return os.path.join(corpus_dir, clip.path)


def clip_present(clip: ClipEntry, corpus_dir: str) -> bool:
    return os.path.exists(resolve_clip_path(clip, corpus_dir))


def resolved_targets(clip: ClipEntry, manifest: Manifest) -> list[int]:
    raw = clip.target_sizes or manifest.defaults.target_sizes
    return [parse_size_str(s) for s in raw]
