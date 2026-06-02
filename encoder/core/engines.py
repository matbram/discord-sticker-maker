"""GIF encoding engines — uniform wrappers over ffmpeg, gifski, and gifsicle.

Each engine turns a ``RenderContext`` (pre-written, possibly down-scaled PNG
frames) plus a ``LeverState`` into a GIF on disk and reports its *measured* size.
Frames are scaled once at write time, so engines never carry their own scale
logic. ``build_*_argv`` helpers construct the exact command lines and are unit
-tested without execution.

Engine roles (spec §7 + the Discord transparency reality):
  * ffmpeg-palette  — the workhorse; handles GIF 1-bit alpha (alpha_threshold).
  * gifski          — opaque video->GIF only (no partial alpha).
  * gifsicle-lossy  — post-processes an ffmpeg base GIF (lossy LZW); keeps alpha.
"""
from __future__ import annotations

import os
import shutil
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass

from PIL import Image

from . import ffmpeg
from .frames import Frames
from .levers import (
    FFMPEG_COLORS,
    GIFSICLE_LOSSY,
    GIFSKI_QUALITY,
    LeverKind,
    LeverState,
)
from .logging import get_logger
from .timing import effective_fps, ms_to_centiseconds

log = get_logger("engines")


@dataclass
class RenderContext:
    """Pre-written frame PNGs (already at output resolution) + timing."""

    frame_dir: str
    frame_paths: list[str]
    fps: float
    delays_cs: list[int]
    width: int
    height: int
    scale: float

    @property
    def n(self) -> int:
        return len(self.frame_paths)

    @property
    def fps_int(self) -> int:
        return max(1, int(round(self.fps)))

    @property
    def uniform_timing(self) -> bool:
        return len(set(self.delays_cs)) <= 1


@dataclass
class EngineOutput:
    path: str
    size_bytes: int
    argv: list[list[str]]
    engine: str
    state: LeverState


def _even(v: int) -> int:
    v = int(round(v))
    v -= v % 2
    return max(2, v)


def prepare_context(frames: Frames, scale: float, dest_dir: str) -> RenderContext:
    """Write (optionally down-scaled) RGBA PNG frames to ``dest_dir`` -> context."""
    if scale >= 1.0:
        w, h = frames.width, frames.height
    else:
        w, h = _even(frames.width * scale), _even(frames.height * scale)
    paths: list[str] = []
    for i, arr in enumerate(frames.frames):
        im = Image.fromarray(arr, "RGBA")
        if (im.width, im.height) != (w, h):
            im = im.resize((w, h), Image.LANCZOS)
        p = os.path.join(dest_dir, f"f{i:05d}.png")
        im.save(p, "PNG")
        paths.append(p)
    delays_cs = ms_to_centiseconds(frames.delays_ms) if any(frames.delays_ms) else [10] * frames.n
    fps = effective_fps(frames.delays_ms) or 10.0
    return RenderContext(dest_dir, paths, fps, delays_cs, w, h, scale)


# --------------------------------------------------------------------------- #
# Argv builders (pure string assembly — unit-tested without running anything)
# --------------------------------------------------------------------------- #

def _scale_note(width: int, height: int) -> None:
    return None


def build_ffmpeg_argv(ctx: RenderContext, state: LeverState, pal_path: str, out_path: str
                      ) -> list[list[str]]:
    colors = state.colors or 256
    dither = state.dither or "sierra2_4a"
    pattern = os.path.join(ctx.frame_dir, "f%05d.png")
    fr = f"{ctx.fps:.4f}"
    pass1 = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-start_number", "0", "-framerate", fr, "-i", pattern,
        "-vf", f"palettegen=max_colors={colors}:reserve_transparent=1",
        pal_path,
    ]
    use = f"paletteuse=dither={dither}:alpha_threshold=128"
    if dither == "bayer":
        use = f"paletteuse=dither=bayer:bayer_scale=3:alpha_threshold=128"
    pass2 = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-start_number", "0", "-framerate", fr, "-i", pattern, "-i", pal_path,
        "-lavfi", use, "-loop", "0", out_path,
    ]
    return [pass1, pass2]


def build_gifski_argv(ctx: RenderContext, state: LeverState, out_path: str) -> list[list[str]]:
    quality = state.quality if state.quality is not None else 90
    cmd = [
        "gifski", "--quality", str(quality), "--fps", str(ctx.fps_int),
        "--width", str(ctx.width), "--height", str(ctx.height),
        "-o", out_path, *ctx.frame_paths,
    ]
    return [cmd]


def build_gifsicle_argv(base_gif: str, state: LeverState, out_path: str) -> list[list[str]]:
    lossy = state.lossy if state.lossy is not None else 0
    cmd = ["gifsicle", "--optimize=3", f"--lossy={lossy}"]
    if state.colors is not None:
        cmd += ["--colors", str(state.colors)]
    if state.dither:
        cmd += ["--dither"]
    cmd += ["--no-warnings", "-o", out_path, base_gif]
    return [cmd]


# --------------------------------------------------------------------------- #
# Engine implementations
# --------------------------------------------------------------------------- #

class Engine(ABC):
    name: str
    primary_lever: LeverKind

    @classmethod
    @abstractmethod
    def available(cls) -> bool: ...

    @abstractmethod
    def supports_alpha(self) -> bool: ...

    @abstractmethod
    def default_state(self) -> LeverState: ...

    @abstractmethod
    def primary_values(self) -> tuple: ...

    @abstractmethod
    def state_for_primary(self, idx: int, base: LeverState | None = None) -> LeverState: ...

    def secondary_neighbors(self, state: LeverState) -> list[LeverState]:
        return []

    @abstractmethod
    def encode(self, ctx: RenderContext, state: LeverState, out_path: str) -> EngineOutput: ...


class FfmpegPaletteEngine(Engine):
    name = "ffmpeg-palette"
    primary_lever = LeverKind.COLORS

    @classmethod
    def available(cls) -> bool:
        return ffmpeg.have_ffmpeg()

    def supports_alpha(self) -> bool:
        return True

    def default_state(self) -> LeverState:
        return LeverState(colors=256, dither="sierra2_4a")

    def primary_values(self) -> tuple:
        return FFMPEG_COLORS

    def state_for_primary(self, idx: int, base: LeverState | None = None) -> LeverState:
        base = base or self.default_state()
        return base.with_(colors=FFMPEG_COLORS[idx])

    def secondary_neighbors(self, state: LeverState) -> list[LeverState]:
        # Cheaper dithers shrink the file; richer ones can raise quality. Try both.
        return [state.with_(dither=d) for d in ("bayer", "none", "floyd_steinberg")
                if d != state.dither]

    def encode(self, ctx: RenderContext, state: LeverState, out_path: str) -> EngineOutput:
        if not self.available():
            raise RuntimeError("ffmpeg not available")
        pal = os.path.join(ctx.frame_dir, "_pal.png")
        argv = build_ffmpeg_argv(ctx, state, pal, out_path)
        for cmd in argv:
            proc = ffmpeg.run(cmd)
            if proc.returncode != 0:
                raise RuntimeError(f"ffmpeg failed: {(proc.stderr or b'')[:300]!r}")
        size = os.path.getsize(out_path)
        return EngineOutput(out_path, size, argv, self.name, state)


class GifskiEngine(Engine):
    name = "gifski"
    primary_lever = LeverKind.QUALITY

    @classmethod
    def available(cls) -> bool:
        return shutil.which("gifski") is not None

    def supports_alpha(self) -> bool:
        return False

    def default_state(self) -> LeverState:
        return LeverState(quality=90)

    def primary_values(self) -> tuple:
        return GIFSKI_QUALITY

    def state_for_primary(self, idx: int, base: LeverState | None = None) -> LeverState:
        base = base or self.default_state()
        return base.with_(quality=GIFSKI_QUALITY[idx])

    def encode(self, ctx: RenderContext, state: LeverState, out_path: str) -> EngineOutput:
        if not self.available():
            raise RuntimeError("gifski not available")
        argv = build_gifski_argv(ctx, state, out_path)
        proc = ffmpeg.run(argv[0])
        if proc.returncode != 0:
            raise RuntimeError(f"gifski failed: {(proc.stderr or b'')[:300]!r}")
        size = os.path.getsize(out_path)
        return EngineOutput(out_path, size, argv, self.name, state)


class GifsicleLossyEngine(Engine):
    name = "gifsicle-lossy"
    primary_lever = LeverKind.LOSSY

    @classmethod
    def available(cls) -> bool:
        # Needs ffmpeg to synthesize the base GIF it then optimizes.
        return shutil.which("gifsicle") is not None and ffmpeg.have_ffmpeg()

    def supports_alpha(self) -> bool:
        return True

    def default_state(self) -> LeverState:
        return LeverState(lossy=0, colors=256)

    def primary_values(self) -> tuple:
        return GIFSICLE_LOSSY

    def state_for_primary(self, idx: int, base: LeverState | None = None) -> LeverState:
        base = base or self.default_state()
        return base.with_(lossy=GIFSICLE_LOSSY[idx])

    def secondary_neighbors(self, state: LeverState) -> list[LeverState]:
        return [state.with_(colors=c) for c in (128, 64) if c != state.colors]

    def encode(self, ctx: RenderContext, state: LeverState, out_path: str) -> EngineOutput:
        if not self.available():
            raise RuntimeError("gifsicle (or its ffmpeg base) not available")
        base = os.path.join(ctx.frame_dir, "_base.gif")
        base_out = FfmpegPaletteEngine().encode(
            ctx, LeverState(colors=256, dither="none"), base
        )
        argv = build_gifsicle_argv(base, state, out_path)
        proc = ffmpeg.run(argv[0])
        if proc.returncode != 0:
            raise RuntimeError(f"gifsicle failed: {(proc.stderr or b'')[:300]!r}")
        size = os.path.getsize(out_path)
        return EngineOutput(out_path, size, base_out.argv + argv, self.name, state)


def _native_dither(name: str | None) -> float:
    """Map the string dither lever to imagequant's 0..1 dithering level."""
    return {"none": 0.0, "bayer": 0.5}.get(name or "", 1.0)


def _native_dither_level(name: str | None) -> float:
    """Dither level for the native engine: from the lever ('none' -> 0, the default),
    overridable via FOVEA_NATIVE_DITHER for experiments."""
    env = os.getenv("FOVEA_NATIVE_DITHER")
    if env:
        try:
            return float(env)
        except ValueError:
            pass
    return _native_dither(name)


def _rgba_frames(ctx: RenderContext) -> list[bytes]:
    """Decode the context's PNG frames to raw RGBA bytes, cached on the context so
    the search's many same-frame probes don't re-decode."""
    cache = getattr(ctx, "_fovea_rgba", None)
    if cache is None:
        cache = [Image.open(p).convert("RGBA").tobytes() for p in ctx.frame_paths]
        ctx._fovea_rgba = cache  # type: ignore[attr-defined]
    return cache


class FoveaNativeEngine(Engine):
    """Native Rust encoder: per-frame local palettes + perceptual OKLab delta.

    This is the engine that dissolves the frames-vs-color frontier. Where the
    ffmpeg engine forces one global ≤256-color palette on every frame (the washout
    cause), this gives *each frame its own* perceptually-chosen palette (so the file
    can hold thousands of colors) and, on opaque clips, reuses unchanged pixels via
    transparency so static content costs ~0 bytes — reclaimed budget that funds both
    rich color and a full frame count. Falls back cleanly to the ffmpeg engines when
    the ``fovea_native`` extension isn't built.
    """

    name = "fovea-native"
    primary_lever = LeverKind.COLORS

    @classmethod
    def available(cls) -> bool:
        try:
            import fovea_native  # noqa: F401
        except Exception:
            return False
        return True

    def supports_alpha(self) -> bool:
        return True

    def default_state(self) -> LeverState:
        # dither "none": in a byte-budgeted per-frame-palette encoder, dithering spends
        # the budget on incompressible LZW noise that *starves* the palette (→ banding),
        # so NO dithering fits far more real colors and bands LESS. Measured on a smooth
        # gradient at 512px/512KB: dither→8 colors (band 0.037) vs none→224 colors
        # (band 0.005). The "always dither GIFs" rule assumes one global palette; it's
        # inverted here. (See STATE.md §0.)
        return LeverState(colors=256, dither="none")

    def primary_values(self) -> tuple:
        # Same per-frame color ladder; "colors" is now a *per-frame* budget.
        return FFMPEG_COLORS

    def state_for_primary(self, idx: int, base: LeverState | None = None) -> LeverState:
        base = base or self.default_state()
        return base.with_(colors=FFMPEG_COLORS[idx])

    def secondary_neighbors(self, state: LeverState) -> list[LeverState]:
        # A no-dither neighbor trades a little banding for a smaller file.
        return [state.with_(dither="none")] if state.dither != "none" else []

    def encode(self, ctx: RenderContext, state: LeverState, out_path: str) -> EngineOutput:
        import fovea_native

        frames = _rgba_frames(ctx)
        colors = int(state.colors or 256)
        dith = _native_dither_level(state.dither)
        delta = float(os.getenv("FOVEA_DELTA_E", "0.02"))  # OKLab ΔE; ~1 JND
        speed = int(os.getenv("FOVEA_NATIVE_SPEED", "5"))
        res = fovea_native.encode(
            frames, ctx.width, ctx.height, list(ctx.delays_cs),
            max_colors=colors, dithering=dith, delta_threshold=delta,
            speed=speed, loop_count=0,
        )
        with open(out_path, "wb") as fh:
            fh.write(res["gif"])
        argv = [[
            "fovea-native", f"colors={colors}", f"dither={state.dither}",
            f"delta_e={delta}", f"mode={res['mode']}", f"distinct={res['distinct_colors']}",
        ]]
        return EngineOutput(out_path, len(res["gif"]), argv, self.name, state)

    def search_to_budget(
        self, ctx: RenderContext, target_bytes: int, out_path: str, *, dither: str = "none",
    ) -> EngineOutput:
        """Hit the byte target in ONE Rust call: bisect the per-frame color budget to
        the largest palette that fits, frames quantized in parallel. Always returns a
        result (the smallest it can if the target is impossible), so the search can
        never time out into an over-budget file."""
        import fovea_native

        frames = _rgba_frames(ctx)
        dith_level = _native_dither_level(dither)
        delta = float(os.getenv("FOVEA_DELTA_E", "0.02"))
        speed = int(os.getenv("FOVEA_NATIVE_SPEED", "5"))
        res = fovea_native.search(
            frames, ctx.width, ctx.height, list(ctx.delays_cs), int(max(1, target_bytes)),
            max_colors=256, min_colors=2, dithering=dith_level, delta_threshold=delta,
            speed=speed, loop_count=0,
        )
        with open(out_path, "wb") as fh:
            fh.write(res["gif"])
        state = LeverState(colors=int(res["colors"]), dither=dither).with_(scale=ctx.scale)
        argv = [[
            "fovea-native-search", f"colors={res['colors']}", f"mode={res['mode']}",
            f"distinct={res['distinct_colors']}", f"under_budget={res['under_budget']}",
        ]]
        return EngineOutput(out_path, len(res["gif"]), argv, self.name, state)


ALL_ENGINES: tuple[type[Engine], ...] = (
    FoveaNativeEngine, FfmpegPaletteEngine, GifskiEngine, GifsicleLossyEngine,
)


def available_engines(names: list[str] | None = None) -> list[Engine]:
    """Instantiate the available engines, optionally filtered by name."""
    out: list[Engine] = []
    for cls in ALL_ENGINES:
        if names is not None and cls.name not in names:
            continue
        if cls.available():
            out.append(cls())
    return out


def get_engine(name: str) -> Engine:
    for cls in ALL_ENGINES:
        if cls.name == name:
            return cls()
    raise ValueError(f"unknown engine: {name!r}")
