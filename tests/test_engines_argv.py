"""Engine command-line construction — asserted without running any binary."""
from __future__ import annotations

from encoder.core.engines import (
    FfmpegPaletteEngine,
    GifskiEngine,
    GifsicleLossyEngine,
    RenderContext,
    build_ffmpeg_argv,
    build_gifsicle_argv,
    build_gifski_argv,
)
from encoder.core.levers import FFMPEG_COLORS, GIFSICLE_LOSSY, LeverState


def _ctx(tmp_path) -> RenderContext:
    paths = [str(tmp_path / f"f{i:05d}.png") for i in range(3)]
    return RenderContext(str(tmp_path), paths, fps=24.0, delays_cs=[4, 4, 4],
                         width=64, height=48, scale=1.0)


def test_ffmpeg_two_pass_argv(tmp_path):
    ctx = _ctx(tmp_path)
    cmds = build_ffmpeg_argv(ctx, LeverState(colors=64, dither="floyd_steinberg"),
                             str(tmp_path / "pal.png"), str(tmp_path / "o.gif"))
    assert len(cmds) == 2
    p1, p2 = " ".join(cmds[0]), " ".join(cmds[1])
    assert cmds[0][0] == "ffmpeg" and cmds[1][0] == "ffmpeg"
    assert "palettegen=max_colors=64:reserve_transparent=1" in p1
    assert "paletteuse=dither=floyd_steinberg:alpha_threshold=128" in p2
    assert "-loop" in cmds[1] and cmds[1][cmds[1].index("-loop") + 1] == "0"


def test_ffmpeg_bayer_adds_scale(tmp_path):
    cmds = build_ffmpeg_argv(_ctx(tmp_path), LeverState(colors=32, dither="bayer"),
                             "pal.png", "o.gif")
    assert "dither=bayer:bayer_scale=3:alpha_threshold=128" in " ".join(cmds[1])


def test_gifski_argv_order(tmp_path):
    ctx = _ctx(tmp_path)
    cmd = build_gifski_argv(ctx, LeverState(quality=70), "out.gif")[0]
    assert cmd[0] == "gifski"
    assert cmd[cmd.index("--quality") + 1] == "70"
    assert cmd[cmd.index("--fps") + 1] == "24"
    assert cmd[cmd.index("-o") + 1] == "out.gif"
    assert cmd[-3:] == ctx.frame_paths           # frames trail the args, in order


def test_gifsicle_argv():
    cmd = build_gifsicle_argv("base.gif", LeverState(lossy=80, colors=128, dither="x"),
                              "out.gif")[0]
    s = " ".join(cmd)
    assert cmd[0] == "gifsicle"
    assert "--lossy=80" in s and "--optimize=3" in s
    assert cmd[cmd.index("--colors") + 1] == "128"
    assert "--dither" in cmd
    assert cmd[-1] == "base.gif"


def test_gifsicle_ladder_reversed():
    eng = GifsicleLossyEngine()
    assert eng.state_for_primary(0).lossy == 200                      # smallest file
    assert eng.state_for_primary(len(GIFSICLE_LOSSY) - 1).lossy == 0  # largest file


def test_ffmpeg_primary_is_increasing():
    eng = FfmpegPaletteEngine()
    vals = [eng.state_for_primary(i).colors for i in range(len(FFMPEG_COLORS))]
    assert vals == list(FFMPEG_COLORS)
    assert vals[0] < vals[-1]


def test_availability_reflects_binaries(monkeypatch):
    import encoder.core.engines as E

    monkeypatch.setattr(E.ffmpeg, "have_ffmpeg", lambda: True)
    monkeypatch.setattr(E.shutil, "which", lambda name: "/usr/bin/" + name)
    assert FfmpegPaletteEngine.available() is True
    assert GifskiEngine.available() is True
    assert GifsicleLossyEngine.available() is True

    monkeypatch.setattr(E.ffmpeg, "have_ffmpeg", lambda: False)
    assert FfmpegPaletteEngine.available() is False
    assert GifsicleLossyEngine.available() is False     # base needs ffmpeg
    assert GifskiEngine.available() is True             # gifski itself doesn't
