"""Synthesize small ffmpeg ``lavfi`` clips mimicking the three corpus categories.

Deterministic and license-clean (ffmpeg-generated, no third-party rights). Used to
guarantee category coverage and as the fallback when real GitHub-hosted clips
can't be fetched (the network policy blocks general video CDNs). Mirrors the
``testsrc`` pattern in ``tests/test_integration_smoke.py``.
"""
from __future__ import annotations

import os

from encoder.core import ffmpeg as _ff

# (id, category, lavfi filtergraph). A moving element keeps motion non-trivial; the
# screen_recording proxy uses smooth gradients (banding-prone, large flat regions).
def _graphs() -> list[tuple[str, str, str]]:
    g: list[tuple[str, str, str]] = []
    # screen_recording: smooth animated gradient + a moving box (flat regions band hard)
    grads = [("0x0a1230", "0x6da0e0", 0.05, 40), ("0x301020", "0xe0c060", 0.08, 55),
             ("0x06301a", "0x80e0c0", 0.04, 70)]
    for i, (c0, c1, sp, vx) in enumerate(grads):
        g.append((f"synth_screen_{i}", "screen_recording",
                  f"gradients=s=320x180:c0={c0}:c1={c1}:speed={sp}:duration=4,"
                  f"drawbox=x=t*{vx}:y={70 + i * 25}:w=46:h=24:color=white@1.0:t=fill,format=yuv420p"))
    # video_clip: continuous-tone mandelbrot zooms (textured, banding-sensitive)
    for i, ss in enumerate((3.0, 2.2, 1.4)):
        g.append((f"synth_video_{i}", "video_clip",
                  f"mandelbrot=s=320x180:rate=15:start_scale={ss}:end_scale=0.3,format=yuv420p"))
    # motion_graphics: flat colors + sharp edges, full-frame motion
    lifes = [(0.12, "0x24d089"), (0.18, "0xe05050"), (0.08, "0x5080ff")]
    for i, (ratio, col) in enumerate(lifes):
        g.append((f"synth_motion_{i}", "motion_graphics",
                  f"life=s=320x180:rate=15:mold=8:r=3:ratio={ratio}:"
                  f"death_color=0x101418:life_color={col},format=yuv420p"))
    return g


SYNTH = _graphs()


def synth_clips(out_dir: str, *, seconds: int = 4, fps: int = 15) -> list[tuple[str, str, str]]:
    """Write the synthetic clips into ``out_dir`` -> list of (id, category, path)."""
    os.makedirs(out_dir, exist_ok=True)
    ff = _ff.ffmpeg_path()
    made: list[tuple[str, str, str]] = []
    for cid, cat, graph in SYNTH:
        out = os.path.join(out_dir, f"{cid}.mp4")
        cmd = [ff, "-y", "-hide_banner", "-loglevel", "error",
               "-f", "lavfi", "-i", graph, "-t", str(seconds), "-r", str(fps), out]
        proc = _ff.run(cmd)
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg synth failed for {cid}: {(proc.stderr or b'')[:300]!r}")
        made.append((cid, cat, out))
    return made


if __name__ == "__main__":
    import sys

    d = sys.argv[1] if len(sys.argv) > 1 else "bench/corpus/train_clips"
    for cid, cat, path in synth_clips(d):
        print(f"synth {cid:24s} [{cat}] -> {path}")
