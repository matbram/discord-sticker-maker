"""Build the labeled (reference, candidate-pair) dataset for the learned judge.

No humans: every pair has a *derivable* ordering. Two label sources:
  * Lever sweep (real GIF artifacts via the existing ffmpeg engine): more colors =>
    closer; and at a fixed low palette, DITHERED => closer than BANDED (the exact
    case MS-SSIM gets backwards — this bakes in the anti-banding prior).
  * Parametric degradations with exact severity ``s`` (banding/flicker/choppiness/
    blur): monotone chains give reliable "lower s is closer" pairs.

We store small sampled uint8 proxy frame stacks per variant (compact, train-ready)
and a ``pairs.jsonl`` referencing them; features are computed at train time so
train/inference use the identical ``judge_features`` code.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import tempfile

import numpy as np
from PIL import Image

from encoder.core.engines import FfmpegPaletteEngine, prepare_context
from encoder.core.frames import InputCaps, frames_from_list, frames_from_source, load_gif
from encoder.core.levers import LeverState

from ..judge_features import PROXY, T_SAMPLES, to_proxy_stack
from .degrade import FAMILIES
from .synth_clips import synth_clips

SEVERITIES = [0.12, 0.28, 0.45, 0.65, 0.9]
LEVER_VARIANTS = [(256, "sierra2_4a"), (32, "sierra2_4a"), (32, "none"),
                  (16, "sierra2_4a"), (16, "none")]
CLIP_EXTS = (".mp4", ".gif", ".webm", ".mkv", ".mov")
WORK_MAX_SIDE = 320


def _downscale(frames: list[np.ndarray], max_side: int) -> list[np.ndarray]:
    h, w = frames[0].shape[:2]
    if max(h, w) <= max_side:
        return frames
    sc = max_side / max(h, w)
    nw, nh = int(round(w * sc)), int(round(h * sc))
    return [np.asarray(Image.fromarray(f, "RGBA").resize((nw, nh), Image.LANCZOS), np.uint8)
            for f in frames]


def _clip_category(path: str) -> str:
    base = os.path.basename(path).lower()
    for cat in ("screen_recording", "video_clip", "motion_graphics"):
        if cat in base:
            return cat
    for key, cat in (("screen", "screen_recording"), ("video", "video_clip"),
                     ("motion", "motion_graphics")):
        if key in base:
            return cat
    return "video_clip"


def build(clips_dir: str, out_dir: str) -> dict:
    ref_dir, var_dir = os.path.join(out_dir, "ref"), os.path.join(out_dir, "var")
    os.makedirs(ref_dir, exist_ok=True)
    os.makedirs(var_dir, exist_ok=True)

    clips = sorted(p for p in glob.glob(os.path.join(clips_dir, "*"))
                   if os.path.splitext(p)[1].lower() in CLIP_EXTS)
    if not clips:
        print("no clips found; synthesizing a fallback corpus…")
        synth_clips(clips_dir)
        clips = sorted(p for p in glob.glob(os.path.join(clips_dir, "*"))
                       if os.path.splitext(p)[1].lower() in CLIP_EXTS)

    caps = InputCaps(max_pixels=1280 * 720, max_frames=120, max_duration_s=60)
    pairs: list[dict] = []
    clip_meta: dict[str, str] = {}

    for path in clips:
        clip = os.path.splitext(os.path.basename(path))[0]
        cat = _clip_category(path)
        try:
            fr = frames_from_source(path, max_fps=15, max_duration_s=4.0, caps=caps)
        except Exception as exc:  # noqa: BLE001
            print(f"skip {clip}: decode failed ({exc})")
            continue
        frames = _downscale(fr.frames, WORK_MAX_SIDE)
        delays = (fr.delays_ms[:len(frames)] if len(fr.delays_ms) >= len(frames)
                  else [80] * len(frames))
        if len(frames) < 2:
            print(f"skip {clip}: too few frames")
            continue

        np.save(os.path.join(ref_dir, f"{clip}.npy"), to_proxy_stack(frames))
        clip_meta[clip] = cat

        def add(a: str, b: str, margin: float, family: str) -> None:
            pairs.append({"clip": clip, "a": a, "b": b, "label": 0,
                          "margin": round(float(margin), 3), "family": family, "category": cat})

        # --- parametric degradations: monotone chains (ref is the s=0 anchor) ---
        for fam, fn in FAMILIES.items():
            pts: list[tuple[str, float]] = [("__ref__", 0.0)]
            for s in SEVERITIES:
                vid = f"deg_{fam}_{int(round(s * 100)):03d}"
                np.save(os.path.join(var_dir, f"{clip}__{vid}.npy"), to_proxy_stack(fn(frames, s)))
                pts.append((vid, s))
            for i in range(len(pts)):
                for j in range(i + 1, len(pts)):
                    add(pts[i][0], pts[j][0], pts[j][1] - pts[i][1], fam)

        # --- lever sweep: real GIF artifacts via the production ffmpeg engine ---
        td = tempfile.mkdtemp(prefix="ds_lev_")
        try:
            ctx = prepare_context(frames_from_list(frames, delays), 1.0, td)
            eng = FfmpegPaletteEngine()
            ok: dict[tuple[int, str], str] = {}
            for colors, dither in LEVER_VARIANTS:
                vid = f"lev_c{colors}_{dither}"
                out = os.path.join(td, f"{vid}.gif")
                eng.encode(ctx, LeverState(colors=colors, dither=dither), out)
                np.save(os.path.join(var_dir, f"{clip}__{vid}.npy"), to_proxy_stack(load_gif(out).frames))
                ok[(colors, dither)] = vid

            def lv(c: int, d: str) -> str:
                return ok[(c, d)]

            for cd in LEVER_VARIANTS:                       # ref closer than any lever variant
                add("__ref__", lv(*cd), 0.5, "lever")
            add(lv(256, "sierra2_4a"), lv(32, "sierra2_4a"), 0.40, "lever")   # more colors closer
            add(lv(256, "sierra2_4a"), lv(16, "sierra2_4a"), 0.50, "lever")
            add(lv(32, "sierra2_4a"), lv(16, "sierra2_4a"), 0.35, "lever")
            # DITHERED closer than BANDED at equal low palette — the MS-SSIM failure case
            add(lv(32, "sierra2_4a"), lv(32, "none"), 0.70, "dither_vs_band")
            add(lv(16, "sierra2_4a"), lv(16, "none"), 0.70, "dither_vs_band")
        except Exception as exc:  # noqa: BLE001
            print(f"lever variants failed for {clip}: {exc}")
        finally:
            shutil.rmtree(td, ignore_errors=True)
        print(f"clip {clip} [{cat}]: {len(frames)} frames -> {len(pairs)} pairs so far")

    with open(os.path.join(out_dir, "pairs.jsonl"), "w") as fh:
        for p in pairs:
            fh.write(json.dumps(p) + "\n")
    meta = {"clips": clip_meta, "severities": SEVERITIES,
            "lever_variants": [list(x) for x in LEVER_VARIANTS],
            "proxy": PROXY, "t_samples": T_SAMPLES,
            "n_pairs": len(pairs), "n_clips": len(clip_meta)}
    with open(os.path.join(out_dir, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"\ndataset: {len(pairs)} pairs over {len(clip_meta)} clips -> {out_dir}")
    return meta


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the learned-judge training dataset.")
    ap.add_argument("--clips-dir", default="bench/corpus/train_clips")
    ap.add_argument("--out", default="bench/corpus/dataset")
    args = ap.parse_args()
    build(args.clips_dir, args.out)


if __name__ == "__main__":
    main()
