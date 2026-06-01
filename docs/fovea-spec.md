# Fovea — Perceptually-Lossless GIF Encoder

Fovea produces the best-looking `.gif` that fits under a **hard file-size cap**,
keeps **every frame**, and stays **invisible to the human eye**. It is the
encoder behind an "upload your video or GIF and we optimize it" tool.

This document is a condensed, implementation-facing summary of the full design.
The authoritative product spec (problem statement, prior-art audit, novel
contributions, roadmap, risks) lives in the project design doc; this file tracks
what the code actually implements and the principles it must honor.

## The problem

You have a good-looking GIF, a platform imposes a hard byte ceiling, and the
usual ways of shrinking it make it look bad. Fovea reframes this as a **budget
problem**: the platform hands you N bytes; spend every one as wisely as possible
so the result looks identical to the source while keeping motion smooth.

Formally: minimize perceptible distortion subject to `output_size ≤ target_size`,
with all frames preserved.

## Hard constraints (non-negotiable)

1. **Spec-compliant GIF.** Output plays anywhere a GIF plays.
2. **All frames, always.** Frame count and timing match the source. Size comes
   from palette / dither / lossy-LZW / frame-reuse / resolution levers — **never**
   from dropping frames. (This is the core difference from the legacy
   `backend/app/pipeline/encode.py`, which subsamples frames to fit.)
3. **Perceptually lossless is the bar**, not bit-exact.
4. **Honesty over silent failure.** If a clip cannot fit invisibly, say so and
   show where the cost landed.
5. **Optimize the real objective.** Decisions are scored against the *measured*
   compressed size and a model of human vision — never a cheap proxy.

## Two operating modes

- **`invisible`** — find the smallest size that stays perceptually lossless and
  report it. A `--target-size` (if given) acts only as an upper ceiling.
- **`cap`** — hit a hard byte target. If the clip fits invisibly, it does;
  otherwise produce the least-noticeable version that fits and report honestly.

## Core method (M1)

A single anytime search minimizes perceptual distortion subject to
`size ≤ target`. The spec's rate–distortion form `cost = distortion + λ·size` is
honored as the *ranking* function; target-hitting itself is a **measured-size
binary search** over each engine's primary lever (robust to the non-smooth
size↔lever steps that pure λ-bisection struggles with). See `architecture.md`.

In this milestone the levers are realized by driving best-in-class **external
engines** (ffmpeg palettegen/paletteuse, gifsicle `--lossy`, gifski). The
novelty is the joint *decision/search*, not a custom LZW. Later milestones (M3)
replace these with Fovea's own internals.

## The two laws we cannot break

- **The format ceiling.** 256 colors/frame, 1980s LZW, no real interframe coding.
  GIF will never match WebP/AVIF/video on ratio; that is not the goal.
- **Rate–distortion–perception (Blau & Michaeli 2019).** "Tiny file and zero
  perceptible loss" has a hard limit. Every clip has a size floor below which
  perceptual loss becomes visible; the honest report is how we keep that from
  being a surprise.

## Milestones

| # | What | Status in this repo |
|---|------|---------------------|
| M0 | Benchmark harness (the yardstick) | **implemented** (`bench/`) |
| M1 | Target-size auto-encoder (foundation) | **implemented** (`encoder/`) |
| M2 | Learned, motion-aware perceptual metric | future (registers in `encoder/metrics/`) |
| M3 | Novel internals: perceptual frame reuse, region-local palettes, joint RD-LZW | future |
| M4 | Service + UI wiring (everything runs on Fovea) | future (`encode()` is the seam) |
| M5 | Learned warm-start; spec-compliance hardening | future |

Until M2 lands, the perceptual judge is a reference metric (MS-SSIM + a temporal
flicker term); see `metrics.md`.
