# Benchmark harness (M0)

The yardstick. Nothing counts as "better" until it beats the best existing tool
at equal size on the corpus. `bench/` produces a reproducible table of
achieved-size + perceptual score per clip per target across the baselines (and,
optionally, Fovea itself).

## Quick start

```bash
fovea-bench validate     # schema-check the manifest; report clip + binary status
fovea-bench list         # the clips x targets that would run
fovea-bench run          # baselines over the corpus -> bench/out/results.{csv,json}
fovea-bench run --engines ffmpeg-palette,gifski,gifsicle-lossy,fovea   # include Fovea
```

With a fresh checkout (no clips, no binaries) `run` skips every cell cleanly and
exits 0 — it does not error.

## Corpus = real clips only

No media is committed. `bench/corpus/manifest.yaml` describes the intended clips;
drop the actual files into `bench/corpus/clips/` (gitignored). See
`bench/corpus/README.md`. Every category must be represented:
`screen_recording`, `video_clip`, `motion_graphics`.

### Manifest schema

```yaml
version: 1
defaults:
  target_sizes: ["256KB", "512KB", "1MB", "2MB", "8MB"]
  max_fps: 50
  fps: null                       # null => source fps (capped at max_fps)
clips:
  - id: screen_terminal_scroll
    path: clips/screen_terminal_scroll.mp4    # relative to the corpus dir
    category: screen_recording                # screen_recording|video_clip|motion_graphics
    duration_s: 6.0
    note: "scrolling terminal; large flat regions"
    license: "self-recorded"
    target_sizes: ["256KB", "512KB"]          # optional per-clip override
```

## Output

- `bench/out/results.csv` — one row per `(clip, target, engine)` cell.
- `bench/out/results.json` — the same records plus a `meta` block (Fovea version,
  Python/platform, requested vs available engines).
- A human-readable summary to stdout, grouped by clip.

Key columns: `achieved_bytes`, `under_target`, `distance` (the judge scalar,
lower is better), `msssim`, `temporal`, `fps`, `lever_setting`, and
`skipped_reason` for any skipped cell.

## Reproducibility

`run` uses **count-based** budgets (`--max-attempts`), so results are reproducible
given the same clips, binaries, and tool versions. (The interactive `fovea
encode` path uses wall-clock budgets and is therefore not bit-reproducible across
machines — by design.)

## The measurement gate (spec §11)

A lever or technique is kept only if it measurably beats the best baseline at
equal size (or reaches equal perceived quality at smaller size) across the
corpus. Secondary reference metrics (SSIM, etc.) are reported for context only,
never optimized against.
