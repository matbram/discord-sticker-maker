# Benchmark corpus

This directory defines the Fovea benchmark corpus. **No media is committed** —
the corpus is *real clips only*. `manifest.yaml` lists the intended clips; you
supply the actual files.

## Adding clips

1. Create the (gitignored) clips directory and drop your media in, matching the
   `path:` values in `manifest.yaml`:

   ```
   bench/corpus/clips/screen_terminal_scroll.mp4
   bench/corpus/clips/talking_head.mp4
   bench/corpus/clips/logo_motion_graphic.gif
   ...
   ```

2. Confirm they're detected and that the baseline engines are installed:

   ```
   fovea-bench validate
   ```

3. Run the benchmark (writes `bench/out/results.csv` and `results.json`):

   ```
   fovea-bench run                      # baselines only
   fovea-bench run --engines ffmpeg-palette,gifski,gifsicle-lossy,fovea
   ```

## Categories

Every category must be represented — they behave very differently:

| category           | character                                              |
|--------------------|--------------------------------------------------------|
| `screen_recording` | large flat regions, partial motion (Fovea's best case) |
| `video_clip`       | footage; mostly-still background + moving subject       |
| `motion_graphics`  | flat colors, sharp edges, often full-frame motion       |

## Licensing

Record the source and license of every clip in the manifest's `license:` field
before publishing any results. Self-recorded screen captures are simplest.

## Reproducibility

`bench run` uses count-based budgets (`--max-attempts`), so the table is
reproducible given the same clips, binaries, and tool versions (captured in the
`meta` block of `results.json`).
