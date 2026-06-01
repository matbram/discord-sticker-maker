# CLI & library reference

## `fovea encode`

```
fovea encode INPUT (--target-size 8MB | --platform discord) [--mode cap|invisible]
    [--fps N] [--max-fps 50] [--tolerance 5%]
    [--budget-seconds 30] [--max-attempts 24]
    [--metric auto|msssim] [--engines ffmpeg-palette,gifski,gifsicle-lossy]
    -o OUT.gif [--report OUT.gif.json] [-v]
```

`INPUT` is a video or GIF (or any image ffmpeg/Pillow can read). Emits `OUT.gif`,
a JSON report sidecar, and a one-line summary:

```
OUT.gif  188.5KB  [perceptually lossless]  14.9 fps
report: OUT.gif.json
```

- `--mode cap` (default): hit a hard byte target; least-noticeable fit if it can't
  be invisible. `--mode invisible`: smallest size that stays perceptually lossless
  (`--target-size`, if given, is only an upper ceiling).
- `--target-size` or `--platform` is required for `cap`. Presets: `discord`
  (512KB), `discord-emoji` (256KB), `slack` (128KB), `telegram` (512KB).
- `--budget-seconds` / `--max-attempts`: the anytime budget. On expiry the best
  result so far is returned and the report notes `stopped_early` + `stop_reason`.

### `--target-size` units (important)

Units are **binary** and the target is **never exceeded**:

| input         | bytes            |
|---------------|------------------|
| `512KB`       | 512 × 1024       |
| `8MB`         | 8 × 1024²        |
| `1MiB`        | 1024²            |
| `1024` / `1024B` | 1024          |

`KB`=1024 (not 1000) is deliberate: platform caps like Discord's "512KB" are
1024-based, and the conservative interpretation can never overshoot a hard cap.

## Library API

```python
from encoder import encode   # -> EncodeResult

res = encode(
    source,                  # path (video/gif) OR a list of HxWx4 RGBA arrays
    target_bytes,            # int bytes, or None for invisible-with-no-ceiling
    mode="cap",              # "cap" | "invisible"
    *,
    delays_ms=None,          # required if source is a list of frames
    fps=None, max_fps=50.0,
    platform=None,           # resolves a preset when target_bytes is None
    tolerance=None,          # "5%" / 0.05 / Tolerance(...)
    budget_seconds=30.0, max_attempts=24,
    metric=None,             # default = metrics.default_metric()
    caps=None,               # InputCaps(max_file_bytes/pixels/frames/duration_s)
    engines=None,            # restrict to a subset of engine names
    out_path=None, report_path=None,
)

res.path                     # output GIF path
res.size_bytes               # achieved size
res.perceptually_lossless    # bool
res.output_fps               # effective fps after the centisecond-grid mapping
res.notes                    # human-readable caveats / warnings
```

The full `EncodeReport` (written to `report_path`) adds `under_target`,
`perceptual_distance`, `metric_name`, `invisible_threshold`, `engine_used`,
`lever_setting`, `loss_locus` (where any visible loss landed), `stopped_early`,
`stop_reason`, `attempts`, and `warnings`.

## Engines

External binaries the encoder drives (detected via `PATH`; missing ones are
skipped):

- **ffmpeg-palette** — the workhorse; handles GIF 1-bit alpha (transparent
  stickers). Primary lever: palette size.
- **gifsicle-lossy** — lossy LZW post-pass on an ffmpeg base GIF; keeps alpha.
- **gifski** — opaque video→GIF only (no partial alpha). Primary lever: quality.

Engine selection: transparent input prefers `ffmpeg-palette` → `gifsicle-lossy`;
opaque input prefers `ffmpeg-palette` → `gifski` → `gifsicle-lossy`.

## Exit codes

`0` success; `1` on a handled error (bad input, no engine available, etc.) with a
message on stderr.
