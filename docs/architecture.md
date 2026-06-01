# Architecture

Fovea is a standalone package at the repo root, independent of the existing
`backend/` service (which it will eventually replace, milestone M4). It does not
import any `backend/app/pipeline` code.

```
encoder/                 # the encoder core + CLI (the IP)
  core/
    frames.py            # Frames + frame I/O (video/gif/image -> RGBA frames); no subsampling
    ffmpeg.py            # ffmpeg/ffprobe discovery, run wrapper, source probing
    timing.py            # ms <-> centisecond grid (error-diffused), delay clamping
    engines.py           # RenderContext + Engine wrappers (ffmpeg/gifski/gifsicle) + argv builders
    levers.py            # LeverState + per-engine ladders (index up => bigger file)
    sizes.py             # parse_size_str, platform presets, Tolerance
    budget.py            # anytime Budget (deadline / attempt cap)
    search.py            # size_target_search + guided_search  <-- the heart of M1
    encode.py            # encode(): orchestration, modes, report assembly
    result.py            # EncodeResult + EncodeReport + LossLocus
    logging.py           # stdlib-based structured logger
  metrics/
    base.py              # Metric ABC + DistanceResult
    msssim.py            # pure-numpy MS-SSIM
    temporal.py          # flicker / temporal-consistency term
    perceptual.py        # composite default judge + invisible_threshold
    external.py          # optional ssimulacra2/butteraugli adapters (hook for M2)
  cli/main.py            # `fovea encode ...`
bench/                   # M0 benchmark harness
  manifest.py records.py runners.py run.py cli.py  corpus/
docs/  tests/
```

## Data flow (encode)

```
source (path | frames)
   │  frames.frames_from_source         (every frame kept)
   ▼
Frames ──► engines.prepare_context(scale)  ──► RenderContext (PNG frames @ output res)
   │
   ▼  search.guided_search(measure, score, budget, mode)
   ├─ SIZE PHASE: size_target_search over the engine's primary lever
   │      measure(scale, idx) = engine.encode(...)  -> REAL measured bytes
   ├─ (resolution lever descends only if nothing fits)
   ├─ QUALITY PHASE: score fitting candidates with the perceptual Metric
   └─ MODE EXIT: cap -> min distance; invisible -> shrink while under threshold
   ▼
chosen Candidate ──► copy out OUT.gif + assemble EncodeReport (honesty report)
```

Two invariants from the spec are enforced here:

- **Real measured size.** `measure` actually runs the engine and reads bytes off
  disk; the search never estimates size.
- **Metric stays out of the byte loop.** The size search compares bytes only; the
  (expensive) perceptual metric runs solely on fitting candidates.

## Key seams / extension points

- **`metrics.Metric`** — the perceptual judge. The M2 learned, motion-aware metric
  registers in `metrics/__init__.py::default_metric` behind this interface; the
  rest of the system is unchanged.
- **`engines.Engine`** — a GIF engine. M3's native internals (perceptual frame
  reuse, region-local palettes, joint RD-LZW, a Rust core) arrive as new `Engine`
  implementations with their own lever ladders; `guided_search` is reused as-is.
- **`search.guided_search`** — decoupled from real encoding via `measure`/`score`
  callbacks, so it is exercised by both `encode()` and the bench, and unit-tested
  with mocks (no binaries).
- **`encode()`** — the integration seam for the M4 service/worker.

## Determinism & anytime behavior

- The selection rule is deterministic given the set of evaluated candidates
  (fixed ladders, fixed neighbor order, tie-break preferring the smaller file).
- The interactive `encode()` path uses a wall-clock + attempt `Budget` and returns
  best-so-far on expiry (`stop_reason` in the report).
- The **bench** uses count-based budgets (`--max-attempts`) so its table is
  reproducible across machines (tool versions captured in `results.json` meta).
