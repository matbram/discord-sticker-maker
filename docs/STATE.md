# Fovea — Project State & Handoff (living document)

> **Read this first when resuming.** It is the canonical "where we are / how it
> works / what's next." Last updated: end of the build+integration session that
> wired Fovea into the live Discord-sticker-maker and tuned the GIF budget logic.
> Companion docs: `docs/fovea-spec.md` (the why), `docs/architecture.md`,
> `docs/metrics.md`, `docs/bench.md`, `docs/cli.md`.

---

## 1. TL;DR — current state

- **Fovea** is a perceptually-lossless GIF encoder: best-looking `.gif` under a
  hard byte cap. Built standalone at the repo root (`/encoder`, `/bench`), per
  spec milestones **M0 (benchmark harness)** + **M1 (target-size auto-encoder)**.
- It is **wired into the LIVE app** for the **GIF and emoji** outputs via
  `backend/app/pipeline/fovea_gif.py` (the "bridge"). The **sticker stays APNG**
  on the legacy encoder (Fovea is GIF-only). The legacy ffmpeg path is the
  automatic fallback if Fovea errors.
- Deployed on **Railway** from branch **`claude/fervent-noether-xaxjS`**
  (Dockerfile build; frontend built in-image). **Redeploy to pick up changes.**
- The most-iterated piece is the **GIF budget logic**: it fills the byte budget
  and offers a **Frames-vs-color** control (smooth / balanced / sharp).
- **Biggest known weakness / next big lever:** the perceptual metric is a
  placeholder (numpy MS-SSIM + temporal) that **under-values dithering/banding**.
  It cannot be trusted to judge color washout — which is why the GIF budget logic
  is heuristic, not metric-driven. The real fix is **M2 (a learned motion-aware
  metric)**.

---

## 2. Repository map

```
pyproject.toml                # dist "fovea"; deps numpy,Pillow,PyYAML,pydantic; scripts: fovea, fovea-bench
encoder/                      # THE IP (standalone; does NOT import backend)
  core/
    frames.py                 # Frames dataclass + video/gif/image -> RGBA frames (ffmpeg/Pillow). NO subsampling here.
    ffmpeg.py                 # ffmpeg/ffprobe discovery, run(), probe_source()
    timing.py                 # ms<->centisecond grid (error-diffused), MIN_DELAY_CS=2 (~50fps), CLAMP_WARN_FPS=20
    engines.py                # RenderContext + Engine ABC + FfmpegPaletteEngine/GifskiEngine/GifsicleLossyEngine + argv builders
    levers.py                 # LeverState + FFMPEG_COLORS (fine ladder), GIFSKI_QUALITY, GIFSICLE_LOSSY, SCALE_VALUES
    sizes.py                  # parse_size_str (binary KB=1024), PLATFORM_PRESETS, Tolerance
    budget.py                 # anytime Budget (deadline / max_attempts)
    search.py                 # size_target_search() + guided_search()  <-- heart of M1
    encode.py                 # encode(): orchestration, modes (cap/invisible), report assembly
    result.py                 # EncodeResult + EncodeReport + LossLocus (pydantic)
    logging.py                # stdlib-based structured logger (FOVEA_LOG_LEVEL)
  metrics/
    base.py msssim.py temporal.py perceptual.py external.py __init__.py
  cli/main.py                 # `fovea encode ...`
bench/                        # M0 harness: manifest.py records.py runners.py run.py cli.py corpus/(no clips committed)
tests/                        # 50 unit tests (no binaries) + test_integration_smoke.py (gated on ffmpeg/gifsicle)
docs/                         # specs + this file
backend/app/pipeline/fovea_gif.py   # *** the live integration bridge (backend) ***
backend/app/pipeline/orchestrator.py# threads budget/dims/priority into the GIF path
backend/app/models.py               # OutputSpec.max_bytes/max_dim/priority; StickerMeta.comparison
backend/app/main.py                 # audit.request / audit.store / audit.serve (+ SHA1)
frontend/src/App.svelte             # size/dimension/priority controls + Fovea-vs-standard comparison
```

---

## 3. Encoder internals (the vivid details)

### 3.1 `encode()` flow (`encoder/core/encode.py`)
`encode(source, target_bytes, mode, ...) -> EncodeResult`:
1. Resolve target (bytes, or platform preset; `invisible` may have no ceiling).
2. Decode to `Frames` (list of HxWx4 uint8 RGBA + per-frame `delays_ms`). **Every
   frame is kept** at this layer — Fovea never drops frames internally; any
   frame reduction in the live app happens in the *bridge*, deliberately.
3. Pick an engine (`_select_engine`): **ffmpeg-palette** is the workhorse (handles
   GIF 1-bit alpha via `alpha_threshold=128` — essential for transparent
   stickers). gifski is opaque-only (no partial alpha) → only for opaque video.
   gifsicle is a post-pass on an ffmpeg base GIF.
4. `guided_search` (see 3.4) drives the engine for **real measured bytes**, scores
   fitting candidates with the metric, returns the chosen `Candidate`.
5. Copy out `OUT.gif`, assemble `EncodeReport` (achieved bytes, `perceptually_lossless`,
   `output_fps`, `lever_setting`, `loss_locus`, `stop_reason`, `warnings`, `tool_versions`).

### 3.2 Engines (`encoder/core/engines.py`)
- Frames are **pre-scaled and written to PNGs once** into a `RenderContext`
  (`prepare_context`), so engines never carry their own scale logic.
- **`FfmpegPaletteEngine`** (primary lever = `colors`): two-pass
  `palettegen=max_colors=N:reserve_transparent=1` then
  `paletteuse=dither=D:alpha_threshold=128 -loop 0`. Default dither **sierra2_4a**
  (error diffusion). This is the engine used in production.
- **`GifskiEngine`** (lever = quality 1–100): opaque only.
- **`GifsicleLossyEngine`** (lever = lossy): synthesizes an ffmpeg base GIF then
  `gifsicle --optimize=3 --lossy=L`.
- `build_*_argv` are pure functions, unit-tested (`tests/test_engines_argv.py`)
  without executing anything.

### 3.3 Levers (`encoder/core/levers.py`)
- **`FFMPEG_COLORS`** is intentionally a **fine ladder**
  `(8,10,12,14,16,18,20,22,24,26,28,30,32,36,40,44,48,56,64,72,80,96,112,128,160,192,224,256)`.
  Reason: the size-vs-colors curve is **lumpy** (palette steps cause big size
  jumps). A fine ladder lets the size search land near the byte target instead of
  stalling a rung below it.
- `SCALE_VALUES` (resolution) is the **last-resort** lever (only when nothing else
  fits). Frame count is fixed by the caller.

### 3.4 The search (`encoder/core/search.py`)
- **`size_target_search(measure, idx_range, target, tol, budget)`** — bisection
  over a monotone-ish lever ladder for the largest setting that fits; tracks
  best-fit and best-overshoot (handles weak non-monotonicity); anytime.
- **`guided_search(...)`** — constrained: minimize perceptual distortion s.t.
  `size ≤ target`. `cap` picks min-distance among fitting; `invisible` shrinks to
  the smallest size under the perceptual threshold. The metric stays **out of the
  byte loop** (bytes-only in the size phase; metric only on fitting candidates).
- Decoupled from real encoding via `measure`/`score` callbacks → unit-tested with
  mocks (`tests/test_search.py`).

### 3.5 Perceptual metric (`encoder/metrics/`) — **load-bearing & the weak point**
- Default judge = **`PerceptualMetric`**: pure-numpy 5-scale **MS-SSIM** on luma
  (`msssim.py`) + a **temporal/flicker** term (`temporal.py`, penalizes
  inter-frame shimmer in regions the source held still). `distance = spatial +
  0.5*temporal`. `invisible_threshold = 0.005` (a **calibration target**, not a
  constant).
- **CRITICAL GOTCHA:** MS-SSIM **does not penalize color banding** and tends to
  rate a *smooth-but-banded* low-color frame as *closer* to the source than a
  *dithered* one (dithering adds high-frequency noise MS-SSIM dislikes). So:
  - We **always dither (sierra2_4a)** and never let the metric choose "no dither."
  - The **GIF budget/quality logic in the bridge is heuristic** (fill the budget,
    target color richness) **rather than letting the metric pick**, precisely
    because the metric can't be trusted on washout. The UI "match %" =
    `(1 - distance)` is MS-SSIM-based → **do not trust it for color quality**.
  - **This is the #1 reason M2 (a real perceptual metric) matters.**
- `external.py` will host optional ssimulacra2/butteraugli adapters; `default_metric()`
  falls back to MS-SSIM when none present (always the case today).

### 3.6 Timing grid (`encoder/core/timing.py`)
GIF stores per-frame delay in **centiseconds**. We map source ms onto the cs grid
with a Bresenham-style accumulator so cumulative duration tracks the source to
<1cs. Floor each delay at `MIN_DELAY_CS=2` (~50fps ceiling). `output_fps` in the
report reflects the emitted grid; we warn when effective fps > ~20 (players clamp).

---

## 4. The live integration — the GIF bridge (`backend/app/pipeline/fovea_gif.py`)

**This is where most of the session's iteration happened. Understand it well.**

### 4.1 How Fovea plugs in
`orchestrator.process()` produces sticker (APNG, legacy), **emoji (GIF)** and
**gif (GIF)**. For the two GIF outputs it calls the bridge:
- `gif_encode(fitted, delays, *, budget, max_colors, fps_cap, priority, notes)`
- `gif_encode_compare(...)` → also runs the legacy encoder for the side-by-side.
Both call **`_run_fovea(fitted, delays, budget, priority)`** and fall back to the
legacy `encode.encode_gif` on any exception. `fitted` is already cropped/fit RGBA
frames; the orchestrator pre-caps frames to the profile `frame_cap`.

### 4.2 The central reality: a rate–distortion–perception **frontier**
At a fixed size + dimensions, **(frames × colors) is a frontier** — more frames
means fewer colors. Concretely, the real test clip (IMG_7064, 180×320, 512KB):
- all 29 frames → only ~16–22 colors (**washed out**),
- 11 frames → 256 colors (rich, but choppy).
**You cannot have all-frames AND rich-color AND a filled budget at a tight size.**
This is the format ceiling, not a bug. The product answer is to (a) always fill
the budget and (b) let the user choose the frames-vs-color point.

### 4.3 `_run_fovea` — 3 phases (the algorithm)
For a given `priority` → a **color-richness floor** via `_color_floor_for`:
`smooth→0`, `balanced→64`, `sharp→160`.
1. **Encode all frames** at the richest palette the budget allows (one Fovea
   `encode`). Often this already fills the budget (just at low colors).
2. **Color-seeking trim** (skipped for `smooth`): drop frames ×0.72 per step until
   the palette reaches the mode's floor (or 256, or `MIN_FRAMES=6`). Fewer frames
   ⇒ a richer palette fits.
3. **Frame-fill** (`_fill_frames_at_colors`): once a palette is chosen, **add
   frames back at that fixed palette** until the byte budget is used (GIF size is
   ~linear in frame count at fixed colors). **This is what stops us leaving budget
   on the table** — the earlier bug was a one-directional trim that stalled at 79%.

Observed spectrum on the real clip @ 512KB (all fill the budget):
| mode | frames | colors | usage |
|---|---|---|---|
| smooth | 29/29 | 16 | 99% |
| balanced | 14/29 | 112 | 97% |
| sharp | 11/29 | 256 | 93% |
(`sharp` caps at 93% only because 256 colors is the GIF max — nothing more to add.)

### 4.4 The comparison + the frame-alignment fix
`gif_encode_compare` runs Fovea **and** legacy and returns a `comparison` dict
(bytes/frames/colors/distance per side, `perceptually_lossless`). **Fairness fix:**
each side's distance is measured against the **source subsampled to that side's
own frame count** (`_aligned_distance`) — otherwise a frame-trimmed candidate was
judged on misaligned frames and looked artificially worse. The baseline (legacy)
GIF is served as a benign extra output keyed `gif__cmp` (excluded from the zip).

### 4.5 Audit trail + SHA1 chain (how we prove what shipped)
`main.py` logs (structlog → JSON in Railway):
- `audit.request` (per-output max_bytes/max_dim/priority),
- `audit.output.budget` (resolved budget + dims),
- `audit.gif.compare` (`fovea_sha1`, `legacy_sha1`, frames, colors, `primary=fovea`),
- `audit.store` (each stored output: bytes, **sha1**, role),
- `audit.serve` (every preview/download: key, bytes, **sha1**, download flag).
`fovea_gif` logs `fovea.fill`/`fovea.framefill` per phase.
**Verification recipe:** the file a user downloads has the same SHA1 as
`audit.serve key=gif download=true`, which equals `audit.gif.compare fovea_sha1`
→ proves the download is Fovea (not the legacy `legacy_sha1`). Users can confirm
locally: `shasum -a 1 file.gif | cut -c1-12`.

---

## 5. Frontend controls (`frontend/src/App.svelte`)
Editor sidebar, per focused output:
- **Max file size** (chips 256KB…8MB + a freeform `800KB`/`3.5MB` input). Binary
  units. Defaults: sticker 512KB, emoji 256KB, gif 5MB.
- **Dimensions** (chips + custom px). Square side for sticker/emoji; longest edge
  for gif.
- **Frames vs. color** (gif): More frames / Balanced / Richer color → `priority`
  smooth/balanced/sharp (shared via `params.priority`).
- Choices **persist** to localStorage (`dsm_limits_v1`).
- **Fovea vs. standard** comparison card under the GIF: both animations, size /
  frames / colors / match%, per-side **Download Fovea / Download standard**, and a
  verdict. The main "Download GIF (Fovea)" button is labeled.
Plumbing: controls → `OutputSpec.max_bytes/max_dim/priority` → orchestrator
(`budget`, `out_size`/`out_max_dim`, `_v(eff.priority)`) → bridge.

---

## 6. Decisions locked (with rationale)
- **Standalone encoder** at repo root; **no reuse** of `backend/app/pipeline/*`
  (the bridge uses the legacy encoder only as a *fallback*). North star: the whole
  product eventually runs on Fovea.
- **M1 drives external engines** (ffmpeg/gifsicle/gifski) under the hood; the
  novelty is the joint decision/search. M3 replaces them with native internals.
- **Binary size units** (`KB`=1024) — never overshoot a platform cap.
- **All frames preserved in `encoder/`**; the live *bridge* may trade frames for
  color/budget **only because the user asked for that control** (frames-vs-color).
- **Reference metric (MS-SSIM)** is an explicit placeholder until M2.

---

## 7. Known issues / gotchas
- **Metric misjudges banding** (see 3.5) — the UI match% is not a reliable color
  quality signal. M2 fixes this.
- **Performance / no async worker yet (M4 not built):** encodes run synchronously
  in the in-process `PIPELINE_EXECUTOR`. With the budget-fill + comparison +
  3 outputs, a job can take **30–70s**. The client has a **120s SSE watchdog**.
  Per-iteration cap: `FOVEA_BUDGET_SECONDS=12`, `FOVEA_MAX_ATTEMPTS=12`, ≤5 trims.
  If jobs time out, the real fix is the §13.8 async split (M4).
- **gifski is not installed in the image** (no apt package). ffmpeg + gifsicle
  are (apt; we disabled two broken PPAs — deadsnakes/ondrej — to install them).
  Fovea works on ffmpeg alone, so gifski is optional.
- **Sticker = APNG = legacy encoder.** Fovea only touches GIF/emoji.
- **`priority` is shared** across outputs (`params.priority`), not per-output yet.
- **Local dev:** numpy/Pillow/etc. live in `backend/.venv` (gitignored) and the
  Fovea root `.venv`; the *system* python has none of them.

---

## 8. Roadmap / what's next

**Immediate / small:**
- Per-output `priority` (decouple GIF from sticker/emoji).
- Tune `_color_floor_for` thresholds and `FOVEA_BUDGET_USE` from real clips.
- Consider exposing the frames-vs-color choice as a labeled slider with live
  preview of the (frames, colors) it will pick.

**M2 — learned, motion-aware perceptual metric (the big one).**
Trains a metric that *does* penalize banding/flicker and agrees with human
preference. Plugs into `encoder/metrics/__init__.py::default_metric` behind the
existing `Metric` interface — **no other code changes**. This is what lets the
search optimize quality directly instead of via the budget-fill heuristic. Keep
the temporal dimension. Validate with small human pairwise studies (spec §9).

**M3 — native internals (replace external engines):**
- **Perceptual (sub-threshold) frame reuse** — carry a pixel over when the change
  is invisible (not just exact-match). Biggest win on partial-motion content.
- **Region-local palettes** — tile a frame, give regions their own ≤256-color
  table. **This is the lever that could break the frames-vs-color frontier**
  (more effective colors without dropping frames). Pay per-block overhead only
  when it wins.
- Joint RD-LZW, and a Rust core (PyO3) for the hot loops once proven.

**M4 — service/UI productionization (§13.8 async worker split):**
API (enqueue) + worker (encode) + Redis (queue/job state) + S3-compatible object
storage (R2/S3 via env). **Requires the user to provision** Redis + a bucket +
a second Railway service. Build with a graceful in-process fallback so a redeploy
never breaks. Needed because encodes are CPU-heavy and shouldn't hold a web worker.

**M5 — learned warm-start (predict good lever settings to cut encode time) +
spec-compliance hardening.**

**Benchmark (M0) is built but unused:** drop real clips into
`bench/corpus/clips/` (gitignored) and run `fovea-bench run` to start *measuring*
gains vs gifski/gifsicle/ffmpeg. Nothing should be claimed "better" without this.

---

## 9. Dev & deploy runbook (how to resume)

**Set up locally**
```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"            # numpy, Pillow, PyYAML, pydantic, pytest
pytest -q                          # 50 unit tests, no binaries needed
# media tools (Debian/Ubuntu): apt-get install -y ffmpeg gifsicle   (gifski optional via cargo/release binary)
# if apt update fails on PPAs: move /etc/apt/sources.list.d/*deadsnakes*,*ondrej* aside, then update
```

**Use it**
```bash
fovea encode IN.mp4 --target-size 512KB --mode cap -o out.gif      # + out.gif.json report
fovea-bench validate    # corpus/binary status
fovea-bench run         # results table (skips cleanly with no clips)
```

**Deploy (Railway)**
- Branch `claude/fervent-noether-xaxjS`, Dockerfile build, frontend built in-image.
- **Redeploy to pick up changes** (the running app does not hot-reload).
- Backend is a single synchronous service today (no worker yet).

**Env tunables (set on Railway, no code change)**
`USE_FOVEA_GIF` (1), `FOVEA_AUTOBALANCE` (1), `FOVEA_COMPARE` (1),
`FOVEA_BUDGET_USE` (0.93 — how full before stopping), `FOVEA_BUDGET_SECONDS` (12),
`FOVEA_MAX_ATTEMPTS` (12). Color-floor per mode is hardcoded in
`fovea_gif._color_floor_for` (smooth=0/balanced=64/sharp=160).

**Debugging a "looks wrong / wrong size" report:** ask for the `audit.*` log lines
for the request_id; the SHA1 chain (§4.5) + `fovea.fill`/`fovea.framefill` lines
tell you exactly which encoder ran, the (frames, colors, usage) it chose, and what
was served/downloaded.

---

## 10. Session journey (how we got here)
1. Built **M0 + M1** standalone (encoder + bench), 54 tests, installed ffmpeg/gifsicle.
2. User saw "no difference" on Railway → Fovea wasn't wired in. **Wired Fovea into
   the live GIF + emoji path** (bridge), image bundles the encoder.
3. Added **size + dimension controls** (the "where do I set 512KB?" ask) and the
   **Fovea-vs-standard comparison** UI.
4. "Am I downloading Fovea?" → added **audit logging + SHA1 chain**; proved
   download = Fovea from the user's own logs.
5. "Colors washed out / not using the budget" → finer color ladder → **auto-balance
   (color floor, trim frames)** → **budget-fill (add frames back to use the budget)**.
6. "Still leaving 100KB; want smooth, don't trim" → root-caused the one-directional
   trim; added the **frame-fill** + the user's idea of a **Frames-vs-color setting**
   (smooth/balanced/sharp), all of which **fill the budget**.

The throughline: the format forces a frames-vs-color tradeoff at a fixed size; we
made it explicit and controllable and ensured the budget is always used — and the
honest ceiling on "automatic quality" is the placeholder metric, which M2 lifts.

---

## 11. Commit reference (this session, newest first)
```
ff65d0b Frames-vs-color control for GIFs that always fills the byte budget
9514723 Budget-fill: trim frames until the palette uses the byte budget
5ae0f53 Auto-balance GIF: hold a color floor, trim frames only as needed
c22ee5b Denser color ladder so the size search climbs into the budget
343c980 End-to-end audit logging with content hashes
5ec92f4 Use the byte budget for richer color; show frames/colors tradeoff
40c5219 Make Fovea-vs-standard explicit on download; custom size input + persisted limits
9137156 Size/dimension controls + Fovea-vs-standard comparison in the UI
48bb9a4 Wire Fovea into the live GIF + emoji encode path (M4 stage 1)
0b79669 Bundle gifsicle in the image; ignore Fovea build artifacts
0ee4d71 Fovea test suite
b2510be Fovea benchmark harness (M0)
cedb0cd Fovea perceptually-lossless GIF encoder core (M1)
```
