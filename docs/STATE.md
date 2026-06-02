# Fovea — Project State & Handoff (living document)

> **Read this first when resuming.** It is the canonical "where we are / how it
> works / what's next." Last updated: the session that built the **native Rust
> encoder** (`fovea-core`) — per-frame local palettes + perceptual OKLab delta that
> **dissolves the frames-vs-color "ceiling"** (§4.2/§4.3) — wired it behind the
> Engine ABC, **activated the M0 corpus**, then (Phase 2) made the metric
> **trustworthy on color** (banding-aware OKLab term) and **retired the bridge's
> frame-trim bandaid** — then fixed a live **over-budget regression** at 512px by moving
> the byte-target search into ONE Rust call (it never exceeds the cap and stays fast).
> Branch `claude/youthful-bell-Nr3rP`. See §0.
> Companion docs: `docs/fovea-spec.md` (the why), `docs/architecture.md`,
> `docs/metrics.md`, `docs/bench.md`, `docs/cli.md`, `docs/m2-judge.md` (the learned judge).

---

## 0. Recent updates (latest session)

### Native Rust encoder + M0 corpus — branch `claude/youthful-bell-Nr3rP`

The big one: a from-scratch **native encoder** that breaks the single-global-palette
ceiling behind GIF washout. This is M3's core idea, built and shipped behind the
existing abstractions (not a rewrite).

**Why this matters.** The doc has called "(frames × colors)" a *format ceiling*
(§4.2/§4.3). It is not — it was an artifact of the old pipeline forcing **one global
≤256-color palette on every frame** (`engines.py` ffmpeg `palettegen`/`paletteuse`
defaults), which is why all 29 frames collapsed to ~16 colors. GIF89a actually allows a
**local color table per image block** (thousands of colors per file) and
**transparency-based inter-frame delta** (unchanged pixels cost ~0 bytes). The native
core uses both.

**What landed.**
- **`fovea-core/`** — Rust crate (`gif` + `imagequant` + an OKLab module), built as an
  **abi3 `fovea_native` wheel** via maturin. Two correct modes, auto-selected:
  *delta* (opaque clips: per-frame local palette + OKLab-ΔE transparency reuse under
  `dispose=Keep`) and *full* (alpha matte: independent per-frame palettes). 8 `cargo`
  tests + 5 pytest tests; GIL released during encode.
- **`FoveaNativeEngine`** wired behind the existing `Engine` ABC and **preferred** by
  `_select_engine`; the ffmpeg engines remain automatic fallbacks/baselines. Reuses
  `guided_search`/the color ladder unchanged. **Dockerfile** gains a `rustbuild` stage;
  runtime pip-installs the wheel (optional — import failure falls back to ffmpeg).
- **M0 corpus activated** — `bench/corpus/generate.py` writes a deterministic synthetic
  corpus as **APNG** (full-color sources) + `bench/corpus/synthetic.yaml`. `fovea-bench
  run` works locally on it (no ffmpeg needed for the native engine).

**Measured (synthetic corpus, native engine, ≤512KB, all frames kept, ~20fps):**
| clip | mode | distinct colors | reuse | note |
|---|---|---|---|---|
| grad_pan (≈600k src colors) | delta | **6090** | 4.5% | full-motion gradient; **24× the 256 ceiling** |
| subject_move (partial motion) | delta | **1688** | **87%** | the sweet spot: rich color + all frames + 75KB |
| full_motion (hard case) | delta | 937 | 14% | near-lossless at 256KB |
| sticker_alpha (transparent) | full | 104 | – | correct alpha path, perceptually perfect |

**What's next (revised order).** Phase 2 — **trustworthy metric + rate-distortion
search**: add an OKLab-ΔE banding term + validate/wire the M2 judge so the search can be
trusted on color, then **retire the bridge's color-floor/frame-trim heuristic** (§0
below) since per-frame palettes make "all frames + rich color" fit. Phase 3 (moonshot) —
**region/multi-block palettes** for >256 colors in a single frame, and **delta-with-alpha**
(erase via per-frame dispose) so stickers also get delta savings. Then head-to-head vs
`ffmpeg-palette` on the corpus in Docker to publish the "better at equal size" table (§11).

### Phase 2 — trustworthy color metric + bandaid retired (same branch)

The metric is now **trusted on color**, and the bridge's frame-trimming bandaid is gone
on the native path. Both shipped + tested.

- **Banding-aware default metric** (`encoder/metrics/banding.py`, `ColorAwareMetric` in
  `perceptual.py`). The old judge scored **luma** MS-SSIM, which is structurally blind to
  *chroma* banding and can even prefer it to dither (the blind spot in `docs/m2-judge.md`
  that forced the color-floor bandaid). The new term low-pass-filters both images in
  **OKLab** and measures ΔE: dithering preserves the local mean (low error), banding
  shifts it in steps (high error) — exactly inverting MS-SSIM's mistake, *and* it sees
  chroma. `distance = (1−MS-SSIM) + β·temporal + γ·banding` (γ=3), now the **default**
  (`default_metric()`); pure MS-SSIM stays available as `"msssim"` for comparison.
  Validated: on a chroma ramp where MS-SSIM rates hard banding at 0.05 (≈lossless), the
  color-aware metric rates it **0.47 vs 0.03 for dither** — reliably prefers dither.
  `invisible_threshold=0.02` lands between good-dither (≈0.008) and banding (≈0.033).
  Tests: `tests/test_banding_metric.py`. *(This makes the always-available metric trusted
  on color, so the M2 CNN is no longer needed as the default — it stays opt-in via
  `FOVEA_METRIC=learned` for motion-awareness; bundling its `.onnx` is deferred until the
  trained artifact exists.)*
- **RD search**: with a trustworthy distance, the existing "minimise distance s.t. ≤
  budget over colors + dither" search *is* rate-distortion and now makes the right color
  call (it will keep dither/rich color instead of banding). A joint delta-threshold lever
  was scoped out (the default ΔE≈0.02 is already ~1 JND; a proper joint search needs
  deeper `guided_search` surgery for marginal gain) — noted as a future refinement.
- **Bridge bandaid retired** (`backend/app/pipeline/fovea_gif.py`). When the native engine
  is built, `_run_fovea` skips the color-seeking frame-trim (2a) and frame-fill (3) — per-
  frame palettes already give rich color with every frame, so trimming only threw frames
  away. The **mandatory fit-rescue (2b)** still runs (the byte cap is hard), and the
  **legacy ffmpeg fallback keeps the full dance** (it still has the single-palette limit).
  Net: on the native path, `priority` (smooth/balanced/sharp) is now effectively a no-op —
  the tradeoff it managed no longer exists (a product decision for whether to repurpose
  the control). `FOVEA_FORCE_LEGACY_BRIDGE=1` is an escape hatch.

### Production fix — over-budget regression at large sizes (same branch)

First live test (512×288, 72-frame video @ 512KB) exposed a P0 the local tests missed:
the native engine **shipped 671KB over the 512KB cap**, at only 20 colors, with frames
trimmed (51) and resolution dropped (410×230). Root cause: the **Python-orchestrated
search made one slow Rust call per color probe** (re-copying frames, PNG round-trips),
and the per-encode time budget expired mid-bisection (`stop_reason=budget_seconds`,
`attempts=2`) → it shipped the best *over*-budget candidate. The metric made it worse:
`distance()` was **~32s** at full 512px. Fixes:

- **`fovea_native.search` — the whole byte-target search in ONE Rust call**
  (`encode_search`). Two-stage so it's fast *and* not conservative: a **parallel
  full-mode bisection** (frames quantized across cores) gives a guaranteed-fitting lower
  bound, then a **delta bisection climbs upward** (delta is much smaller on real video, so
  it reaches a far richer palette that still fits). It **runs to completion**, so it can
  never time out into an over-budget file — and always returns the smallest it can.
- **Metric ~8× faster** (`encoder/metrics/perceptual.py`): downscale both sides to
  `FOVEA_METRIC_MAXDIM` (192px) before scoring — banding/structure are low-frequency, so
  it's lossless to the *ranking* — plus a slicing-based box blur. ~32s → ~4s.
- **Bridge trusts the native encode** (`fovea_gif.py`): one `_encode_once`, no
  invisible→cap→fit-rescue triple-encode (the search self-guarantees the fit and shrinks
  invisible internally). Legacy ffmpeg path keeps the full dance (autouse-fixture-forced in
  `tests/test_fill_and_fps.py` / `test_invisible_fallback.py`).

Verified end-to-end on a dark/partial-motion 512px clip: **9s cap / 13s invisible, 327KB ≤
512KB, 256 colors, perceptually lossless, all 72 frames, full 512×288** (was 50s+, 671KB,
20 colors, 51 frames, 410×230). New Rust tests:
`search_always_fits_the_target_and_keeps_all_frames`, `search_returns_smallest_when_nothing_fits`.

---

### Prior session — branch `claude/inspiring-cray-wBFKB` → **PR #3** (pushing more commits updates that
PR; do not open another). Full suite is **68 tests green**. Six threads landed:

**Quick wins (§8.2) — shipped.** Per-output `priority` **and** `mode`: the GIF's
Frames-vs-color (and the new invisible toggle) no longer move the sticker/emoji, and the
choices persist to `dsm_outopts_v1`. **Invisible mode** is wired end-to-end
(`OutputSpec.mode` → orchestrator → bridge). The encoder's **honesty report** reaches
`StickerMeta.report` and renders as a one-line verdict in the UI. A **warning** note now
fires when the Fovea-vs-standard comparison falls back instead of failing silently. A
**global per-job deadline** (`FOVEA_JOB_SECONDS`, 100s, under the 120s watchdog) bounds
the whole multi-encode job. *Deferred:* defaults tuning (still needs M0).

**M2 learned judge — foundation built, flag-gated** (full writeup: `docs/m2-judge.md`).
A small ONNX/onnxruntime CNN judge (`encoder/metrics/learned.py`) + a training pipeline
(`encoder/metrics/training/`) over real GitHub-hosted clips + synthetic degradations with
*derivable* labels. On held-out clips it scores **100% on the dithered-vs-banded case
where MS-SSIM sits at chance (50%)** — the exact blind spot that forces the bridge's
color-floor heuristic (§3.5/§4.3). **OFF by default** (`FOVEA_METRIC=learned` to opt in);
MS-SSIM stays default with a graceful fallback when the model/runtime is absent. The
`.onnx` is gitignored (reproduce via the pipeline). Harness: `fovea-bench judge-eval`.

**Production-log fixes.** Encoder logging now goes to **stdout** (was stderr, which
Railway filed as an ERROR on every encode). Invisible mode **falls back to cap
budget-fill** when a clip can't be made losslessly small — the encoder never trims
frames, so a stuck low-palette invisible result was washed out and *worse* than the
default; falling through guarantees invisible is never worse than cap
(`tests/test_invisible_fallback.py`).

**Latency (per-edit).** A **source cache** (`backend/app/source_cache.py`) lets the
browser upload once and reference the source by `source_id` on every regenerate (POST
~13s → ~6ms; transparent re-upload on a 409 `source_expired`). A **legacy-baseline
cache** stops the side-by-side re-running the standard encoder when only `priority`
changed. SSE keepalive tightened **60s → 15s**, plus a **per-output deadline**
(`FOVEA_OUTPUT_SECONDS=45`) — together these killed the `ERR_CONNECTION_RESET` /
`watchdog_timeout` on slow 512px encodes.

**Bridge quality (3 fixes from production logs).** (1) **fps floor** (`FOVEA_MIN_FPS=8`,
`FOVEA_MIN_FPS_SHARP=5`) + a **fit-rescue** so trimming never makes a 2-fps slideshow yet
still yields to the hard byte budget. (2) **Budget-fill dither fix**: frame-fill now
re-encodes at the *chosen* candidate's dither (was always `sierra2_4a`), so the size
estimate is right and we stop leaving budget on the table. (3) **Single source of
truth**: the comparison card's Fovea %/lossless now come from the encoder report, so the
side-by-side badge can't contradict the honesty line. Verified on the 512px/512KB sharp
case: **6f / 2.1fps / 80% → 12f / 5.2fps / 95%**.

**What's left (next).** Gate M2 to default via **human-preference validation**, then drop
the color-floor heuristic and let the search optimize the learned distance directly; wire
the M2 `.onnx` into the Docker image; activate the **M0 corpus** (§8.1); add **resolution
as a bridge lever** for the genuinely resolution-bound cases (the 512px clip is
dimension-limited, not budget-limited); then M3/M4/M5 (§8.4–8.6). See §8.8 for order.

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
  metric)** — whose **foundation is now built and flag-gated** (§0), pending human
  validation before it can become the default.

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
    base.py msssim.py temporal.py perceptual.py external.py __init__.py (flag-gated registry)
    learned.py judge_features.py   # M2 learned judge (onnxruntime); OFF unless FOVEA_METRIC=learned
    training/                      # M2 data+train+export pipeline (synth/fetch clips, degrade, model, train, export)
  cli/main.py                 # `fovea encode ...`
bench/                        # M0 harness: manifest.py records.py runners.py run.py cli.py judge_eval.py corpus/(no clips)
tests/                        # 68 unit tests (no binaries) + test_integration_smoke.py (gated on ffmpeg/gifsicle)
docs/                         # specs + this file + m2-judge.md
backend/app/pipeline/fovea_gif.py   # *** the live integration bridge *** (fps floor, fit-rescue, dither-accurate fill)
backend/app/pipeline/orchestrator.py# threads budget/dims/priority/mode + deadlines into the GIF path
backend/app/source_cache.py         # upload-once source cache (source_id) — cuts per-edit latency
backend/app/models.py               # OutputSpec.max_bytes/max_dim/priority/mode; StickerMeta.comparison/report
backend/app/main.py                 # audit.* + SHA1; source_id ingest; SSE keepalive; per-job deadline
frontend/src/App.svelte             # per-output size/dim/priority/mode + invisible + honesty line + comparison
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

### 4.3 `_run_fovea` — the algorithm (`cap`: phases; `invisible`: shrink-or-fall-back)
For a given `priority` → a **color-richness floor** via `_color_floor_for`:
`smooth→0`, `balanced→64`, `sharp→160`. Also an **fps floor** so trimming can't make a
slideshow: `min_n = clamp(round(min_fps × duration), MIN_FRAMES=6, total)`, where
`min_fps` = `FOVEA_MIN_FPS_SHARP` (5) for sharp else `FOVEA_MIN_FPS` (8). Duration is
preserved across subsampling, so frames ÷ duration == output fps.

**`invisible` mode** asks the encoder for the smallest perceptually-lossless GIF under the
cap (every frame kept). If the clip *can't* be made lossless under the ceiling it **falls
through to `cap`** below — the encoder never trims frames, so a stuck low-palette
invisible result is washed out and worse than the default (`fovea.invisible_fallback`).

**`cap` mode — phases:**
1. **Encode all frames** at the richest palette the budget allows (one Fovea `encode`).
   Often this already fills the budget (just at low colors).
2. **(a) Color-seeking trim** (balanced/sharp, skipped for `smooth`): drop frames ×0.72
   per step toward the palette floor, **clamped at the fps floor** (`f = max(min_n, …)`)
   so we never trim into a slideshow. **(b) Fit-rescue:** if even the fps-floor result
   still overshoots the byte budget, keep trimming below the floor toward `MIN_FRAMES` —
   fitting the hard cap is mandatory; the fps floor is only a preference.
3. **Frame-fill** (`_fill_frames_at_colors`): add frames back at the chosen palette **and
   the chosen candidate's dither** until the byte budget is used (GIF size is ~linear in
   frame count at fixed colors+dither). Passing the *right* dither is what fixed a
   budget-under-fill bug — frame-fill used to assume `sierra2_4a` while the cap often
   picked `dither=none`, so the size estimate (and the stopping point) were wrong.

Every phase respects the per-encode/per-job deadline (§7): when time's nearly up it keeps
the best candidate so far instead of starting another encode. Log lines: `fovea.fill`
(phases 1–2a), `fovea.fitrescue` (2b), `fovea.framefill` (3), `fovea.invisible*`.

Observed spectrum on the real portrait clip (180×320) @ 512KB (all fill the budget):
| mode | frames | colors | usage |
|---|---|---|---|
| smooth | 29/29 | 16 | 99% |
| balanced | 14/29 | 112 | 97% |
| sharp | 11/29 | 256 | 93% |
(`sharp` caps at 93% only because 256 colors is the GIF max — nothing more to add.) On a
*small* 512px-square clip at 512KB the case is resolution-bound, not budget-bound; the
fps floor + fit-rescue + dither-accurate fill took sharp from **6f/2.1fps/80% to
12f/5.2fps/95%**. The deeper fix (resolution as a bridge lever) is noted in §8.2.

### 4.4 The comparison + single source of truth
`gif_encode_compare` runs Fovea **and** legacy and returns a `comparison` dict
(bytes/frames/colors/distance per side, `perceptually_lossless`). **Single source of
truth:** the Fovea side's distance + lossless flag come straight from the encoder
**report** — the same numbers the honesty line shows — so the side-by-side badge can never
contradict it. Only the **legacy** baseline gets its own score, via `_aligned_distance`
(it has no Fovea report). **Fairness fix (legacy side):** that distance is measured
against the **source subsampled to the legacy GIF's own frame count** — otherwise a
frame-trimmed candidate is judged on misaligned frames and looks artificially worse. The
legacy encode is **cached** (`_legacy_*`, keyed on the fitted frames + size/colors/fps,
TTL 900s) so changing only `priority` doesn't re-run it. The baseline GIF is served as a
benign extra output keyed `gif__cmp` (excluded from the zip).

### 4.5 Audit trail + SHA1 chain (how we prove what shipped)
`main.py` logs (structlog → JSON in Railway):
- `audit.request` (per-output max_bytes/max_dim/priority),
- `audit.output.budget` (resolved budget + dims),
- `audit.gif.compare` (`fovea_sha1`, `legacy_sha1`, frames, colors, `primary=fovea`),
- `audit.store` (each stored output: bytes, **sha1**, role),
- `audit.serve` (every preview/download: key, bytes, **sha1**, download flag).
`fovea_gif` logs `fovea.fill`/`fovea.fitrescue`/`fovea.framefill`/`fovea.invisible`/
`fovea.invisible_fallback` per phase; `main.py` logs `source_cache.hit` when a regenerate
reused an already-uploaded source (no re-upload).
**Verification recipe:** the file a user downloads has the same SHA1 as
`audit.serve key=gif download=true`, which equals `audit.gif.compare fovea_sha1`
→ proves the download is Fovea (not the legacy `legacy_sha1`). Users can confirm
locally: `shasum -a 1 file.gif | cut -c1-12`.

---

## 5. Frontend controls (`frontend/src/App.svelte`)
Editor sidebar, per focused output:
- **Max file size** (chips 256KB…8MB + a freeform `800KB`/`3.5MB` input). Binary
  units. Defaults: sticker 512KB, emoji 256KB, gif 5MB. Persist to `dsm_limits_v1`.
- **Dimensions** (chips + custom px). Square side for sticker/emoji; longest edge
  for gif.
- **Frames vs. color** (gif + animated emoji): More frames / Balanced / Richer color →
  `priority` smooth/balanced/sharp — now **per-output** (changing the GIF no longer moves
  the sticker/emoji).
- **Shrink until invisible** toggle (gif + emoji) → `mode` cap/invisible.
- Per-output `priority`+`mode` **persist** to `dsm_outopts_v1` (separate from the
  size/dimension `dsm_limits_v1`).
- **Honesty line** under the focused output (`focusOut.meta.report`): the encoder verdict
  — "stayed perceptually lossless" vs "visible loss worst around frame N" (+ stop reason /
  invisible-shrink result), styled green when lossless.
- **Fovea vs. standard** comparison card under the GIF: both animations, size /
  frames / colors / match%, per-side **Download Fovea / Download standard**, and a
  verdict. The match% + lossless flag come from the encoder report (§4.4). The main
  "Download GIF (Fovea)" button is labeled.
- **Latency:** the source uploads **once**; regenerates send a `source_id` (the source
  cache, §0/§7) instead of re-posting the file, with a transparent re-upload if the
  server's cache entry expired.
Plumbing: controls → `OutputSpec.max_bytes/max_dim/priority/mode` → orchestrator
(`budget`, `out_size`/`out_max_dim`, `_v(eff.priority)`, `_v(spec.mode)`) → bridge.

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
  quality signal. M2 (foundation built, flag-gated — §0/§8.3) is the fix, but not the
  default yet.
- **Performance / no async worker yet (M4 not built):** encodes run synchronously
  in the in-process `PIPELINE_EXECUTOR`. With the budget-fill + comparison + 3 outputs a
  job can take **30–70s**. Bounds now in place: a **global per-job deadline**
  (`FOVEA_JOB_SECONDS=100`, under the 120s client watchdog) and a **per-output deadline**
  (`FOVEA_OUTPUT_SECONDS=45`) both threaded through the bridge; per-encode
  `FOVEA_BUDGET_SECONDS=12`, `FOVEA_MAX_ATTEMPTS=12`. The **SSE keepalive is 15s** so a
  long encode no longer trips `ERR_CONNECTION_RESET`/`watchdog_timeout`. Per-edit latency
  is cut by the **source cache** (upload once, then `source_id`) + the **legacy-baseline
  cache** (don't re-run the standard encoder when only `priority` changed). The real fix
  for scale is still the §8.5 async split (M4).
- **Resolution-bound small clips:** at a *small* dimension (e.g. 512px square) a clip can
  be limited by pixels, not bytes — the budget-fill + fps-floor get it to ~95%, but the
  proper lever (drop resolution to buy frames/colors) isn't wired into the bridge yet
  (§8.2).
- **gifski is not installed in the image** (no apt package). ffmpeg + gifsicle
  are (apt; we disabled two broken PPAs — deadsnakes/ondrej — to install them).
  Fovea works on ffmpeg alone, so gifski is optional.
- **Sticker = APNG = legacy encoder.** Fovea only touches GIF/emoji.
- **M2 `.onnx` is gitignored and not in the Docker image** — `FOVEA_METRIC=learned` only
  does something where the model + onnxruntime are present (else it warns and falls back
  to MS-SSIM). Wiring the artifact into the image is outstanding (§0/§8.3).
- **Local dev:** numpy/Pillow/etc. live in `backend/.venv` (gitignored) and the
  Fovea root `.venv`; the *system* python has none of them.

---

## 8. Remaining scopes (detailed build plan)

Everything below is *not yet built*. Each scope says **why**, **where it plugs
into our code**, the **approach**, **prerequisites**, **risks/open questions**, and
**done when**. Suggested order is in §8.8. The throughline: M1 hit a hard
size target by driving external engines and a *heuristic* GIF budget loop; the
remaining work makes the quality judgment trustworthy (M2), replaces the engines
with native internals that break the frames-vs-color frontier (M3), and turns the
service into a scalable, honest product (M4/M5) — all proven against the
benchmark (M0).

### 8.0 Cross-cutting principles to preserve (do not regress)
- **Spec-compliant GIF** that plays everywhere; **keep every frame** in `encoder/`
  by default (frame trimming is a *bridge*-level, user-chosen behavior only).
- **Real measured size** (always actually encode; never estimate) and the
  **anytime budget** (return best-so-far on a deadline).
- **Honesty over silent failure** — report whether it stayed invisible and where
  any loss landed.
- **The measurement gate (spec §11):** nothing ships as "better" until it beats
  the best baseline at equal size on the corpus. → makes §8.1 a hard prerequisite.

### 8.1 M0 activation — make the benchmark real (do this FIRST; cheap)
- **Why:** the harness exists (`bench/`) but has **no clips**, so we have zero
  hard evidence Fovea beats gifski/gifsicle/ffmpeg. Every future "better" claim
  (and every M2/M3 gate) depends on this.
- **Where:** `bench/corpus/manifest.yaml` (+ drop media into the gitignored
  `bench/corpus/clips/`), `bench/run.py`, `bench/runners.py` (already supports a
  `fovea` engine via `run_clip_target_fovea`).
- **Approach:** assemble a fixed, licensed corpus spanning the three categories
  (screen_recording / video_clip / motion_graphics) × a few durations × the
  target-size ladder. Run `fovea-bench run --engines ffmpeg-palette,gifski,
  gifsicle-lossy,fovea`. Commit the **results table** (CSV/JSON `meta` captures
  versions) and a short written readout per category.
- **Risks:** clip licensing; the MS-SSIM primary metric is itself unreliable for
  color (see 3.5) so treat the table as *directional* until M2; report SSIM/etc.
  as secondary context only.
- **Done when:** `fovea-bench run` produces a reproducible per-clip/per-target
  table and we can state, per category, where Fovea wins/ties/loses at equal size.

### 8.2 Quick wins / polish (incremental, low-risk)
**Shipped this session (§0):**
- ✅ **Per-output `priority` + `mode`** in `App.svelte` (persisted `dsm_outopts_v1`); the
  GIF's controls no longer move the sticker/emoji.
- ✅ **Invisible mode in the UI** ("shrink until invisible"), wired through
  `OutputSpec.mode`; falls back to cap when not losslessly possible.
- ✅ **Honesty reporting surfaced** (`StickerMeta.report` → the UI honesty line).
- ✅ **Warning on silent comparison fallback.**
- ✅ **Global per-job deadline** (`FOVEA_JOB_SECONDS`) + **per-output deadline**
  (`FOVEA_OUTPUT_SECONDS`) threaded through the bridge.

**Still open:**
- **gifski in the image (optional):** add via a pinned release binary or a
  `rust:` builder stage; enables the opaque video→GIF path. Fovea works without it.
- **Tune defaults from data:** `_color_floor_for` (smooth=0/balanced=64/sharp=160),
  `FOVEA_BUDGET_USE` (0.93), and the fps floors (`FOVEA_MIN_FPS` 8 / `_SHARP` 5) were set
  by eye; recalibrate against the corpus (needs M0, §8.1).
- **Resolution as a bridge lever:** for genuinely resolution-bound clips (small dims at a
  generous byte cap — e.g. 512px square @ 512KB) the bridge can't currently spend the
  budget on *pixels*. Add a controlled resolution step (the encoder already has
  `SCALE_VALUES` as a last-resort lever) so the bridge can trade a little resolution for
  frames/colors when frames×colors is maxed but bytes remain — the deeper fix behind the
  95%-not-100% sharp case in §4.3.

### 8.3 M2 — learned, motion-aware perceptual metric (the quality unlock)
**Status: foundation BUILT and flag-gated this session (§0; full writeup `docs/m2-judge.md`).**
The `LearnedMetric` (`encoder/metrics/learned.py`), the training pipeline
(`encoder/metrics/training/`), ONNX export, the flag-gated registry, and the acceptance
harness (`fovea-bench judge-eval`) all exist; on held-out clips it scores **100% on the
dithered-vs-banded case** where MS-SSIM is at chance. **What remains to make it the
default:** (1) **human-preference validation** (the standing gate — no labelers this
session), (2) wire the `.onnx` artifact into the Docker image (it's gitignored), and (3)
once trusted, drop `_run_fovea`'s color-floor/budget-fill heuristic for metric-driven
optimization. The rest of this section is the original design rationale, still valid.

- **Why (the single most important scope):** MS-SSIM **cannot see banding** and
  rates a smooth-but-banded frame as *closer* than a dithered one (3.5). Because
  the judge is untrustworthy, we **cannot let `guided_search` optimize quality
  directly** — so the whole GIF budget logic in `fovea_gif._run_fovea` is a
  hand-tuned heuristic (color floors + budget-fill). A metric that matches human
  preference lets us **delete that heuristic** and have the search choose the
  frames-vs-color point by actually minimizing perceived distortion (with the
  user's `priority` as a weight, not a hardcoded floor).
- **Where it plugs in:** implement a `Metric` subclass (e.g.
  `encoder/metrics/learned.py::LearnedMetric`) with `distance(reference,
  candidate) -> DistanceResult` and its own calibrated `invisible_threshold`;
  register it in `encoder/metrics/__init__.py::default_metric()`. **No other code
  changes** — `guided_search`, `encode()`, and the bench all consume whatever the
  registry returns. Then simplify `_run_fovea` to lean on the metric.
- **Approach:** a small CNN in the spirit of **GIFnets' BandingNet** (Yoo et al.,
  CVPR 2020) **extended to the temporal dimension** (penalize flicker/choppiness,
  not just per-frame banding). Inputs = aligned (reference, candidate) frame
  stacks; output = scalar distance. Train in **PyTorch**, export to **ONNX**, run
  in-loop with **onnxruntime** (already a backend dependency — so the runtime
  image needs *no* torch). Operate on luma + a small color term; consider a
  downscaled proxy for in-loop speed and full-res only for the final report.
- **Training data:** pairs of (source, two equal-size encodings) with human
  preference labels ("which looks closer to the source"), generated by sweeping
  levers (colors, dither mode, frame count, lossy) on the corpus, plus synthetic
  banding/flicker positives. Small, structured pairwise studies (spec §9).
- **Validation (standing, not one-time):** the metric is trusted only to the
  extent its pairwise rankings agree with held-out human preference; disagreement
  is a **defect to fix, not a number to chase**. Keep MS-SSIM / SSIMULACRA2 as
  sanity references.
- **Risks:** the **biggest research risk in the project** (spec §9, §14). A weak
  or gameable judge silently degrades everything. Data collection is real work.
- **Done when:** pairwise rankings clear the agreed human-agreement threshold on a
  held-out set; swapping it in measurably improves blind quality at equal size on
  the corpus (spec §11 gate); and `_run_fovea` can drop its color-floor/budget-fill
  heuristic in favor of metric-driven optimization.

### 8.4 M3 — native internals (the IP frontier; replace external engines)
These are new `Engine` implementations (`encoder/core/engines.py::Engine` ABC) +
new `LeverState` fields (`encoder/core/levers.py`), reusing `guided_search`
unchanged. They need a **native GIF writer** (Python first, then Rust) because
ffmpeg emits a single global palette and can't express the structures below.
**Each needs M2** to judge its rate-distortion tradeoffs honestly.

- **8.4.a Region-local palettes — the lever that breaks the frames-vs-color
  frontier.** GIF allows **multiple image blocks per displayed frame, each with
  its own local ≤256-color table** (almost no encoder uses this). Tile a frame and
  give busy/distinct regions their own palette → far more *effective* colors per
  frame **without dropping a single frame**. This directly dissolves the
  washout-vs-smoothness tension we fought all session (e.g. all 29 frames *and*
  rich color at 512KB). New levers: tile grid, per-tile palette size, a
  when-to-tile RD decision (pay the per-block overhead — descriptor + LZW reset +
  up to ~768B palette — only when fidelity gain > cost). **Open question (spec
  §14):** how often tiling actually pays off — empirical, gate per spec §11. Verify
  multi-block frames render correctly across browsers/Discord/Slack.
- **8.4.b Perceptual (sub-threshold) frame reuse.** GIF's only interframe trick:
  with "do not dispose," a later frame redraws only changed pixels and marks the
  rest transparent (nearly free). gifsicle already does this for **exact** matches;
  our novel extension is to reuse a pixel when the change is **below the visible
  threshold** (needs M2's perceptual model). Compounds: more unchanged pixels →
  larger flat regions → better LZW. Implement as a frame-diff pass producing
  transparency masks fed to the native writer. **Biggest win on partial-motion**
  (screen recordings, talking heads); small on full-frame motion (honest bound).
- **8.4.c Joint RD-LZW + dithering co-tuning.** gifsicle `--lossy` (approximate run
  matching) shrinks the file; *light* lossy increases run redundancy that partly
  pays back the size cost dithering adds — so co-tune lossy strength **with**
  dither and colors against the perceptual judge, never past visible smearing.
  Make lossy-LZW a first-class lever in the native writer (today it's only a
  secondary gifsicle path).
- **8.4.d Rust core.** Once the native writer + reuse + local palettes are proven
  in Python, move the hot loops (LZW, palette quantization, frame diff, region
  segmentation) to **Rust via PyO3/maturin**, mirroring gifski/libimagequant.
  Multi-stage Dockerfile (rust builder → slim runtime).
- **Done when:** measured, repeatable quality-per-byte win over M1 on the corpus
  (strongest on partial-motion), each lever added behind the spec §11 gate, output
  still spec-compliant across target players.

### 8.5 M4 — async worker split + service productionization (spec §13.8)
- **Why:** encodes are CPU-heavy (seconds–minutes, and **growing** with the
  budget-fill loop, the comparison double-encode, and M3). Today they run in the
  in-process `ThreadPoolExecutor` (`main.py::PIPELINE_EXECUTOR`) — which holds a
  web worker for the whole job, relies on the ephemeral container FS, can't scale
  across replicas, and lives under a 120s client watchdog. §13.8 mandates a split.
- **Target topology (two services from one image):**
  - **API** (FastAPI): accept upload → write input to object storage → enqueue a
    job (storage keys + params) → return `job_id`; status endpoint reads job state;
    result endpoint hands a storage download URL. Returns in ms, never encodes.
  - **Worker**: pull job → read input from storage → run `orchestrator.process`
    (Fovea) → write outputs to storage → update status + `EncodeReport` in the
    store.
  - **Redis** (one-click on Railway): job queue (**RQ** default; Celery optional)
    + small status store `{job_id, status, result_keys, report}`.
  - **Object storage**: S3-compatible via `boto3`, configured by env
    (`S3_ENDPOINT/BUCKET/KEY/SECRET`) so it works with **Cloudflare R2 / AWS S3 /
    MinIO**. Payloads carry **storage keys, not file paths**.
  - **Progress**: worker → Redis pub/sub → API SSE (replaces the in-process
    `asyncio.Queue` in `jobs.py`); keep the existing SSE client contract.
- **Provisioning the USER must do (cannot be coded):** create the Redis service, a
  storage bucket + credentials, and a **second Railway service** (worker start
  command) from this repo; set the env vars/secrets.
- **Build with a graceful fallback:** if `REDIS_URL`/storage env are unset, stay in
  **today's in-process synchronous mode** — so a redeploy never breaks before the
  user provisions anything.
- **Also in scope:** enforce **input caps** at upload (max file size / resolution /
  duration / frame count — `encoder.core.frames.InputCaps` exists; surface via
  ffprobe up front) to bound memory/time/cost; honor a **per-job wall-clock
  budget** end-to-end (the encoder's `Budget` supports it; thread a deadline
  through the bridge).
- **Files:** split `main.py` into api + worker entrypoints; add `storage.py` and a
  queue module; Dockerfile two start commands; a second-service config in/alongside
  `railway.json`; an env-config module (nothing hardcoded).
- **Risks:** provisioning friction; progress streaming across services; metered
  compute cost. **Done when:** the API/worker/queue/storage topology runs on
  Railway, long encodes never block a request, progress works, and the in-process
  fallback still works with no infra.

### 8.6 M5 — learned warm-start + spec-compliance hardening
- **Learned warm-start:** a small model predicting good *starting* lever settings
  (colors/frames/dither) from cheap clip features (resolution, motion, color
  complexity, duration) so most of the search is skipped — directly cutting the
  multi-encode cost that makes M4 necessary. Layer on **after** M2/M3 are proven
  (spec §10); it's a speed optimization, not a prerequisite.
- **Spec-compliance hardening:** a compliance test suite over edge cases —
  disposal methods, 1-bit transparency, loop counts, very-short-delay clamping,
  odd dimensions, single-frame, huge frame counts, exotic source formats — with
  playback verification across browsers / Discord / Slack / email. **Especially
  important once M3** emits non-trivial structures (multi-block frames, sub-
  threshold transparency). **Done when:** the suite passes and the encode-time
  target is met.

### 8.7 Benchmark is the gate, not an afterthought
`bench/` (M0) is built but **unused**. It is the instrument for every claim above:
add real clips, run `fovea-bench run` with the `fovea` engine included, and keep
the table current. Per spec §11, a lever or model ships only if it measurably beats the
best baseline at equal size (or matches quality at smaller size) across the corpus.

### 8.8 Dependency graph / suggested order
1. **M0 activation (§8.1)** — cheap; unlocks measurement; gates everything else. *(Still
   open — the one cheap prerequisite that now blocks defaults tuning and the M2/M3 gates.)*
2. **Quick wins (§8.2)** — ✅ mostly shipped (per-output priority+mode, invisible, honesty,
   deadlines); still open: gifski, defaults tuning (needs M0), resolution-as-a-lever.
3. **M2 metric (§8.3)** — ✅ foundation built + flag-gated; **remaining: human validation +
   Docker wiring** before it's the default and we can drop the budget-fill heuristic.
4. **M3 internals (§8.4)** — region-local palettes + perceptual frame reuse + joint
   RD-LZW, then the Rust core. Needs M2 (trusted) to judge tradeoffs honestly.
5. **M4 async split (§8.5)** — increasingly necessary as encode cost grows (M3) and
   for scale; can proceed in **parallel** (it's infra) whenever timeouts/scale
   demand and the user provisions Redis + storage + a worker service. *(The per-job /
   per-output deadlines + source/legacy caches from §0 are interim mitigations.)*
6. **M5 (§8.6)** — warm-start speedups + compliance hardening once internals exist.

---

## 9. Dev & deploy runbook (how to resume)

**Set up locally**
```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"            # numpy, Pillow, PyYAML, pydantic, pytest
pytest -q                          # 68 unit tests, no binaries needed
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
`FOVEA_MAX_ATTEMPTS` (12), `FOVEA_MIN_FPS` (8), `FOVEA_MIN_FPS_SHARP` (5),
`FOVEA_JOB_SECONDS` (100 — whole-job deadline), `FOVEA_OUTPUT_SECONDS` (45 — per-output
deadline), `FOVEA_METRIC` (unset → MS-SSIM; `learned` opts into M2 *iff* the model +
onnxruntime are present, else warns and falls back). Color-floor per mode is hardcoded in
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
7. **This session — quick wins + the M2 foundation.** Shipped per-output priority+mode,
   the invisible toggle, the honesty line, and the comparison-fallback warning; then built
   the **M2 learned judge** end-to-end (flag-gated, beats MS-SSIM on the banding case) as
   the path to eventually deleting the budget-fill heuristic.
8. **This session — hardening from production logs.** Routed encoder logs to stdout (no
   more phantom ERRORs); made invisible mode fall back to cap so it's never worse; added
   an **fps floor + fit-rescue** (no 2-fps slideshows) and a **dither-accurate frame-fill**
   (stopped leaving budget on the table); made the comparison badge read from the encoder
   report (**single source of truth** — the user's "conflicting metrics" report); and cut
   per-edit latency with a **source cache** (`source_id`) + **legacy-baseline cache**,
   tightening the **SSE keepalive (15s)** + **per-output/per-job deadlines** to end the
   connection-reset timeouts. 68 tests; PR #3.

The throughline: the format forces a frames-vs-color tradeoff at a fixed size; we
made it explicit and controllable and ensured the budget is always used — and the
honest ceiling on "automatic quality" is the placeholder metric, which M2 lifts.

---

## 11. Commit reference (newest first)

**This session (branch `claude/inspiring-cray-wBFKB` → PR #3):**
```
1584a58 Bridge: fps floor (no slideshows), accurate budget-fill, single-source metrics
61ca61a Keep SSE alive during long encodes; cap per-output encode time
4f339cd Cut per-regenerate latency: cache the uploaded source and the comparison baseline
b777067 Fix encoder log severity (stderr->stdout) and invisible-mode quality regression
57004e2 Update STATE.md: record §8.2 quick wins + M2 learned-judge foundation
d63cfa5 M2: learned perceptual judge (flag-gated) that fixes the banding blind spot
1d32a73 Quick wins: per-output priority, invisible mode, honesty report, job deadline
```

**Earlier (handoff doc + the Fovea build/integration session, merged via PR #2):**
```
46d2d26 Expand STATE.md §8 into a detailed remaining-scopes build plan
5d20618 Add living handoff doc (docs/STATE.md) + root README
ff65d0b Add a Frames-vs-color control for GIFs that always fills the byte budget
9514723 Budget-fill: trim frames until the palette actually uses the byte budget
5ae0f53 Auto-balance GIF: hold a color floor, trim frames only as needed
c22ee5b Denser color ladder so the size search climbs into the available budget
343c980 Add end-to-end audit logging with content hashes
5ec92f4 Use the byte budget for richer color; show frames/colors tradeoff
40c5219 Make Fovea-vs-standard explicit on download; custom size input + persisted limits
9137156 Add size/dimension controls and a Fovea-vs-standard comparison to the UI
48bb9a4 Wire Fovea into the live GIF + emoji encode path (M4 stage 1)
0b79669 Bundle gifsicle in the image; ignore Fovea build artifacts
0ee4d71 Add Fovea test suite
b2510be Add Fovea benchmark harness (M0)
cedb0cd Add Fovea perceptually-lossless GIF encoder core (M1)
```
