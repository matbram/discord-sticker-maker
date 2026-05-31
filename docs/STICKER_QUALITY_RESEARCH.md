# Maximizing Animated Sticker Quality at 320×320 / ≤512 KB

Deep-research synthesis (5 parallel research agents, primary sources prioritized:
encoder source code for gifski/libimagequant/ffmpeg/apngasm/oxipng, the GIF89a &
APNG specs, and benchmarks). Confidence tags: HIGH = spec/source-code level or
multiply-corroborated; MED = single benchmark or directional; LOW = inferred.

## TL;DR

1. **Color reduction is the worst lever we have, yet we reach for it first.**
   gifski's own source comment: *"128c → 64c is only 7% smaller."* Our `encode_gif`
   ladders 256→128→64→32 to hit budget — destroying color (the exact user complaint)
   for almost no bytes. **Fix: drop frames/fps before colors.** (HIGH)
2. **Don't write a custom encoder.** PhotoDemon's author wrote one and reports it
   beats gifsicle by only "a percent or two." Orchestrating best-in-class tools gets
   ~95% of the frontier. (HIGH)
3. **Produce BOTH formats, keep the best valid file.** The GIF↔APNG crossover is
   content-dependent and unpredictable a priori. (HIGH)

## What moves bytes, ranked (resolution is fixed at 320×320)

| Lever | Savings | Quality cost | We use it? |
|---|---|---|---|
| Frame count / FPS | ~linear (halve fps ≈ halve size) | choppier; eye saturates ~15–20 fps | partially |
| **Inter-frame diffing** | 1.5–10× (content-dependent) | **none (lossless)** | APNG yes (apngasm `-i1`); GIF no |
| Lossy LZW (gifsicle `--lossy`) | 30–50% | mild noise, hidden by dither | **no** |
| Zopfli / oxipng final squeeze | 3–8% | **none (lossless)** | APNG partial (`-z2` not final oxipng) |
| **Color reduction** | **~7% per halving** | **severe banding/washout** | **yes — first! (wrong)** |

We lean hardest on the worst row and underuse the top four.

## Format guidance (with numbers)

- **APNG wins** for photographic / gradient / soft-alpha content — truecolor + 8-bit
  alpha vs GIF's 256 colors + 1-bit. Real example: 386 KB APNG vs 472 KB GIF, APNG
  better-looking. (MED)
- **GIF wins** for flat / cartoon / few-color and long clips. 600×600 case: GIF
  2.26 MB vs APNG 2.46 MB. (MED)
- **gifski beats our ffmpeg GIF** measurably: 1.69 MB vs 2.15 MB on one clip, with
  *visible* ffmpeg dithering artifacts — because gifski (read from source) uses
  per-frame local palettes (libimagequant) + a 5-frame temporal denoiser +
  inter-frame importance maps + cross-frame palettes. (MED/HIGH)

## Our pipeline: confirmed bugs & gaps

- **GIF (`encode_gif`): colors-first reduction is the core washout bug.** ~7%/halving
  for severe quality loss. Should drop frames/fps and keep ~256 colors. Also: we use
  ffmpeg global palette, not gifski.
- **APNG (`encode_animated`): inter-frame diffing IS happening** (apngasm `-i1`, which
  "automatically calculates and stores the difference between frames"). Compositing to
  full RGBA per frame is fine *because* apngasm re-derives diffs. (HIGH) But:
  - **No final oxipng/zopflipng pass.** Optimal chain is apngasm (`-z2`, high `-i`) →
    **oxipng `-o max`** LAST (oxipng can't redo inter-frame, so order matters). +3–8%.
  - We still color-reduce in some paths; truecolor-first is correct for photographic.
- **Tooling gap:** Dockerfile ships only `ffmpeg pngquant apngasm`. Missing **gifski**,
  **gifsicle**, **oxipng/zopflipng** — the tools that close the gap. All have in-process
  Python bindings: `pygifsicle`, `pyoxipng`, `apngasm-python` (in-memory, ~2× faster
  than our temp-file approach), gifski C-lib. (HIGH)

## Quantization & dithering (refinements, don't change the strategy)

- **libimagequant is the best classical quantizer** (modified median-cut + variance
  splitting + Voronoi/k-means refinement + gamma + perceptual edge weighting). Shared
  by pngquant AND gifski — picking gifski gets it for free. (HIGH)
- **Perceptual color space (CIELAB) barely helps** median-cut — RGB already correlates
  with lightness. Don't chase it. (HIGH)
- **Dither vs size vs flicker — no free lunch:** error diffusion (Floyd–Steinberg,
  sierra2_4a) reduces banding but adds HF noise that (a) inflates files and (b) causes
  frame-to-frame "swarming" on static regions. The only temporally stable dither is
  **Bayer**; or constrain error diffusion to the changed rect (`diff_mode=rectangle`).
  libimagequant's adaptive dither self-limits to non-edge flat/gradient areas. (HIGH)
- **NeuQuant is NOT clearly better than modern libimagequant** (that claim is dated, vs
  naive median-cut). (MED)

## Novel cheap win: saliency/face-weighted palettes

libimagequant exposes a **per-pixel importance/weight map**. Feeding a cheap face/
saliency mask as that weight gives faces more palette colors — the GIF-world analog of
ROI bit-allocation that *no* shipping tool does. CPU-feasible, no ML training. (MED)
Worth a Phase-4 experiment; not core.

## ML verdict

Skip it for the core encode — learned codecs don't output GIF/APNG (format-incompatible).
Only defensible roles: (a) the saliency mask above, (b) optional super-resolution if we
ship at reduced dimensions. Classical tools win on effort/risk. (HIGH)

## Recommended two-tier architecture

**Preview tier (sub-2s, interactive):** one fast encode — ffmpeg palettegen/paletteuse
or `gifski --fast`; no zopfli. Visual feedback only.

**Export tier ("Maximize quality", race candidates, keep smallest-passing /
highest-quality under 512 KB):**
- **GIF:** gifski (high `--quality`) → gifsicle `-O3 --lossy=N`.
- **APNG:** pngquant *only if content needs ≤256 colors, else truecolor* → apngasm
  (`-z2`, high `-i`) → oxipng `-o max` (final lossless pass).
- **Hit budget by dropping FPS/frames first, keep full color.**
- Produce both formats; keep the best valid file.

**Size targeting** is the orchestrator's job (no tool targets bytes natively): probe
encode → predict → 2–4 step search. (HIGH that binary-search is standard practice;
probe-predict is our optimization, not a cited algorithm.)

## Phased implementation for this repo (safest first)

1. **Stop reducing colors first in `encode_gif`** — drop frames/fps instead. Biggest
   user-visible win, smallest change, **no new deps.** (Partly done — verify ladder
   order.)
2. **Add oxipng final pass** to the APNG path (`apngasm → oxipng -o max`). Lossless,
   small, safe. New dep: `oxipng`/`pyoxipng`.
3. **Add gifski** for the GIF path (Dockerfile + binding). Measurable quality jump.
4. **"Race both formats, keep best"** export tier behind the existing fast preview.
   Optional: saliency-weighted palettes (Phase 4 experiment).

## Honest caveats

- Many primary pages (kornel.ski, blog.pkh.me, christianselig, 30fps.net) 403'd the
  fetchers; spec/source-code facts are solid, but several single-clip benchmark
  *numbers* are directional, not laws.
- **No source measured gifski/apngasm wall-clock at 320×320 / ~50 frames** — the
  "10–30 s" budget is an estimate. **Benchmark locally before committing UX to it.**
- "Probe → predict size in one shot" is synthesis, not a cited result; binary search
  (2–4 encodes) is the documented norm.

## Key sources

gifski (source): https://github.com/ImageOptim/gifski ·
libimagequant: https://github.com/ImageOptim/libimagequant ·
pngquant: https://github.com/kornelski/pngquant ·
oxipng: https://github.com/oxipng/oxipng ·
apngasm: https://apngasm.sourceforge.net/ ·
ffmpeg palette filters: https://ffmpeg.org/ffmpeg-filters.html ·
ubitux high-quality GIF: https://blog.pkh.me/p/21-high-quality-gif-with-ffmpeg.html ·
ImageMagick APNG bloat (6.3× w/o opt): https://github.com/ImageMagick/ImageMagick/issues/6147 ·
gifsicle man: https://www.lcdf.org/gifsicle/man.html ·
Discord sticker FAQ: https://support.discord.com/hc/en-us/articles/4402687377815 ·
bindings: https://pypi.org/project/pyoxipng/ , https://pypi.org/project/apngasm-python/ , https://pypi.org/project/pygifsicle/
