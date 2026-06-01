# M2 — the learned, motion-aware judge (foundation)

The default judge (`metrics/perceptual.py`, MS-SSIM + temporal) is **blind to color
banding**: at a fixed low palette it rates a *banded* frame as closer to the source
than a *dithered* one. That blind spot is why the GIF bridge can't trust the metric
to optimize quality and instead uses hand-tuned color-floor heuristics
(`backend/app/pipeline/fovea_gif.py`). **M2 replaces the judge with a small CNN
trained to grade quality the way a person would** — and it is the prerequisite for
later deleting that heuristic.

This is the **machinery + a working, demonstrably-better model**, shipped **behind a
flag** with MS-SSIM as the safe default. Human-preference validation is the
remaining real-world gate (see *Limits*).

## How it plugs in (no churn to the rest of the system)
- `encoder/metrics/learned.py::LearnedMetric(Metric)` — loads `models/judgenet.onnx`
  via **onnxruntime** (no torch at encode time) and returns the same `DistanceResult`
  shape as `PerceptualMetric`, so it is a drop-in behind the `Metric` interface.
- `encoder/metrics/judge_features.py` — the shared, numpy-only feature builder used by
  **both** training and inference (no train/serve skew). Channels per sampled frame:
  ref/cand luma, |Δ|luma, cand & ref temporal deltas (the flicker signal), Cb/Cr error.
- `encoder/metrics/__init__.py` — **opt-in only**: `default_metric()` returns the
  learned judge *only* when `FOVEA_METRIC=learned` and the model loads; otherwise
  MS-SSIM. `get_metric("learned")` / `--metric learned` work explicitly. Missing
  model or onnxruntime → warn and fall back (never crash).

## Pipeline (`encoder/metrics/training/`)
1. `fetch_clips.py` — pull small permissively-licensed **real** clips from GitHub
   (`raw.githubusercontent.com` is reachable here; general video CDNs are 403). Got
   Big Buck Bunny (CC-BY).
2. `synth_clips.py` — ffmpeg `lavfi` clips covering screen_recording / video_clip /
   motion_graphics (guaranteed coverage + gap-fill).
3. `degrade.py` + `gen_dataset.py` — build labeled pairs with **derivable** orderings
   (no humans): parametric banding/flicker/choppiness/blur severity chains, plus real
   GIF lever variants where *more colors ⇒ closer* and, at a fixed low palette,
   *dithered ⇒ closer than banded*.
4. `model.py` (`JudgeNet`, ~100K params) + `train.py` (CPU, weighted-hinge pairwise
   ranking + identity anchor) → `export.py` → `judgenet.onnx`.

Corpus, dataset, and `models/` artifacts are gitignored; regenerate with the commands
in *Reproduce*. The model is **not committed** — a fresh checkout falls back to
MS-SSIM until the pipeline (or a release artifact) provides `judgenet.onnx`.

## Result (this build: 10 clips incl. real BBB, 700 pairs, CPU)
Pair-ranking accuracy via `fovea-bench judge-eval --metrics msssim,learned`
(`bench/judge_eval.py`). **Held-out clips** (never trained on):

| metric  | all | **dither_vs_band** | banding | flicker | choppiness | blur |
|---------|-----|--------------------|---------|---------|------------|------|
| msssim  | 97% | **50% (chance)**   | 100%    | 100%    | 93%        | 100% |
| learned | 98% | **100%**           | 100%    | 100%    | 90%        | 100% |

The headline: on the dithered-vs-banded case MS-SSIM is **at chance** (it can't tell
them apart) while the learned judge is **100%** — the blind spot M2 exists to fix —
and the learned judge is ≥ MS-SSIM overall on held-out clips. Across *all* pairs the
tiny model is a little fuzzier than MS-SSIM on the easy monotone-severity families
(it trades some precision there for correctness on banding); that, plus more data and
a larger model, is future work.

## Reproduce
```bash
pip install -e ".[dev]" && pip install torch onnxruntime onnx     # torch from default PyPI
python -m encoder.metrics.training.fetch_clips bench/corpus/train_clips   # best-effort real clips
python -m encoder.metrics.training.synth_clips bench/corpus/train_clips   # guaranteed coverage
python -m encoder.metrics.training.gen_dataset --out bench/corpus/dataset
python -m encoder.metrics.training.train  --data bench/corpus/dataset --out encoder/metrics/models
python -m encoder.metrics.training.export --ckpt encoder/metrics/models/judgenet.pt --out encoder/metrics/models
fovea-bench judge-eval --metrics msssim,learned          # writes bench/out/judge_eval.{md,json}
FOVEA_METRIC=learned fovea encode IN.mp4 --target-size 512KB -o out.gif   # encode with the learned judge
```

## Limits / honest bounds
- **Not human-validated.** Labels are a synthetic + lever *oracle*, not human votes
  (no labelers in this environment). Human-preference agreement on real content is the
  outstanding gate **before this can become the default** (`metrics/__init__.py`).
- Small CPU model on a small, largely-synthetic corpus (one real clip, BBB). It
  *demonstrably* fixes the banding blind spot and generalizes to held-out clips, but is
  not a calibrated human-perception model.
- Not yet wired into the Docker image (model is gitignored); a build step or release
  artifact must provide `judgenet.onnx` before production can opt in.
- Once human-validated, `fovea_gif._run_fovea` can drop its color-floor/budget-fill
  heuristic and let `guided_search` minimize the learned distance directly.
