# The perceptual judge

Everything depends on the quality judge: if it is wrong, the encoder "wins" on
paper while shipping something worse. Until the learned, motion-aware metric
(milestone M2) is trained, Fovea uses a **reference** metric — clearly labeled as
a placeholder — behind the pluggable `Metric` interface.

## Default metric: MS-SSIM + temporal flicker

`metrics.perceptual.PerceptualMetric` returns a scalar distance (0 = identical,
larger = worse):

```
distance = spatial + beta * temporal          # beta = 0.5 by default
```

**Spatial — `metrics/msssim.py`.** Pure-numpy Multi-Scale SSIM (Wang et al. 2003)
over luminance. A separable Gaussian window is applied as a 'valid' convolution;
the number of scales is reduced automatically for small frames, falling back to a
single global-statistics SSIM below the window size. `spatial = 1 - mean_frames(MS-SSIM)`.

**Temporal — `metrics/temporal.py`.** GIF quantization artifacts are largely
*temporal*: palette banding shimmers between frames in regions the source held
still. We compare the candidate's consecutive-frame luma deltas to the
reference's, weighting toward pixels the source kept static (`w_still =
exp(-|Δref|/τ)`). Genuine motion (where the source itself changes) is not
penalized — motion masking is respected.

Both terms composite RGBA over a fixed background first, so 1-bit GIF transparency
does not pollute the score. The candidate is resized to the reference resolution,
so the resolution lever is handled transparently.

`DistanceResult` also exposes `per_frame` distances and `worst_frame`, which feed
the report's "where did the loss land" (`loss_locus`).

## The "invisible" threshold

`PerceptualMetric.invisible_threshold` (default `0.005`) defines perceptually
lossless: `distance ≤ threshold`. **It is a calibration target, not a physical
constant.** The M0 benchmark table is the instrument used to tune it: run the
corpus, find the distance at which artifacts become humanly visible per category,
and adjust `beta` / the threshold. This is a standing activity, not a one-time
step.

## Guarding against a gameable metric (spec §9)

- Periodically check the metric's rankings against real human ratings on a
  held-out sample (small pairwise preference studies).
- Trust the metric only to the extent it agrees with humans; disagreement is a
  defect to fix, not a number to chase.
- Reference metrics (MS-SSIM, and optionally `ssimulacra2` / `butteraugli` via
  `metrics/external.py` when those binaries are present) provide sanity checks.

## How M2 plugs in

The learned metric implements `Metric.distance(reference, candidate) ->
DistanceResult` and supplies its own `invisible_threshold`, then registers in
`metrics/__init__.py::default_metric`. No other code changes: `encode()` and the
bench consume whatever the registry returns. Keep the temporal dimension — the
artifacts that matter most in animated GIF are across time, not within a frame.
