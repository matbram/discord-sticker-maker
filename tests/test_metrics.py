"""Property tests for the MS-SSIM + temporal perceptual judge (no images on disk)."""
from __future__ import annotations

import numpy as np
import pytest

from encoder.metrics.msssim import msssim_per_frame
from encoder.metrics.perceptual import PerceptualMetric, composite_luma_stack
from encoder.metrics.temporal import temporal_distance


def test_identity_is_zero_distance(make_frames):
    fr = make_frames(6, 64, 64, seed=1)
    res = PerceptualMetric().distance(fr, fr)
    assert res.distance == pytest.approx(0.0, abs=1e-6)
    assert res.spatial == pytest.approx(0.0, abs=1e-6)
    assert res.temporal == pytest.approx(0.0, abs=1e-9)
    assert res.extra["msssim_mean"] == pytest.approx(1.0, abs=1e-6)


def test_msssim_identity_is_one():
    rng = np.random.default_rng(3)
    stack = rng.integers(0, 256, size=(4, 80, 80)).astype(np.float64)
    vals = msssim_per_frame(stack, stack)
    assert np.allclose(vals, 1.0, atol=1e-6)
    assert np.all(vals >= 0.0) and np.all(vals <= 1.0)


def test_distance_monotonic_in_noise(make_frames, add_noise):
    fr = make_frames(5, 48, 48, seed=2)
    metric = PerceptualMetric()
    d0 = metric.distance(fr, fr).distance
    d1 = metric.distance(fr, add_noise(fr, 8, seed=11)).distance
    d2 = metric.distance(fr, add_noise(fr, 28, seed=11)).distance
    assert d0 < d1 < d2
    assert d0 >= 0.0


def test_worst_frame_points_at_injected_artifact(make_frames):
    fr = make_frames(6, 64, 64, seed=4)
    frames = [f.copy() for f in fr.frames]
    frames[3][..., :3] = 255 - frames[3][..., :3]  # invert frame 3 only
    from encoder.core.frames import frames_from_list

    cand = frames_from_list(frames, 100)
    res = PerceptualMetric().distance(fr, cand)
    assert res.worst_frame == 3
    assert res.per_frame[3] == max(res.per_frame)


def test_temporal_penalizes_flicker_more_than_motion(make_frames):
    static = composite_luma_stack(make_frames(6, 48, 48, seed=5, static=True).frames)
    moving = composite_luma_stack(make_frames(6, 48, 48, seed=5, motion=True).frames)
    rng = np.random.default_rng(9)
    noise = rng.normal(0.0, 12.0, static.shape)
    t_static, _ = temporal_distance(static, static + noise)
    t_motion, _ = temporal_distance(moving, moving + noise)
    assert t_static > 0.0
    assert t_static > t_motion        # shimmer on still regions is penalized harder
    # identical motion (no added noise) is not penalized at all
    assert temporal_distance(moving, moving)[0] == pytest.approx(0.0, abs=1e-9)


def test_tiny_frames_do_not_crash(make_frames, add_noise):
    metric = PerceptualMetric()
    tiny = make_frames(3, 8, 8, seed=6)
    assert metric.distance(tiny, tiny).distance == pytest.approx(0.0, abs=1e-6)
    assert metric.distance(tiny, add_noise(tiny, 20, seed=1)).distance > 0.0
    two = make_frames(2, 2, 2, seed=6)              # below window -> global SSIM path
    assert 0.0 <= metric.distance(two, add_noise(two, 20)).distance


def test_resolution_mismatch_is_handled(make_frames):
    """Candidate smaller than reference (resolution lever) must still score."""
    from encoder.core.frames import frames_from_list
    from PIL import Image

    ref = make_frames(4, 64, 64, seed=8)
    small = []
    for fr in ref.frames:
        im = Image.fromarray(fr, "RGBA").resize((32, 32), Image.BILINEAR)
        small.append(np.asarray(im, dtype=np.uint8))
    cand = frames_from_list(small, 100)
    res = PerceptualMetric().distance(ref, cand)
    assert 0.0 <= res.distance < 1.0
