"""Tests for the M2 learned judge. Skipped cleanly when judgenet.onnx is absent
(mirrors test_integration_smoke's gating), so CI without the model still passes."""
import numpy as np
import pytest

from encoder.metrics import available_metrics, default_metric, get_metric
from encoder.metrics.learned import model_available

pytestmark = pytest.mark.skipif(not model_available(), reason="judgenet.onnx not built")


def _frames(seed: int, banded: bool = False, n: int = 5, hw: int = 64):
    base = np.linspace(0, 255, hw, dtype=np.float32)
    out = []
    for i in range(n):
        grad = np.tile((base + i * 5) % 256, (hw, 1))
        rgb = np.stack([grad, grad * 0.6, 255 - grad], -1)
        if banded:
            rgb = np.round(rgb / 64.0) * 64.0      # heavy posterize -> banding
        a = np.full((hw, hw, 1), 255, np.float32)
        out.append(np.clip(np.dstack([rgb, a]), 0, 255).astype(np.uint8))
    return out


def test_learned_registered_and_loads():
    assert "learned" in available_metrics()
    m = get_metric("learned")
    assert m.name == "learned"
    assert m.invisible_threshold > 0


def test_distance_shape_and_banding_order():
    from encoder.core.frames import frames_from_list

    m = get_metric("learned")
    ref = frames_from_list(_frames(1), [80] * 5)
    res = m.distance(ref, ref)
    assert res.distance >= 0.0
    assert len(res.per_frame) == 5
    # A heavily-banded candidate must not be judged closer than the source itself.
    banded = frames_from_list(_frames(1, banded=True), [80] * 5)
    assert res.distance <= m.distance(ref, banded).distance + 1e-6


def test_default_metric_is_opt_in(monkeypatch):
    monkeypatch.delenv("FOVEA_METRIC", raising=False)
    assert default_metric().name != "learned"        # MS-SSIM stays the default
    monkeypatch.setenv("FOVEA_METRIC", "learned")
    assert default_metric().name == "learned"         # explicit opt-in switches it on
