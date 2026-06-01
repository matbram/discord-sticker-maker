"""LearnedMetric (M2) — the learned, motion-aware judge, run via onnxruntime.

Loads ``models/judgenet.onnx`` and returns the same ``DistanceResult`` shape as
``PerceptualMetric`` so it is a drop-in behind the ``Metric`` interface — no torch
at encode time. ``distance`` is the network's learned scalar (on the downscaled
proxy, T-sampled); ``per_frame``/``worst_frame`` are a cheap full-length luma-error
localization (reusing the perceptual luma utilities) so the honesty report's
``loss_locus`` still points at a real source frame.

It is OFF by default: the registry only returns it on explicit opt-in
(``FOVEA_METRIC=learned`` or ``--metric learned``), with MS-SSIM as the safe
default and a graceful fallback if the model or onnxruntime is missing.
"""
from __future__ import annotations

import json
import os

import numpy as np

from ..core.frames import Frames
from .base import DistanceResult, Metric
from .judge_features import N_CHANNELS, T_SAMPLES, features, to_proxy_stack
from .perceptual import _resize_luma_stack, composite_luma_stack

_MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
DEFAULT_ONNX = os.path.join(_MODEL_DIR, "judgenet.onnx")
DEFAULT_META = os.path.join(_MODEL_DIR, "judgenet.meta.json")


def model_available(path: str = DEFAULT_ONNX) -> bool:
    """True iff the ONNX model exists and onnxruntime can be imported."""
    if not os.path.exists(path):
        return False
    try:
        import onnxruntime  # noqa: F401
    except Exception:  # noqa: BLE001
        return False
    return True


class LearnedMetric(Metric):
    name = "learned"

    def __init__(self, model_path: str = DEFAULT_ONNX, meta_path: str = DEFAULT_META) -> None:
        import onnxruntime as ort

        self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self._input = self.session.get_inputs()[0].name
        self.invisible_threshold = 0.05
        try:
            with open(meta_path) as fh:
                self.invisible_threshold = float(json.load(fh).get("invisible_threshold", 0.05))
        except Exception:  # noqa: BLE001
            pass

    def _score(self, feat: np.ndarray) -> float:
        x = feat.reshape(1, N_CHANNELS * T_SAMPLES, feat.shape[-2], feat.shape[-1]).astype(np.float32)
        out = self.session.run(None, {self._input: x})[0]
        return float(np.asarray(out).reshape(-1)[0])

    def distance(self, reference: Frames, candidate: Frames) -> DistanceResult:
        rf, cf = reference.frames, candidate.frames
        n = min(len(rf), len(cf))
        rf, cf = rf[:n], cf[:n]

        # Learned scalar on the T-sampled proxy stack.
        dist = self._score(features(to_proxy_stack(rf), to_proxy_stack(cf)))

        # Cheap full-length luma-error per frame for localization (reuses perceptual utils).
        ref_l = composite_luma_stack(rf)
        cand_l = _resize_luma_stack(composite_luma_stack(cf), ref_l.shape[1:])
        per = (np.abs(ref_l - cand_l).reshape(n, -1).mean(axis=1) / 255.0)
        worst = int(np.argmax(per)) if per.size else 0

        return DistanceResult(
            distance=dist, per_frame=per.tolist(), spatial=dist, temporal=0.0,
            worst_frame=worst, extra={"metric": "learned", "model": "judgenet"},
        )
