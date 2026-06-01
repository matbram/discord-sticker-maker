"""Export the trained JudgeNet to ONNX and verify torch/onnxruntime parity.

The runtime ``LearnedMetric`` loads only the ``.onnx`` via onnxruntime, so the
encode-time image needs no torch. The model is exported with a flattened
``(batch, IN_CH, PROXY, PROXY)`` input (batch axis dynamic).
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from ..judge_features import PROXY
from .model import IN_CH, JudgeNet


def export(ckpt: str, out_dir: str) -> str:
    net = JudgeNet()
    net.load_state_dict(torch.load(ckpt, map_location="cpu"))
    net.eval()

    onnx_path = os.path.join(out_dir, "judgenet.onnx")
    dummy = torch.zeros(1, IN_CH, PROXY, PROXY)
    torch.onnx.export(
        net, dummy, onnx_path, input_names=["x"], output_names=["distance"],
        dynamic_axes={"x": {0: "batch"}, "distance": {0: "batch"}}, opset_version=17,
        dynamo=False,   # use the legacy TorchScript exporter (no onnxscript dependency)
    )

    import onnxruntime as ort

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    x = np.random.RandomState(0).randn(3, IN_CH, PROXY, PROXY).astype(np.float32)
    with torch.no_grad():
        t = net(torch.from_numpy(x)).numpy().reshape(-1)
    o = np.asarray(sess.run(None, {"x": x})[0]).reshape(-1)
    err = float(np.max(np.abs(t - o)))
    print(f"exported {onnx_path}  (torch-vs-onnx max abs err {err:.2e})")
    if err >= 1e-4:
        raise SystemExit(f"ONNX parity check failed: {err}")
    return onnx_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Export JudgeNet to ONNX.")
    ap.add_argument("--ckpt", default="encoder/metrics/models/judgenet.pt")
    ap.add_argument("--out", default="encoder/metrics/models")
    args = ap.parse_args()
    export(args.ckpt, args.out)


if __name__ == "__main__":
    main()
