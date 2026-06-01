"""M2 training pipeline for the learned perceptual judge.

These modules build a corpus, synthesize labeled (reference, candidate) pairs with
*derivable* severity orderings (no human labels), train a small CPU CNN, and export
it to ONNX. They are NOT imported at encode time — the runtime ``LearnedMetric``
(``encoder/metrics/learned.py``) only needs onnxruntime + numpy. Build order and
commands live in ``docs/STATE.md``/the plan; see each module's docstring.
"""
