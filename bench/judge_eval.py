"""Acceptance harness for the perceptual judge (M2).

Evaluates any ``Metric`` by pair-ranking accuracy on the labeled dataset: for a
pair ``(ref, a, b)`` with a known closer side, the metric is "correct" when it
assigns the smaller distance to the closer candidate. The HEADLINE is the
``dither_vs_band`` family (equal low palette, dithered vs banded) — the exact case
MS-SSIM gets backwards. Reports held-out (generalization) and all-pairs numbers.

This is NOT a human study; it measures agreement with the synthetic + lever oracle
only. Human-preference agreement remains the outstanding gate before the learned
judge could become the default.
"""
from __future__ import annotations

import glob
import json
import os
import random

import numpy as np

FAMILY_ORDER = ["_all", "dither_vs_band", "banding", "flicker", "choppiness", "blur", "lever"]


def _load_dataset(ds_dir: str):
    stacks: dict = {}
    for p in glob.glob(os.path.join(ds_dir, "ref", "*.npy")):
        stacks[(os.path.splitext(os.path.basename(p))[0], "__ref__")] = np.load(p)
    for p in glob.glob(os.path.join(ds_dir, "var", "*.npy")):
        clip, vid = os.path.splitext(os.path.basename(p))[0].split("__", 1)
        stacks[(clip, vid)] = np.load(p)
    pairs = [json.loads(line) for line in open(os.path.join(ds_dir, "pairs.jsonl"))]
    meta = json.load(open(os.path.join(ds_dir, "meta.json")))
    return stacks, pairs, meta


def _frames(stack: np.ndarray):
    from encoder.core.frames import frames_from_list

    n = int(stack.shape[0])
    return frames_from_list([stack[i] for i in range(n)], [80] * n)


def _val_clips(meta: dict, seed: int = 0) -> set:
    clips = sorted(meta["clips"].keys())
    random.Random(seed).shuffle(clips)
    return set(clips[:max(1, len(clips) // 5)])


def _eval_metric(metric, pairs: list[dict], stacks: dict) -> dict:
    cache: dict = {}

    def dist(clip: str, vid: str) -> float:
        key = (clip, vid)
        if key not in cache:
            cache[key] = metric.distance(_frames(stacks[(clip, "__ref__")]), _frames(stacks[key])).distance
        return cache[key]

    tally: dict = {}
    for p in pairs:
        da, db = dist(p["clip"], p["a"]), dist(p["clip"], p["b"])
        pred = "a" if da < db else "b"
        truth = "a" if p["label"] == 0 else "b"
        ok = int(pred == truth)
        for key in ("_all", p["family"]):
            t = tally.setdefault(key, [0, 0])
            t[0] += ok
            t[1] += 1
    return {k: {"acc": (c / n if n else float("nan")), "n": n} for k, (c, n) in tally.items()}


def run_judge_eval(metric_names: list[str], ds_dir: str, out_dir: str) -> dict:
    from encoder.metrics import get_metric

    stacks, pairs, meta = _load_dataset(ds_dir)
    val = _val_clips(meta)
    splits = {"held_out": [p for p in pairs if p["clip"] in val], "all": pairs}

    results: dict = {}
    for name in metric_names:
        try:
            metric = get_metric(name)
        except Exception as exc:  # noqa: BLE001
            print(f"  metric '{name}' unavailable: {exc}")
            continue
        results[name] = {split: _eval_metric(metric, ps, stacks) for split, ps in splits.items()}

    os.makedirs(out_dir, exist_ok=True)
    payload = {
        "val_clips": sorted(val), "n_held_out_pairs": len(splits["held_out"]),
        "n_all_pairs": len(splits["all"]),
        "dataset_meta": {k: meta[k] for k in ("n_clips", "n_pairs", "proxy", "t_samples")},
        "results": results,
    }
    with open(os.path.join(out_dir, "judge_eval.json"), "w") as fh:
        json.dump(payload, fh, indent=2)
    md = _format_md(payload)
    with open(os.path.join(out_dir, "judge_eval.md"), "w") as fh:
        fh.write(md)
    print(md)
    return payload


def _table(results: dict, split: str) -> list[str]:
    fams = [f for f in FAMILY_ORDER if any(f in r.get(split, {}) for r in results.values())]
    lines = ["| metric | " + " | ".join(fams) + " |", "|" + "---|" * (len(fams) + 1)]
    for name, by_split in results.items():
        r = by_split.get(split, {})
        cells = [name]
        for f in fams:
            cells.append(f"{r[f]['acc'] * 100:.0f}% (n={r[f]['n']})" if f in r else "—")
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def _format_md(payload: dict) -> str:
    out = ["# Judge evaluation — pair-ranking accuracy", "",
           f"Held-out clips: {payload['val_clips']}  "
           f"({payload['n_held_out_pairs']} held-out pairs / {payload['n_all_pairs']} total).", "",
           "## Held-out clips (generalization)"]
    out += _table(payload["results"], "held_out")
    out += ["", "## All pairs", *_table(payload["results"], "all"), "",
            "**Headline:** on `dither_vs_band` (equal low palette, dithered vs banded) MS-SSIM is",
            "expected at/below chance — it prefers the *banded* frame — while the learned judge",
            "should score high. That is the blind spot M2 exists to fix.", "",
            "> Human-preference validation was NOT performed (no labelers in this environment).",
            "> These numbers measure agreement with the synthetic + lever oracle only; human",
            "> agreement on real content remains the outstanding gate before the learned judge",
            "> can become Fovea's default metric."]
    return "\n".join(out) + "\n"
