"""Train JudgeNet on the labeled pairs (CPU, pairwise ranking + identity anchor).

Loss: a weighted hinge that wants ``score(farther) > score(closer) + m0`` (pairs
weighted by their derived confidence ``margin``), plus an identity anchor pushing
``net(features(ref, ref)) -> 0`` so the scalar is calibrated near zero for lossless.
Clips are split train/val so no clip leaks across the split. We save the best model
by validation pair-ranking accuracy and calibrate ``invisible_threshold`` from the
identity-score distribution.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from ..judge_features import features
from .model import IN_CH, JudgeNet


def _load_stacks(ds_dir: str) -> dict:
    stacks: dict = {}
    for p in glob.glob(os.path.join(ds_dir, "ref", "*.npy")):
        clip = os.path.splitext(os.path.basename(p))[0]
        stacks[(clip, "__ref__")] = np.load(p)
    for p in glob.glob(os.path.join(ds_dir, "var", "*.npy")):
        clip, vid = os.path.splitext(os.path.basename(p))[0].split("__", 1)
        stacks[(clip, vid)] = np.load(p)
    return stacks


def _precompute_feats(stacks: dict) -> dict:
    refs = {c: arr for (c, v), arr in stacks.items() if v == "__ref__"}
    return {(clip, vid): torch.from_numpy(features(refs[clip], arr))
            for (clip, vid), arr in stacks.items()}


class Pairs(Dataset):
    def __init__(self, pairs: list[dict], feats: dict) -> None:
        self.pairs, self.feats = pairs, feats

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, i: int):
        p = self.pairs[i]
        fa, fb = self.feats[(p["clip"], p["a"])], self.feats[(p["clip"], p["b"])]
        close, far = (fa, fb) if p["label"] == 0 else (fb, fa)
        return close, far, torch.tensor(p["margin"], dtype=torch.float32)


def run(ds_dir: str, out_dir: str, *, epochs: int, batch: int, lr: float, seed: int) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.set_num_threads(4)

    meta = json.load(open(os.path.join(ds_dir, "meta.json")))
    pairs = [json.loads(line) for line in open(os.path.join(ds_dir, "pairs.jsonl"))]
    feats = _precompute_feats(_load_stacks(ds_dir))

    clips = sorted(meta["clips"].keys())
    random.Random(seed).shuffle(clips)
    n_val = max(1, len(clips) // 5)
    val_clips = set(clips[:n_val])
    tr = [p for p in pairs if p["clip"] not in val_clips]
    va = [p for p in pairs if p["clip"] in val_clips]
    print(f"clips: {len(clips) - n_val} train / {n_val} val ; pairs: {len(tr)} train / {len(va)} val")

    train_clips = [c for c in clips if c not in val_clips]
    anchors = torch.stack([feats[(c, "__ref__")] for c in train_clips])
    dl = DataLoader(Pairs(tr, feats), batch_size=batch, shuffle=True, num_workers=0)

    net = JudgeNet()
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    m0, lam = 0.1, 0.3

    def evaluate(subset: list[dict]) -> tuple[float, float]:
        net.eval()
        correct = dvb_c = dvb_n = 0
        with torch.no_grad():
            for p in subset:
                sa = net(feats[(p["clip"], p["a"])][None]).item()
                sb = net(feats[(p["clip"], p["b"])][None]).item()
                close, far = (sa, sb) if p["label"] == 0 else (sb, sa)
                ok = int(far > close)
                correct += ok
                if p["family"] == "dither_vs_band":
                    dvb_n += 1
                    dvb_c += ok
        net.train()
        return correct / max(1, len(subset)), (dvb_c / dvb_n if dvb_n else float("nan"))

    best = {"acc": -1.0, "dvb": float("nan"), "ep": 0}
    os.makedirs(out_dir, exist_ok=True)
    ckpt = os.path.join(out_dir, "judgenet.pt")
    for ep in range(epochs):
        net.train()
        tot = 0.0
        for close, far, w in dl:
            opt.zero_grad()
            sc, sf = net(close), net(far)
            rank = (torch.relu(m0 - (sf - sc)) * w).mean()
            idx = torch.randint(0, anchors.shape[0], (min(batch, anchors.shape[0]),))
            anchor = (net(anchors[idx]) ** 2).mean()
            loss = rank + lam * anchor
            loss.backward()
            opt.step()
            tot += loss.item()
        acc, dvb = evaluate(va)
        print(f"epoch {ep + 1:02d}  loss {tot / max(1, len(dl)):.4f}  "
              f"val_pair_acc {acc:.3f}  dither_vs_band {dvb:.3f}")
        if acc >= best["acc"]:
            best = {"acc": acc, "dvb": dvb, "ep": ep + 1}
            torch.save(net.state_dict(), ckpt)

    # Calibrate invisible_threshold from the identity-score distribution.
    net.load_state_dict(torch.load(ckpt))
    net.eval()
    with torch.no_grad():
        id_scores = net(torch.stack([feats[(c, "__ref__")] for c in clips])).numpy()
    thr = float(np.percentile(id_scores, 95)) + 1e-4
    jm = {"invisible_threshold": thr, "proxy": meta["proxy"], "t_samples": meta["t_samples"],
          "in_channels": IN_CH, "val_pair_acc": best["acc"], "val_dither_vs_band": best["dvb"],
          "best_epoch": best["ep"], "identity_score_mean": float(id_scores.mean()),
          "n_pairs": len(pairs), "n_clips": len(clips)}
    json.dump(jm, open(os.path.join(out_dir, "judgenet.meta.json"), "w"), indent=2)
    print(f"\nsaved {ckpt}  (best val_pair_acc={best['acc']:.3f} @ epoch {best['ep']}, "
          f"dither_vs_band={best['dvb']:.3f})  invisible_threshold={thr:.4f}")
    return jm


def main() -> None:
    ap = argparse.ArgumentParser(description="Train the learned judge (CPU).")
    ap.add_argument("--data", default="bench/corpus/dataset")
    ap.add_argument("--out", default="encoder/metrics/models")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    run(args.data, args.out, epochs=args.epochs, batch=args.batch, lr=args.lr, seed=args.seed)


if __name__ == "__main__":
    main()
