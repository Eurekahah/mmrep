"""Train a linear CIFAR-100 probe on cached frozen image embeddings."""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from data.config import REPO_ROOT
from tools.logger import RunLogger

MODELS = ("clip", "siglip")
SETTINGS = ("pretrained", "lit", "lora_r8", "lora_r16")
REPO_PATH = Path(REPO_ROOT)


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load(path: Path):
    with np.load(path, allow_pickle=False) as data:
        features = torch.from_numpy(data["img_raw"].astype(np.float32, copy=False))
        labels = torch.from_numpy(data["labels"].astype(np.int64, copy=False))
    features = torch.nn.functional.normalize(features, dim=1)
    return features, labels


def _accuracy(logits, labels):
    top1 = (logits.argmax(dim=1) == labels).sum().item()
    top5 = logits.topk(5, dim=1).indices.eq(labels[:, None]).any(dim=1).sum().item()
    return top1, top5


def run_probe(model: str, setting: str, seed: int, *, feature_root=REPO_PATH / "features_cifar",
              device="cuda", batch_size=1024, epochs=100, lr=1e-3,
              weight_decay=1e-4, num_workers=0):
    if model not in MODELS or setting not in SETTINGS:
        raise ValueError("invalid model or setting")
    seed_everything(seed)
    target = torch.device(device)
    if target.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; use --device cpu")
    base = Path(feature_root) / model / setting
    train_x, train_y = _load(base / "train.npz")
    test_x, test_y = _load(base / "test.npz")
    if train_x.ndim != 2 or train_x.shape[1] <= 0:
        raise ValueError("invalid feature shape")
    if len(train_x) != 50000 or len(test_x) != 10000:
        raise ValueError("CIFAR-100 train/test feature counts must be 50000/10000")
    train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=batch_size,
                              shuffle=True, generator=torch.Generator().manual_seed(seed),
                              num_workers=num_workers, pin_memory=target.type == "cuda")
    test_loader = DataLoader(TensorDataset(test_x, test_y), batch_size=batch_size,
                             shuffle=False, num_workers=num_workers,
                             pin_memory=target.type == "cuda")
    linear = nn.Linear(train_x.shape[1], 100).to(target)
    optimizer = torch.optim.AdamW(linear.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    final_loss = None
    with RunLogger("probe_cifar100", model=model, setting=setting, seed=seed,
                   track_cuda=target.type == "cuda", epochs=epochs,
                   probe_batch_size=batch_size, lr=lr, weight_decay=weight_decay) as log:
        for _epoch in range(epochs):
            linear.train()
            total_loss, total = 0.0, 0
            for x, y in train_loader:
                x, y = x.to(target, non_blocking=True), y.to(target, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(linear(x), y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(y)
                total += len(y)
            final_loss = total_loss / total
        linear.eval()
        correct1 = correct5 = total = 0
        with torch.inference_mode():
            for x, y in test_loader:
                top1, top5 = _accuracy(linear(x.to(target)), y.to(target))
                correct1 += top1
                correct5 += top5
                total += len(y)
        result = {"model": model, "setting": setting, "seed": seed,
                  "top1": correct1 / total, "top5": correct5 / total,
                  "final_loss": final_loss, "n_test": total}
        log.metric("top1", result["top1"])
        log.metric("top5", result["top5"])
        log.metric("final_loss", final_loss)
    return result


def write_results(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Keep the handoff CSV compatible with the group's agreed schema.  Loss
    # and test-count remain available in the per-run JSON log.
    fields = ["model", "setting", "seed", "top1", "top5"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row[key] for key in fields} for row in rows)


def write_summary(rows, path):
    """Write mean/std across the requested seeds for P8 and report tables."""
    grouped = {}
    for row in rows:
        grouped.setdefault((row["model"], row["setting"]), []).append(row)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["model", "setting", "top1_mean", "top1_std", "top5_mean", "top5_std"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for (model, setting), values in sorted(grouped.items()):
            top1 = np.asarray([r["top1"] for r in values], dtype=np.float64)
            top5 = np.asarray([r["top5"] for r in values], dtype=np.float64)
            writer.writerow({"model": model, "setting": setting,
                             "top1_mean": top1.mean(), "top1_std": top1.std(ddof=0),
                             "top5_mean": top5.mean(), "top5_std": top5.std(ddof=0)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=(*MODELS, "all"), default="all")
    parser.add_argument("--setting", choices=(*SETTINGS, "all"), default="all")
    parser.add_argument("--feature-root", default=str(REPO_PATH / "features_cifar"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--out", default=str(REPO_PATH / "results" / "cifar100_top1.csv"))
    parser.add_argument("--summary-out", default=str(REPO_PATH / "results" / "cifar100_summary.csv"))
    args = parser.parse_args()
    models = MODELS if args.model == "all" else (args.model,)
    settings = SETTINGS if args.setting == "all" else (args.setting,)
    rows = []
    for model in models:
        for setting in settings:
            for seed in args.seeds:
                rows.append(run_probe(model, setting, seed, feature_root=args.feature_root,
                                      device=args.device, batch_size=args.batch_size,
                                      epochs=args.epochs))
                write_results(rows, args.out)
                write_summary(rows, args.summary_out)
                print(json.dumps(rows[-1], ensure_ascii=False))
    write_results(rows, args.out)
    write_summary(rows, args.summary_out)


if __name__ == "__main__":
    main()
