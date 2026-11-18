"""Extract frozen CLIP/SigLIP image features for CIFAR-100.

Examples:
    python -m extract.extract_cifar --model clip --setting pretrained --device cuda
    python -m extract.extract_cifar --model siglip --setting all --data-root /data/mmrep/cifar100

The image tower is never trained here.  Raw embeddings are stored as float16
to keep the cache small; the probe casts them to float32 before optimization.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CIFAR100

from data.config import REPO_ROOT, get_paths
from data.preprocess import load_resized_01, normalize
from models.loader import SETTINGS, load_model
from tools.logger import RunLogger

MODELS = ("clip", "siglip")
REPO_PATH = Path(REPO_ROOT)


class _CifarImages(Dataset):
    def __init__(self, base: CIFAR100, model: str):
        self.base, self.model = base, model

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        image, label = self.base[index]
        x01 = load_resized_01(image, self.model)
        return normalize(x01, self.model), int(label)


def _amp_context(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    from contextlib import nullcontext
    return nullcontext()


def _class_prompts():
    path = REPO_PATH / "splits" / "cifar100_classes.json"
    with path.open("r", encoding="utf-8") as f:
        meta = json.load(f)
    prompts = meta.get("prompts")
    if prompts is None:
        prompts = [f"a photo of a {name.replace('_', ' ')}" for name in meta["classes"]]
    if len(prompts) != 100:
        raise ValueError(f"Expected 100 CIFAR-100 prompts, got {len(prompts)}")
    return prompts


def extract(model: str, setting: str, *, data_root: str | os.PathLike,
            out_root: str | os.PathLike = REPO_PATH / "features_cifar",
            device: str = "cuda", batch_size: int = 512,
            num_workers: int = 0, amp: bool = True, cache_dir: str | None = None,
            local_files_only: bool = False):
    if model not in MODELS:
        raise ValueError(f"model must be one of {MODELS}")
    if setting not in SETTINGS:
        raise ValueError(f"setting must be one of {SETTINGS}")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    target = torch.device(device)
    if target.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; use --device cpu")
    wrapper = load_model(model, setting, device=str(target), cache_dir=cache_dir,
                         local_files_only=local_files_only)
    wrapper.eval()
    root = Path(data_root)
    out_dir = Path(out_root) / model / setting
    out_dir.mkdir(parents=True, exist_ok=True)
    all_features = {}
    for split, train in (("train", True), ("test", False)):
        base = CIFAR100(root=str(root), train=train, download=False)
        ds = _CifarImages(base, model)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=target.type == "cuda")
        chunks, labels = [], []
        with RunLogger("extract_cifar", model=model, setting=setting,
                       extra_split=split, batch_size=batch_size) as log:
            with torch.inference_mode():
                for pixels, y in loader:
                    pixels = pixels.to(target, non_blocking=True)
                    with _amp_context(target, amp):
                        features = wrapper.encode_image(pixels)
                    chunks.append(features.float().cpu().numpy())
                    labels.append(y.numpy().astype(np.int64, copy=False))
            image_features = np.concatenate(chunks, axis=0).astype(np.float16)
            y_all = np.concatenate(labels, axis=0)
            log.metric("n_images", int(len(y_all)))
            log.metric("embed_dim", int(image_features.shape[1]))
        path = out_dir / f"{split}.npz"
        np.savez(path, img_raw=image_features, labels=y_all)
        all_features[split] = path

    # Text embeddings are generated once per model/setting using the same
    # wrapper, and are kept separate from image train/test caches.
    prompts = _class_prompts()
    text_chunks = []
    with torch.inference_mode():
        for start in range(0, len(prompts), batch_size):
            tokens = wrapper.tokenize(prompts[start:start + batch_size])
            tokens = {k: v.to(target) for k, v in tokens.items()}
            with _amp_context(target, amp):
                text = wrapper.encode_text(**tokens)
            text_chunks.append(text.float().cpu().numpy())
    text_features = np.concatenate(text_chunks, axis=0).astype(np.float16)
    np.savez(out_dir / "class_txt.npz", txt_raw=text_features)
    return {"model": model, "setting": setting,
            "train": str(all_features["train"]), "test": str(all_features["test"]),
            "class_txt": str(out_dir / "class_txt.npz"),
            "n_train": 50000, "n_test": 10000}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--setting", choices=(*SETTINGS, "all"), required=True)
    parser.add_argument("--data-root", default=None,
                        help="CIFAR root containing cifar-100-python; defaults to MMREP_DATA_ROOT/cifar100")
    parser.add_argument("--out-root", default=str(REPO_PATH / "features_cifar"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--cache-dir")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    data_root = args.data_root
    if data_root is None:
        data_root = Path(get_paths()[0]) / "cifar100"
    settings = SETTINGS if args.setting == "all" else (args.setting,)
    for setting in settings:
        result = extract(args.model, setting, data_root=data_root, out_root=args.out_root,
                         device=args.device, batch_size=args.batch_size,
                         num_workers=args.num_workers, amp=not args.no_amp,
                         cache_dir=args.cache_dir, local_files_only=args.local_files_only)
        print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
