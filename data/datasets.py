"""统一数据接口：get_dataset(name, split, cond, model)。

每个样本是 dict：
  image     [3,S,S]  已按 model 的官方方式 resize、按 cond 扰动、归一化（CLIP S=224，SigLIP2 S=256）
  captions  list[str]  图文数据集为 5 条，CIFAR-100 为空
  idx       int  样本在该 split 内的序号（也是扰动的随机种子偏移）
  id        str  图片文件名（CIFAR 为序号字符串）
  label     int  类别（Flickr 为 -1）
"""
import json
import os

import torch
from PIL import Image
from torch.utils.data import Dataset

from .config import get_paths, load_models_cfg, load_perturb_cfg
from .perturb import STRENGTH_MODES, ALL_MODES, perturb
from .preprocess import load_resized_01, normalize

IMAGE_TEXT = ("flickr", "coco")
NAMES = ("flickr", "coco", "cifar100")


def image_dir(name, data_root):
    if name == "flickr":
        return os.path.join(data_root, "flickr30k", "flickr30k-images")
    if name == "coco":
        return os.path.join(data_root, "coco", "val2017")
    raise ValueError(name)


def load_split(name, splits_dir=None):
    _, default_splits = get_paths()
    splits_dir = splits_dir or default_splits
    with open(os.path.join(splits_dir, f"{name}_split.json"), "r", encoding="utf-8") as f:
        return json.load(f)


def check_cond(name, split, cond):
    if cond not in ALL_MODES:
        raise ValueError(f"unknown cond {cond}, expected one of {ALL_MODES}")
    if cond in STRENGTH_MODES:
        scope = [tuple(x) for x in load_perturb_cfg()["strength_scope"]]
        if (name, split) not in scope:
            raise ValueError(f"cond {cond} 只在 {scope} 上使用，不支持 ({name}, {split})")


class ImageTextDataset(Dataset):
    def __init__(self, name, split, cond="clean", model="clip",
                 data_root=None, splits_dir=None, cifar_download=False):
        assert name in NAMES, name
        assert model in load_models_cfg(), model
        check_cond(name, split, cond)
        self.name, self.split, self.cond, self.model = name, split, cond, model
        dr, sd = get_paths()
        self.data_root = data_root or dr
        self.splits_dir = splits_dir or sd
        if name in IMAGE_TEXT:
            self.meta = load_split(name, self.splits_dir)
            if split not in self.meta or len(self.meta[split]) == 0:
                raise ValueError(f"{name} 没有 {split} 划分（COCO 只有 val/test）")
            self.ids = list(self.meta[split])
            self.img_dir = image_dir(name, self.data_root)
        else:
            from torchvision.datasets import CIFAR100
            self.cifar = CIFAR100(root=os.path.join(self.data_root, "cifar100"),
                                  train=(split == "train"), download=cifar_download)
            self.ids = [str(i) for i in range(len(self.cifar))]

    def __len__(self):
        return len(self.ids)

    def _pil(self, i):
        if self.name == "cifar100":
            return self.cifar[i][0]
        return Image.open(os.path.join(self.img_dir, self.ids[i]))

    def __getitem__(self, i):
        pil = self._pil(i)
        x = load_resized_01(pil, self.model)
        x = perturb(x, self.cond, i)
        x = normalize(x, self.model)
        if self.name == "cifar100":
            caps, label = [], int(self.cifar.targets[i])
        else:
            id_ = self.ids[i]
            caps = list(self.meta["captions"][id_])
            label = int(self.meta["labels"][id_])
        return {"image": x, "captions": caps, "idx": i, "id": self.ids[i], "label": label}


def get_dataset(name, split, cond="clean", model="clip", **kw):
    return ImageTextDataset(name, split, cond, model, **kw)


def collate_fn(batch):
    return {
        "image": torch.stack([b["image"] for b in batch]),
        "captions": [b["captions"] for b in batch],
        "idx": torch.tensor([b["idx"] for b in batch], dtype=torch.long),
        "id": [b["id"] for b in batch],
        "label": torch.tensor([b["label"] for b in batch], dtype=torch.long),
    }
