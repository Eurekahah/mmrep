"""画扰动示例图：每个模型一张，行=样本，列=clean + 三档噪声 + 三档遮挡（共 7 列），画的是模型实际看到的画面。

用法：python -m tools.make_perturb_examples --split coco --n 8 --out docs
需要真实图片已放在 data_root 下（见 configs/paths.yaml）。
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from data.config import get_paths, load_models_cfg
from data.datasets import ImageTextDataset
from data.perturb import ALL_MODES

TITLE = {"clean": "clean", "noise_lo": "noise s=0.05", "noise": "noise s=0.10", "noise_hi": "noise s=0.20",
         "occl_lo": "occl 20%", "occl": "occl 30%", "occl_hi": "occl 40%"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="coco", choices=["flickr", "coco"])
    ap.add_argument("--split", default="val")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--out", default="docs")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    cfg = load_models_cfg()
    for model in ("clip", "siglip"):
        # 画图时不做归一化，直接取 resize+扰动后的 [0,1] 图像
        from data.datasets import load_split
        from data.preprocess import load_resized_01
        from data.perturb import perturb
        from PIL import Image
        root, _ = get_paths()
        meta = load_split(a.name)
        d = os.path.join(root, "flickr30k", "flickr30k-images") if a.name == "flickr" else os.path.join(root, "coco", "val2017")
        ids = meta[a.split][:a.n]
        fig, axs = plt.subplots(a.n, 7, figsize=(14, 2 * a.n))
        for r, id_ in enumerate(ids):
            x = load_resized_01(Image.open(os.path.join(d, id_)), model)
            for c, mode in enumerate(ALL_MODES):
                axs[r, c].imshow(perturb(x, mode, r).permute(1, 2, 0).clamp(0, 1).numpy())
                axs[r, c].axis("off")
                if r == 0:
                    axs[r, c].set_title(TITLE[mode], fontsize=9)
        fig.suptitle(f"{model}: input {cfg[model]['input_size']}x{cfg[model]['input_size']}, {cfg[model]['resize']['mode']}", fontsize=11)
        plt.tight_layout()
        path = os.path.join(a.out, f"perturb_examples_{model}.png")
        plt.savefig(path, dpi=110)
        plt.close(fig)
        print("wrote", path)


if __name__ == "__main__":
    main()
