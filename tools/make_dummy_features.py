"""按 CONTRACT 3.7 的格式造假特征，让下游（P5/P7/P8/P9）在真实特征产出前就能开发和自测。

假特征带一点图文配对结构和模态间隙，并按设置/扰动做了“合理方向”的变化，仅用于调通代码，
任何指标数值都没有科学意义，禁止写进报告。

用法：python -m tools.make_dummy_features --out features_dummy [--n 200]
"""
import argparse
import os

import numpy as np

MODELS = {"clip": 512, "siglip": 768}
SETTINGS = ["pretrained", "lit", "lora_r8", "lora_r16"]
DATASETS = ["flickr", "coco"]
SPLITS = ["val", "test"]
MAIN_CONDS = ["clean", "noise", "occl"]
STRENGTH_CONDS = ["noise_lo", "noise_hi", "occl_lo", "occl_hi"]
NOISE_SCALE = {"clean": 0.0, "noise_lo": 0.15, "noise": 0.3, "noise_hi": 0.6,
               "occl_lo": 0.1, "occl": 0.2, "occl_hi": 0.4}
SETTING_SHIFT = {"pretrained": (0.0, 0.0), "lit": (0.0, 0.25),
                 "lora_r8": (0.2, 0.3), "lora_r16": (0.35, 0.45)}   # (图像漂移, 文本漂移)


def _base(model, dataset, split, n, dim, seed=42):
    rs = np.random.RandomState(seed + sum(map(ord, model + dataset + split)))
    d = 64
    z = rs.randn(n, d).astype(np.float32)
    A = rs.randn(d, dim).astype(np.float32) / np.sqrt(d)
    B = A + 0.3 * rs.randn(d, dim).astype(np.float32) / np.sqrt(d)
    gap = rs.randn(dim).astype(np.float32) * 0.6
    return rs, z, A, B, gap


def make(out, n=200, seed=42):
    count = 0
    for model, dim in MODELS.items():
        for dataset in DATASETS:
            for split in SPLITS:
                rs, z, A, B, gap = _base(model, dataset, split, n, dim, seed)
                ids = np.array([f"{dataset}_{split}_{i:05d}.jpg" for i in range(n)])
                labels = (rs.randint(0, 80, n) if dataset == "coco" else -np.ones(n)).astype(np.int64)
                img_clean = z @ A + 0.1 * rs.randn(n, dim).astype(np.float32)
                txt = np.repeat(z, 5, axis=0) @ B + 0.4 * rs.randn(5 * n, dim).astype(np.float32) + gap
                txt2img = np.repeat(np.arange(n), 5).astype(np.int64)
                conds = MAIN_CONDS + (STRENGTH_CONDS if (dataset, split) == ("flickr", "val") else [])
                for setting in SETTINGS:
                    di, dt = SETTING_SHIFT[setting]
                    rs2 = np.random.RandomState(seed + sum(map(ord, model + dataset + split + setting)))
                    img_shift = di * rs2.randn(n, dim).astype(np.float32)
                    txt_shift = dt * rs2.randn(5 * n, dim).astype(np.float32)
                    d_ = os.path.join(out, model, setting)
                    os.makedirs(d_, exist_ok=True)
                    for cond in conds:
                        rs3 = np.random.RandomState(seed + sum(map(ord, cond)))
                        img = img_clean + img_shift + NOISE_SCALE[cond] * rs3.randn(n, dim).astype(np.float32)
                        # 扰动越强，图像向“均值”塌缩一点，间隙变大
                        img = img * (1 - 0.3 * NOISE_SCALE[cond])
                        data = dict(img_raw=(img * 5).astype(np.float32), img_ids=ids)
                        if cond == "clean":
                            data["txt_raw"] = ((txt + txt_shift) * 5).astype(np.float32)
                            data["txt2img"] = txt2img
                        if dataset == "coco":
                            data["labels"] = labels
                        np.savez_compressed(os.path.join(d_, f"{dataset}_{split}_{cond}.npz"), **data)
                        count += 1
    print(f"wrote {count} npz files under {out}")
    return count


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="features_dummy")
    ap.add_argument("--n", type=int, default=200)
    a = ap.parse_args()
    make(a.out, a.n)
