"""图像预处理：按各模型官方处理器的方式 resize，扰动在 resize 之后、归一化之前。

顺序固定：PIL 图像 -> resize(/裁剪) 到 input_size -> 转 [0,1] 张量 -> perturb -> normalize
"""
import numpy as np
import torch
from PIL import Image

from .config import load_models_cfg

_INTERP = {
    "bicubic": Image.BICUBIC,
    "bilinear": Image.BILINEAR,
}


def input_size(model):
    return int(load_models_cfg()[model]["input_size"])


def load_resized_01(pil_img, model):
    """PIL 图像 -> [3,S,S] float32，取值 [0,1]，已按该模型官方方式缩放（裁剪）。"""
    cfg = load_models_cfg()[model]
    size = int(cfg["input_size"])
    mode = cfg["resize"]["mode"]
    interp = _INTERP[cfg["resize"]["interpolation"]]
    img = pil_img.convert("RGB")
    w, h = img.size
    if mode == "shortest_edge_center_crop":
        # 与 HF CLIPImageProcessor 一致：短边缩放到 size，长边按比例取整，再中心裁剪
        if w <= h:
            new_w, new_h = size, int(size * h / w)
        else:
            new_w, new_h = int(size * w / h), size
        img = img.resize((new_w, new_h), interp)
        left = (new_w - size) // 2
        top = (new_h - size) // 2
        img = img.crop((left, top, left + size, top + size))
    elif mode == "squash":
        # 与 HF SiglipImageProcessor 一致：直接缩放到 size x size
        img = img.resize((size, size), interp)
    else:
        raise ValueError(f"unknown resize mode: {mode}")
    arr = np.asarray(img, dtype=np.uint8).copy()
    return torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0


def normalize(x01, model):
    cfg = load_models_cfg()[model]
    mean = torch.tensor(cfg["mean"], dtype=x01.dtype).view(3, 1, 1)
    std = torch.tensor(cfg["std"], dtype=x01.dtype).view(3, 1, 1)
    return (x01 - mean) / std


def preprocess_pil(pil_img, model, cond="clean", idx=0):
    """完整流水线：resize -> perturb -> normalize，返回可直接送进模型的张量。"""
    from .perturb import perturb
    x = load_resized_01(pil_img, model)
    x = perturb(x, cond, idx, fill=load_models_cfg()[model]["mean"])
    return normalize(x, model)
