"""在线扰动（不落盘）。输入 [3,H,W]，取值 [0,1]，返回同形状。

mode:
  clean                              原图
  noise_lo / noise / noise_hi        高斯噪声，sigma = 0.05 / 0.10 / 0.20，加完 clip 到 [0,1]
  occl_lo / occl / occl_hi           随机矩形遮挡（Random Erasing 风格），面积占整图 10% / 25% / 50%，
                                     长宽比对数均匀采样于 [0.3, 3.3]，位置随机，填该模型的均值像素

可复现性：随机数只由 (seed, idx) 决定。
  * 同一样本在同一模型的所有设置下扰动完全相同；
  * 三档噪声共用同一个噪声场（只差 sigma 倍数）；三档遮挡共用同一组 (长宽比, u, v)，只有面积不同，
    所以强度之间只有强度一个变量；
  * 遮挡的形状和位置都用相对坐标，两个输入尺寸不同的模型遮挡原图的同一块区域（取整误差内）。
"""
import torch

from .config import load_perturb_cfg

MAIN_MODES = ["clean", "noise", "occl"]
STRENGTH_MODES = ["noise_lo", "noise_hi", "occl_lo", "occl_hi"]
ALL_MODES = ["clean", "noise_lo", "noise", "noise_hi", "occl_lo", "occl", "occl_hi"]

# 两个独立随机流，避免噪声与遮挡位置互相影响
_NOISE_OFFSET = 0
_OCCL_OFFSET = 1_000_003


def _gen(seed, idx, offset):
    g = torch.Generator()
    g.manual_seed(int(seed) + int(idx) + offset)
    return g


def noise_field(shape, idx, seed=None):
    """标准正态噪声场，N(0,1)。三档噪声共用，强度只是 sigma 倍数。"""
    seed = load_perturb_cfg()["seed"] if seed is None else seed
    return torch.randn(shape, generator=_gen(seed, idx, _NOISE_OFFSET))


def occl_params(idx, seed=None):
    """(r, u, v)：长宽比、位置的相对坐标。三档遮挡共用，保证强度之间只有面积不同。"""
    cfg = load_perturb_cfg()
    seed = cfg["seed"] if seed is None else seed
    lo, hi = cfg["occl_aspect"]
    g = _gen(seed, idx, _OCCL_OFFSET)
    t = torch.rand(3, generator=g)
    r = float(torch.exp(torch.log(torch.tensor(lo)) + t[0] * (torch.log(torch.tensor(hi)) - torch.log(torch.tensor(lo)))))
    return r, float(t[1]), float(t[2])


def occl_box(h, w, area, idx, seed=None):
    """返回遮挡矩形 (top, left, bh, bw)。相对高 = sqrt(area / r)，相对宽 = sqrt(area * r)；
    若超出画面就把长边截到画面边界，同时放宽另一边保持面积（确定性，不依赖拒绝采样）。"""
    r, u, v = occl_params(idx, seed)
    rh, rw = (area / r) ** 0.5, (area * r) ** 0.5
    if rh > 1.0:
        rh, rw = 1.0, min(1.0, area)
    elif rw > 1.0:
        rw, rh = 1.0, min(1.0, area)
    bh = max(1, min(h, int(round(rh * h))))
    bw = max(1, min(w, int(round(rw * w))))
    top = int(round(u * (h - bh)))
    left = int(round(v * (w - bw)))
    return top, left, bh, bw


def _fill_tensor(fill, ref):
    if fill is None:
        return torch.full((3, 1, 1), 0.5, dtype=ref.dtype)
    return torch.as_tensor(fill, dtype=ref.dtype).view(3, 1, 1)


def perturb(img01, mode, idx, seed=None, fill=None):
    """fill：遮挡填充的 RGB 值（[0,1] 空间，长度 3）。传该模型的 mean 即“归一化后为 0”；默认 0.5 灰。"""
    if mode == "clean":
        return img01
    cfg = load_perturb_cfg()
    seed = cfg["seed"] if seed is None else seed
    if mode in cfg["noise_sigma"]:
        sigma = float(cfg["noise_sigma"][mode])
        out = img01 + noise_field(tuple(img01.shape), idx, seed) * sigma
        return out.clamp(0.0, 1.0)
    if mode in cfg["occl_area"]:
        _, h, w = img01.shape
        top, left, bh, bw = occl_box(h, w, float(cfg["occl_area"][mode]), idx, seed)
        out = img01.clone()
        out[:, top:top + bh, left:left + bw] = _fill_tensor(fill, img01).expand(3, bh, bw)
        return out
    raise ValueError(f"unknown perturb mode: {mode}")
