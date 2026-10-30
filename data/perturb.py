"""在线扰动（不落盘）。输入 [3,H,W]，取值 [0,1]，返回同形状。

mode:
  clean                              原图
  noise_lo / noise / noise_hi        高斯噪声，sigma = 0.05 / 0.10 / 0.20，加完 clip 到 [0,1]
  occl_lo / occl / occl_hi           随机方块遮挡，边长 = 输入边长 x 0.2 / 0.3 / 0.4，填 0

可复现性：随机数只由 (seed, idx) 决定。
  * 同一样本在同一模型的所有设置下扰动完全相同；
  * 三档强度共用同一个噪声场（只差 sigma 倍数）和同一组遮挡位置参数 (u, v)，所以强度之间只有强度一个变量；
  * 遮挡位置用相对坐标 (u, v) in [0,1]，两个输入尺寸不同的模型遮挡原图的同一块区域。
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


def occl_position(idx, seed=None):
    """遮挡位置的相对坐标 (u, v)，均匀分布在 [0,1]。"""
    seed = load_perturb_cfg()["seed"] if seed is None else seed
    uv = torch.rand(2, generator=_gen(seed, idx, _OCCL_OFFSET))
    return float(uv[0]), float(uv[1])


def occl_box(h, w, ratio, idx, seed=None):
    """返回遮挡方块 (top, left, side)。边长按 h 取整（输入为正方形）。"""
    side = int(round(ratio * h))
    side = max(1, min(side, h, w))
    u, v = occl_position(idx, seed)
    top = int(round(u * (h - side)))
    left = int(round(v * (w - side)))
    return top, left, side


def perturb(img01, mode, idx, seed=None):
    if mode == "clean":
        return img01
    cfg = load_perturb_cfg()
    seed = cfg["seed"] if seed is None else seed
    if mode in cfg["noise_sigma"]:
        sigma = float(cfg["noise_sigma"][mode])
        out = img01 + noise_field(tuple(img01.shape), idx, seed) * sigma
        return out.clamp(0.0, 1.0)
    if mode in cfg["occl_ratio"]:
        _, h, w = img01.shape
        top, left, side = occl_box(h, w, float(cfg["occl_ratio"][mode]), idx, seed)
        out = img01.clone()
        out[:, top:top + side, left:left + side] = 0.0
        return out
    raise ValueError(f"unknown perturb mode: {mode}")
