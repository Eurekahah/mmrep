"""Shared P9 figure conventions (safe to import from other modules)."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODALITY_COLORS = {"image": "#2878B5", "text": "#E58632"}
SETTING_COLORS = {"pretrained": "#4477AA", "lit": "#228833",
                  "lora_r8": "#CC6677", "lora_r16": "#AA3377"}
CONDITION_MARKERS = {"clean": "o", "noise": "^", "occl": "s",
                     "noise_lo": "v", "noise_hi": "P", "occl_lo": "D", "occl_hi": "X"}
MODEL_NAMES = {"clip": "CLIP", "siglip": "SigLIP2"}
SETTING_NAMES = {"pretrained": "Pretrained", "lit": "LiT", "lora_r8": "LoRA r=8", "lora_r16": "LoRA r=16"}
DPI = 300


def apply_style():
    plt.rcParams.update({"font.family": "sans-serif",
                         "font.sans-serif": ["DejaVu Sans", "Microsoft YaHei", "SimHei", "Noto Sans CJK SC"],
                         "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
                         "legend.fontsize": 9, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.unicode_minus": False,
                         "savefig.dpi": DPI, "svg.fonttype": "none"})


def save_figure(fig, stem):
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".svg"):
        fig.savefig(stem.with_suffix(suffix), dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
