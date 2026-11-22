"""从 geometry_metrics.csv 绘制指标图。"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


MODELS = ("clip", "siglip")
SETTINGS = ("pretrained", "lit", "lora_r8", "lora_r16")
MAIN_CONDS = ("clean", "noise", "occl")
STRENGTH = {
    "noise": ("noise_lo", "noise", "noise_hi"),
    "occl": ("occl_lo", "occl", "occl_hi"),
}
COLORS = {"pretrained": "#4C78A8", "lit": "#F58518", "lora_r8": "#54A24B", "lora_r16": "#E45756"}
MARKERS = {"clean": "o", "noise": "s", "occl": "^", "noise_lo": "<", "noise_hi": ">", "occl_lo": "v", "occl_hi": "P"}


def _read(path):
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def _val_rows(rows):
    return [r for r in rows if r["split"] == "val"] if rows and "split" in rows[0] else rows


def plot_gap_bars(rows, out_dir):
    """绘制val集上按模型、数据集分面的条件间隙柱状图。"""
    rows = [r for r in _val_rows(rows) if r["cond"] in MAIN_CONDS]
    datasets = sorted({r["dataset"] for r in rows})
    if not rows:
        return
    fig, axes = plt.subplots(len(MODELS), max(1, len(datasets)), figsize=(6 * len(datasets), 4.4 * len(MODELS)), squeeze=False)
    width = 0.24
    for i, model in enumerate(MODELS):
        for j, dataset in enumerate(datasets):
            ax = axes[i, j]
            subset = [r for r in rows if r["model"] == model and r["dataset"] == dataset]
            x = np.arange(len(SETTINGS))
            for k, cond in enumerate(MAIN_CONDS):
                vals = [next((float(r["gap"]) for r in subset if r["setting"] == s and r["cond"] == cond), np.nan) for s in SETTINGS]
                ax.bar(x + (k - 1) * width, vals, width, label=cond)
            ax.set_title(f"{model.upper()} · {dataset}")
            ax.set_xticks(x, SETTINGS, rotation=20)
            ax.set_ylabel("Modality gap")
            ax.grid(axis="y", alpha=0.25)
            if i == 0 and j == 0:
                ax.legend(title="Condition")
    fig.suptitle("Image–text modality gap (validation split)")
    fig.tight_layout()
    fig.savefig(out_dir / "modality_gap_bars.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_alignment_uniformity(rows, out_dir):
    """绘制对齐与图像/文本均匀性的散点关系。"""
    rows = _val_rows(rows)
    datasets = sorted({r["dataset"] for r in rows})
    if not rows:
        return
    fig, axes = plt.subplots(len(MODELS), max(1, len(datasets)), figsize=(6 * len(datasets), 4.6 * len(MODELS)), squeeze=False)
    for i, model in enumerate(MODELS):
        for j, dataset in enumerate(datasets):
            ax = axes[i, j]
            for modality, ykey, marker, fill in (("image", "unif_img", "o", True), ("text", "unif_txt", "X", False)):
                for r in rows:
                    if r["model"] != model or r["dataset"] != dataset:
                        continue
                    ax.scatter(float(r["align_cos"]), float(r[ykey]),
                               color=COLORS[r["setting"]], marker=MARKERS[r["cond"]] if modality == "image" else marker,
                               facecolors=COLORS[r["setting"]] if fill else "none", edgecolors=COLORS[r["setting"]],
                               s=42, alpha=0.8, label=f"{r['setting']} / {r['cond']} / {modality}")
            ax.set_title(f"{model.upper()} · {dataset}")
            ax.set_xlabel("Alignment distance (lower is better)")
            ax.set_ylabel("Uniformity (lower is more uniform)")
            ax.grid(alpha=0.2)
    # 使用两个紧凑图例避免把每行组合都重复展示。
    handles = [plt.Line2D([], [], color=c, marker="o", linestyle="", label=s) for s, c in COLORS.items()]
    handles += [plt.Line2D([], [], color="gray", marker=MARKERS[c], linestyle="", label=c) for c in MAIN_CONDS]
    handles += [plt.Line2D([], [], color="gray", marker=m, markerfacecolor="none", linestyle="", label=mod)
                for mod, m in (("image", "o"), ("text", "X"))]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=8)
    fig.suptitle("Alignment–uniformity relationship (validation split)")
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    fig.savefig(out_dir / "alignment_uniformity_scatter.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_strength(rows, out_dir, metric, ylabel, filename):
    """绘制Flickr val中噪声和遮挡弱、中、强三档的趋势。"""
    rows = [r for r in rows if r["dataset"] == "flickr" and r.get("split", "val") == "val"]
    if not rows:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), squeeze=False)
    axes = axes[0]
    for ax, family in zip(axes, ("noise", "occl")):
        conds = STRENGTH[family]
        x = np.arange(3)
        for model in MODELS:
            for setting in SETTINGS:
                vals = []
                for cond in conds:
                    match = next((r for r in rows if r["model"] == model and r["setting"] == setting and r["cond"] == cond), None)
                    vals.append(float(match[metric]) if match else np.nan)
                ax.plot(x, vals, marker="o", color=COLORS[setting], linestyle="-" if model == "clip" else "--",
                        alpha=0.9, label=f"{model.upper()} · {setting}")
        ax.set_xticks(x, ("low", "medium", "high"))
        ax.set_title(f"{family.capitalize()} strength")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8)
    fig.suptitle(f"{ylabel} by perturbation strength (Flickr validation)")
    fig.tight_layout(rect=(0, 0.16, 1, 0.92))
    fig.savefig(out_dir / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="绘制几何指标图")
    parser.add_argument("--csv", default="results/geometry_metrics.csv")
    parser.add_argument("--out", default="results/figs")
    args = parser.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = _read(Path(args.csv))
    
    for row in rows:
        row.setdefault("split", "val")
    plot_gap_bars(rows, out_dir)
    plot_alignment_uniformity(rows, out_dir)
    plot_strength(rows, out_dir, "gap", "Modality gap", "strength_modality_gap.png")
    plot_strength(rows, out_dir, "align_cos", "Alignment distance", "strength_alignment.png")
    print(f"Saved figures to {out_dir}")


if __name__ == "__main__":
    main()
