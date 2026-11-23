"""Report figures from P5 CSVs: python -m eval.plot_p5."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from .p5_common import (MAIN_CONDS, MODELS, REPO_ROOT, SETTINGS, cli_main, write_csv)


COND_COLORS = {"clean": "#3979ad", "noise": "#df8b32", "occl": "#49987d"}
SETTING_COLORS = dict(zip(SETTINGS, ("#3979ad", "#df8b32", "#49987d", "#9763ab")))
MODEL_NAMES = {"clip": "CLIP", "siglip": "SigLIP2"}
REF_NAMES = {"same_cond": "vs pretrained, same condition",
             "pretrained_clean": "vs pretrained clean",
             "self_clean": "pretrained corruption vs clean"}


def _settings(frame):
    return [setting for setting in SETTINGS if setting in set(frame["setting"])]


def _bar_panel(ax, frame, value, *, conditions=MAIN_CONDS, error=None):
    settings = _settings(frame)
    x = np.arange(len(settings))
    width = 0.78 / len(conditions)
    if frame.empty:
        ax.text(0.5, 0.5, "No available features", ha="center", va="center", transform=ax.transAxes)
    for i, cond in enumerate(conditions):
        selected = frame[frame["cond"] == cond].set_index("setting")
        heights = np.asarray([selected.at[s, value] if s in selected.index else np.nan for s in settings], dtype=float)
        positions = x + (i - (len(conditions) - 1) / 2) * width
        valid = np.isfinite(heights)
        errors = None
        if error is not None:
            errors = np.asarray([selected.at[s, error] if s in selected.index else np.nan for s in settings], dtype=float)[valid]
        bars = ax.bar(positions[valid], heights[valid], width=width * 0.93,
                      color=COND_COLORS[cond], yerr=errors, capsize=2, label=cond)
        for j, (bar, height) in enumerate(zip(bars, heights[valid])):
            top = height + (float(errors[j]) if errors is not None else 0)
            ax.annotate(f"{height:.1f}" if error is None else f"{height:.3f}",
                        (bar.get_x() + bar.get_width() / 2, top),
                        xytext=(0, 4), textcoords="offset points", ha="center", fontsize=7)
    ax.set_xticks(x, settings)
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)


def plot_all(out_dir, *, dpi=180):
    out_dir = Path(out_dir)
    frames = []
    for split in ("val", "test"):
        path = out_dir / f"retrieval_{split}.csv"
        if path.exists():
            frame = pd.read_csv(path)
            frame["split"] = split
            frames.append(frame)
    if not frames or all(f.empty for f in frames):
        raise ValueError(f"No retrieval results to plot in {out_dir}")
    retrieval = pd.concat(frames, ignore_index=True)
    drops = pd.read_csv(out_dir / "retrieval_drop.csv")
    drift = pd.read_csv(out_dir / "drift_cosine.csv")
    fig_dir = out_dir / "figs"
    fig_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42})

    def save(fig, stem, caption, kind, dataset, split):
        fig.tight_layout(rect=(0, 0.045, 1, 0.94))
        for extension in ("png", "pdf"):
            path = fig_dir / f"{stem}.{extension}"
            fig.savefig(path, dpi=dpi, bbox_inches="tight")
            manifest.append({"file": f"figs/{path.name}", "kind": kind,
                             "dataset": dataset, "split": split, "caption": caption})
        plt.close(fig)
        print(f"figure {stem}.png / .pdf", flush=True)

    legend = [Patch(facecolor=COND_COLORS[cond], label=cond) for cond in MAIN_CONDS]
    for (dataset, split), group in retrieval.groupby(["dataset", "split"], sort=True):
        models = [m for m in MODELS if m in set(group["model"])]
        fig, axes = plt.subplots(2, len(models), figsize=(6 * len(models), 7), squeeze=False)
        for row, direction in enumerate(("i2t", "t2i")):
            for col, model in enumerate(models):
                ax = axes[row, col]
                _bar_panel(ax, group[(group["model"] == model) & group["cond"].isin(MAIN_CONDS)], f"{direction}_R1")
                ax.set(title=f"{MODEL_NAMES[model]} | {direction}", ylabel="Recall@1 (%)", ylim=(0, 108))
        fig.suptitle(f"{dataset.upper()} {split}: bidirectional retrieval")
        fig.legend(handles=legend, loc="lower center", ncol=3, frameon=False)
        save(fig, f"p5_recall_{dataset}_{split}",
             "R@1 in percent; panels separate models and retrieval directions. Missing groups are omitted.",
             "retrieval", dataset, split)

        subset = drops[(drops["dataset"] == dataset) & (drops["split"] == split) & drops["cond"].isin(("noise", "occl"))]
        if not subset.empty:
            fig, axes = plt.subplots(2, len(models), figsize=(6 * len(models), 6.5), squeeze=False)
            scale = max(1.0, float(subset["drop_pp"].abs().max()))
            for row, direction in enumerate(("i2t", "t2i")):
                for col, model in enumerate(models):
                    ax = axes[row, col]
                    selected = subset[(subset["model"] == model) & (subset["metric"] == f"{direction}_R1")]
                    settings = _settings(group[group["model"] == model])
                    grid = selected.pivot(index="setting", columns="cond", values="drop_pp").reindex(index=settings, columns=["noise", "occl"])
                    values = grid.to_numpy(dtype=float)
                    cmap = plt.get_cmap("RdBu_r").copy()
                    cmap.set_bad("#eeeeee")
                    image = ax.imshow(np.ma.masked_invalid(values), cmap=cmap, vmin=-scale, vmax=scale, aspect="auto")
                    for i in range(len(settings)):
                        for j in range(2):
                            value = values[i, j]
                            ax.text(j, i, f"{value:.1f}" if np.isfinite(value) else "N/A", ha="center", va="center",
                                    color="white" if np.isfinite(value) and abs(value) > 0.65 * scale else "black")
                    ax.set(title=f"{MODEL_NAMES[model]} | {direction}", xticks=[0, 1], xticklabels=["noise", "occl"],
                           yticks=np.arange(len(settings)), yticklabels=settings)
                    fig.colorbar(image, ax=ax, label="R@1 drop (percentage points)", shrink=0.8)
            fig.suptitle(f"{dataset.upper()} {split}: drop relative to clean")
            save(fig, f"p5_drop_{dataset}_{split}",
                 "Absolute R@1 drop in percentage points; positive means lower recall, negative means higher recall.",
                 "drop", dataset, split)

        for modality in ("image", "text"):
            selected = drift[(drift["dataset"] == dataset) & (drift["split"] == split) & (drift["modality"] == modality)]
            if modality == "text":
                selected = selected[selected["cond"] == "clean"]
            else:
                selected = selected[selected["cond"].isin(MAIN_CONDS)]
            refs = [ref for ref in REF_NAMES if ref in set(selected["ref"])]
            if selected.empty:
                continue
            fig, axes = plt.subplots(len(refs), len(models), figsize=(6 * len(models), 3.3 * len(refs)), squeeze=False)
            for row, ref in enumerate(refs):
                for col, model in enumerate(models):
                    ax = axes[row, col]
                    sub = selected[(selected["model"] == model) & (selected["ref"] == ref)]
                    _bar_panel(ax, sub, "mean", conditions=("clean",) if modality == "text" else MAIN_CONDS, error="std")
                    lower = float((sub["mean"] - sub["std"]).min()) if not sub.empty else 0
                    ax.set(title=f"{MODEL_NAMES[model]} | {REF_NAMES[ref]}",
                           ylabel="Paired cosine (mean +/- std)", ylim=(min(0, lower - 0.05), 1.12))
            fig.suptitle(f"{dataset.upper()} {split}: {modality} representation similarity")
            fig.legend(handles=legend[:1] if modality == "text" else legend,
                       loc="lower center", ncol=3, frameon=False)
            save(fig, f"p5_drift_{modality}_{dataset}_{split}",
                 f"Paired {modality} cosine similarity, error bars are population standard deviation. "
                 + ("Text is shown only once (clean), since image corruptions leave text unchanged."
                    if modality == "text" else "References are separated into panels; identity controls equal one."),
                 f"drift_{modality}", dataset, split)

    for kind, conds, labels in (
        ("noise", ("noise_lo", "noise", "noise_hi"), ("weak (0.05)", "medium (0.10)", "strong (0.20)")),
        ("occl", ("occl_lo", "occl", "occl_hi"), ("weak (10%)", "medium (25%)", "strong (50%)")),
    ):
        selected = drops[(drops["dataset"] == "flickr") & (drops["split"] == "val") & drops["cond"].isin(conds)]
        if selected.empty:
            continue
        models = [m for m in MODELS if m in set(selected["model"])]
        fig, axes = plt.subplots(2, len(models), figsize=(6 * len(models), 7), squeeze=False)
        for row, direction in enumerate(("i2t", "t2i")):
            for col, model in enumerate(models):
                ax = axes[row, col]
                sub = selected[(selected["model"] == model) & (selected["metric"] == f"{direction}_R1")]
                for setting in _settings(sub):
                    values = sub[sub["setting"] == setting].set_index("cond")["drop_pp"].reindex(conds)
                    ax.plot(np.arange(3), values, marker="o", color=SETTING_COLORS[setting], label=setting)
                ax.axhline(0, color="#777777", linewidth=0.7)
                ax.set(title=f"{MODEL_NAMES[model]} | {direction}", ylabel="R@1 drop (percentage points)",
                       xticks=np.arange(3), xticklabels=labels)
                ax.grid(alpha=0.2)
                ax.legend(frameon=False)
        fig.suptitle(f"Flickr val: {kind} strength vs R@1 drop")
        save(fig, f"p5_strength_{kind}",
             "Flickr val only. Each line is one setting; both directions and models are separate panels. "
             "Points are measured values, without enforcing monotonicity.", "strength", "flickr", "val")
    write_csv(out_dir / "p5_figures.csv", pd.DataFrame(manifest, columns=["file", "kind", "dataset", "split", "caption"]))
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "results")
    parser.add_argument("--dpi", type=int, default=180)
    args = parser.parse_args()
    plot_all(args.out_dir.resolve(), dpi=args.dpi)


if __name__ == "__main__":
    cli_main(main)
