"""Joint, reproducible P9 projections. Run with python -m viz.tsne_umap."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from importlib.metadata import version
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from threadpoolctl import threadpool_limits

from eval.p5_common import (REPO_ROOT, MODELS, SETTINGS, MAIN_CONDS, FeatureKey,
                            load_features, assert_aligned, unit_rows, sha256,
                            write_csv, write_json)
from viz.style import (plt, apply_style, save_figure, MODALITY_COLORS,
                       SETTING_COLORS, SETTING_NAMES, MODEL_NAMES, CONDITION_MARKERS)


@dataclass(frozen=True)
class Group:
    model: str
    dataset: str
    split: str
    name: str
    members: tuple  # (setting, condition)
    kind: str


def groups(models, datasets, split="val", scope="all", strength_model="clip"):
    result = []
    for model in models:
        for dataset in datasets:
            if scope in ("all", "settings"):
                for cond in MAIN_CONDS:
                    result.append(Group(model, dataset, split, f"settings_{cond}",
                                        tuple((s, cond) for s in SETTINGS), "settings"))
            if scope in ("all", "conditions"):
                for setting in SETTINGS:
                    result.append(Group(model, dataset, split, f"conditions_{setting}",
                                        tuple((setting, c) for c in MAIN_CONDS), "conditions"))
            if scope in ("all", "strength") and model == strength_model and dataset == "flickr" and split == "val":
                result.append(Group(model, dataset, split, "strength_pretrained",
                                    tuple(("pretrained", c) for c in
                                          ("clean", "noise_lo", "noise", "noise_hi", "occl_lo", "occl", "occl_hi")),
                                    "strength"))
    return result


def sample_indices(ids, n, seed):
    if n < 2 or n > len(ids):
        raise ValueError(f"Requested {n} images; available {len(ids)}. Set --n explicitly for a smoke run.")
    # Canonical ID order also makes selections stable across models and input permutations.
    ordered = np.argsort(ids, kind="stable")
    chosen = np.random.default_rng(seed).choice(len(ids), n, replace=False)
    return ordered[np.sort(chosen)]


def prepare_group(root, group, n=1000, seed=42, splits_root=None):
    keys = {FeatureKey(group.model, s, group.dataset, group.split, c) for s, c in group.members}
    keys |= {k.with_(cond="clean") for k in keys}
    loaded = {k: load_features(root, k, splits_root) for k in sorted(keys, key=lambda k: str(k.relative_path))}
    reference = next(iter(loaded.values()))
    for key, data in loaded.items():
        assert_aligned(reference, data, str(key.relative_path))
    idx = sample_indices(reference.ids, n, seed)
    pair_set = set(np.random.default_rng(seed + 1).choice(n, min(30, n), replace=False).tolist())
    arrays, records = [], []
    for setting, cond in group.members:
        key = FeatureKey(group.model, setting, group.dataset, group.split, cond)
        data = loaded[key]
        clean = loaded[key.with_(cond="clean")]
        blocks = [("image", data.img[idx], cond, key)]
        # Text is invariant to corruptions. Fit it ONCE and use the same coordinates in every panel.
        if group.kind == "settings" or cond == "clean":
            blocks.append(("text", clean.txt[5 * idx], "clean", key.with_(cond="clean")))
        for modality, raw, block_cond, source_key in blocks:
            arrays.append(unit_rows(raw))
            for j, original in enumerate(idx):
                records.append({"model": group.model, "dataset": group.dataset, "split": group.split,
                                "setting": setting, "cond": block_cond, "modality": modality,
                                "img_id": str(data.ids[original]), "sample_index": j,
                                "source_row": int(original if modality == "image" else 5 * original),
                                "label": int(data.labels[original]) if data.labels is not None else -1,
                                "paired_line": j in pair_set,
                                "source_npz": source_key.relative_path.as_posix()})
    source_files = [{"path": k.relative_path.as_posix(), "sha256": sha256(Path(root) / k.relative_path)}
                    for k in sorted(keys, key=lambda k: str(k.relative_path))]
    return np.concatenate(arrays), pd.DataFrame(records), source_files


def reduce_joint(x, method, seed=42, max_iter=1000):
    if min(x.shape) < 50:
        raise ValueError("P9 requires at least 50 rows and dimensions for PCA(50)")
    pca = PCA(n_components=50, svd_solver="randomized", random_state=seed)
    z = pca.fit_transform(x)
    if method == "tsne":
        reducer = TSNE(n_components=2, perplexity=30, init="pca", random_state=seed,
                       learning_rate="auto", max_iter=max_iter, n_jobs=1)
    elif method == "umap":
        from umap import UMAP
        reducer = UMAP(n_components=2, n_neighbors=15, min_dist=0.1,
                       random_state=seed, n_jobs=1, metric="euclidean")
    else:
        raise ValueError(f"Unknown method: {method}")
    xy = reducer.fit_transform(z)
    if not np.isfinite(xy).all():
        raise ValueError("Projection contains non-finite coordinates")
    return xy, float(pca.explained_variance_ratio_.sum())


def plot_group(frame, group, method, stem, category=False, demo=False):
    apply_style()
    members = group.members
    count = len(members)
    cols = 4 if count > 4 else count
    rows = (count + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4.3 * rows + .5), squeeze=False, sharex=True, sharey=True)
    lo = frame[["x", "y"]].min().to_numpy()
    hi = frame[["x", "y"]].max().to_numpy()
    pad = np.maximum((hi - lo) * 0.07, 0.1)
    top = (frame.drop_duplicates("img_id")["label"].value_counts()
           .rename_axis("label").reset_index(name="n").sort_values(["n", "label"], ascending=[False, True]).head(10)["label"].tolist())
    colors = {label: plt.get_cmap("tab10")(i) for i, label in enumerate(top)}
    handles = []
    for ax, (setting, cond) in zip(axes.flat, members):
        img = frame[(frame.setting == setting) & (frame.cond == cond) & (frame.modality == "image")]
        txt = frame[(frame.setting == setting) & (frame.modality == "text")]
        for modality, points in (("image", img), ("text", txt)):
            c = [colors.get(int(l), "#BDBDBD") for l in points.label] if category else MODALITY_COLORS[modality]
            marker = CONDITION_MARKERS[cond] if modality == "image" else "x"
            artist = ax.scatter(points.x, points.y, s=9, c=c, marker=marker,
                                alpha=0.55, linewidths=0.45 if modality == "text" else 0, label=modality)
            if not handles or len(handles) == 1:
                handles.append(artist)
        pairs = img[img.paired_line].merge(txt, on="img_id", suffixes=("_i", "_t"))
        for row in pairs.itertuples():
            ax.plot([row.x_i, row.x_t], [row.y_i, row.y_t], color="#777777", lw=0.45, alpha=0.35, zorder=0)
        if group.kind == "strength" and cond != "clean":
            base = frame[(frame.cond == "clean") & (frame.modality == "image")]
            motion = img[img.paired_line].merge(base, on="img_id", suffixes=("_p", "_c"))
            ax.quiver(motion.x_c, motion.y_c, motion.x_p-motion.x_c, motion.y_p-motion.y_c,
                      angles="xy", scale_units="xy", scale=1, color="#222222", alpha=.6, width=.003)
        ax.set(title=f"{SETTING_NAMES[setting]} | {cond}", xlim=(lo[0]-pad[0], hi[0]+pad[0]),
               ylim=(lo[1]-pad[1], hi[1]+pad[1]), xlabel="Dimension 1", ylabel="Dimension 2")
        ax.set_aspect("equal", adjustable="box")
        ax.spines["bottom"].set_color(SETTING_COLORS[setting])
        ax.spines["bottom"].set_linewidth(2)
    for ax in list(axes.flat)[count:]:
        ax.set_visible(False)
    if category:
        from matplotlib.lines import Line2D
        handles = [Line2D([], [], color=colors[l], marker="o", linestyle="", label=f"COCO {l}") for l in top]
        handles += [Line2D([], [], color="#BDBDBD", marker="o", linestyle="", label="Other")]
        labels = [h.get_label() for h in handles]
    else:
        labels = ["Image (shape = condition)", "Text (first caption)"]
    fig.legend(handles, labels, loc="lower center", ncol=6 if category else 2, frameon=False)
    suffix = " | SYNTHETIC DEMO — NOT EXPERIMENTAL RESULTS" if demo else ""
    fig.suptitle(f"{MODEL_NAMES[group.model]} | {group.dataset.upper()} {group.split} | {method.upper()} | {group.name}{suffix}", y=.995)
    fig.tight_layout(rect=(0, .16 if rows == 1 else .11, 1, .96))
    save_figure(fig, stem)


def run_group(root, out, group, method, n=1000, seed=42, splits_root=None, max_iter=1000, demo=False, resume=False):
    out = Path(out)
    name = f"{group.model}_{method}_{group.name}_{group.dataset}"
    if group.split != "val":
        name += f"_{group.split}"
    x, frame, sources = prepare_group(root, group, n, seed, splits_root)
    config = {"model": group.model, "dataset": group.dataset, "split": group.split,
              "group": group.name, "kind": group.kind, "members": group.members,
              "method": method, "n_images": n, "seed": seed, "demo": demo,
              "pca_components": 50, "perplexity": 30, "init": "pca", "max_iter": max_iter,
              "n_neighbors": 15, "min_dist": 0.1, "text_fit": "once_per_setting", "sources": sources,
              "script_sha256": sha256(Path(__file__)), "style_sha256": sha256(Path(__file__).with_name("style.py")),
              "versions": {p: version(p) for p in ("numpy", "scikit-learn", "umap-learn", "matplotlib", "pandas")}}
    # JSON normalizes tuples before comparing a previous run.
    config = json.loads(json.dumps(config))
    meta_path = out / "coordinates" / f"{name}.json"
    csv_path = meta_path.with_suffix(".csv")
    stems = [out / "figures" / name]
    if group.dataset == "coco":
        stems.append(out / "figures" / f"{name}_classes")
    if resume and meta_path.exists():
        previous = json.loads(meta_path.read_text(encoding="utf-8"))
        if previous["config"] == config and all((out / p).is_file() and sha256(out / p) == h for p, h in previous["artifacts"].items()):
            return previous
    t0 = time.perf_counter()
    with threadpool_limits(limits=2):
        xy, energy = reduce_joint(x, method, seed, max_iter)
    frame["x"], frame["y"] = xy[:, 0], xy[:, 1]
    write_csv(csv_path, frame)
    for stem in stems:
        plot_group(frame, group, method, stem, category=stem.name.endswith("_classes"), demo=demo)
    paths = [csv_path] + [s.with_suffix(ext) for s in stems for ext in (".png", ".svg")]
    result = {"config": config, "pca_explained_variance_ratio": energy,
              "wall_sec": round(time.perf_counter()-t0, 3), "points": len(frame),
              "coordinates": csv_path.relative_to(out).as_posix(),
              "figures": [s.with_suffix(".png").relative_to(out).as_posix() for s in stems],
              "artifacts": {p.relative_to(out).as_posix(): sha256(p) for p in paths}}
    write_json(meta_path, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-root", type=Path, default=REPO_ROOT / "features")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "results" / "p9")
    parser.add_argument("--model", choices=("all", *MODELS), default="all")
    parser.add_argument("--dataset", choices=("all", "flickr", "coco"), default="all")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--method", choices=("all", "tsne", "umap"), default="all")
    parser.add_argument("--group", choices=("all", "settings", "conditions", "strength"), default="all")
    parser.add_argument("--strength-model", choices=tuple(MODELS), default="clip")
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--demo", action="store_true", help="Disable official split checks; label all outputs as synthetic")
    parser.add_argument("--allow-missing", action="store_true", help="Record missing groups instead of failing")
    parser.add_argument("--resume", action="store_true", help="Reuse only artifacts whose inputs/configuration/hashes match")
    args = parser.parse_args()
    if args.demo and args.out_dir.resolve() == (REPO_ROOT / "results" / "p9").resolve():
        parser.error("--demo needs a separate --out-dir, e.g. results/p9_demo")
    if args.max_iter < 250:
        parser.error("--max-iter must be at least 250")
    selected = groups(tuple(MODELS) if args.model == "all" else (args.model,),
                      ("flickr", "coco") if args.dataset == "all" else (args.dataset,), args.split, args.group, args.strength_model)
    methods = ("tsne", "umap") if args.method == "all" else (args.method,)
    manifest = {"demo": args.demo, "requested_groups": len(selected)*len(methods), "completed": [], "missing": []}
    manifest_name = "manifest.json" if args.model == "all" and args.method == "all" else f"manifest_{args.model}_{args.method}.json"
    for group in selected:
        for method in methods:
            required = {FeatureKey(group.model, s, group.dataset, group.split, c).relative_path for s, c in group.members}
            required |= {FeatureKey(group.model, s, group.dataset, group.split, "clean").relative_path for s, _ in group.members}
            missing = [str(p) for p in sorted(required) if not (args.features_root / p).exists()]
            if missing:
                manifest["missing"].append({"group": group.name, "model": group.model, "dataset": group.dataset, "method": method, "files": missing})
                if not args.allow_missing:
                    raise FileNotFoundError(f"Missing features: {missing}")
                continue
            print(f"{group.model} {group.dataset} {group.name} {method}", flush=True)
            result = run_group(args.features_root, args.out_dir, group, method, args.n, args.seed,
                               None if args.demo else REPO_ROOT / "splits", args.max_iter, args.demo, args.resume)
            manifest["completed"].append(result)
            write_json(args.out_dir / manifest_name, manifest)
    manifest["complete"] = not manifest["missing"] and bool(selected)
    write_json(args.out_dir / manifest_name, manifest)
    print(f"Completed {len(manifest['completed'])}; missing {len(manifest['missing'])}")


if __name__ == "__main__":
    main()
