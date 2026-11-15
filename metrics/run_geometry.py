"""遍历特征文件，计算几何指标并写出 CSV 结果。"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from metrics.geometry import (
    alignment_cos,
    alignment_w,
    gap_linsep,
    modality_gap,
    proto_distance,
    proto_nn_acc,
    prototypes,
    uniformity,
)


MODELS = ("clip", "siglip")
SETTINGS = ("pretrained", "lit", "lora_r8", "lora_r16")
DATASETS = ("flickr", "coco")
SPLITS = ("val", "test")
MAIN_CONDS = ("clean", "noise", "occl")
STRENGTH_CONDS = ("noise_lo", "noise_hi", "occl_lo", "occl_hi")
GEOMETRY_FIELDS = (
    "dataset", "model", "setting", "cond", "align_cos", "align_w",
    "unif_img", "unif_txt", "gap", "gap_ci_low", "gap_ci_high", "gap_linsep",
)
PROTOTYPE_FIELDS = (
    "dataset", "model", "setting", "cond", "proto_dist_mean",
    "proto_dist_median", "proto_nn_acc",
)


def _load(path: Path) -> dict:
    """读取 NPZ 中的数组副本，关闭文件句柄后仍可使用。"""
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def _gap_bootstrap_ci(img, txt_all, n=200, seed=42, confidence=0.95):
    """分别重采样图像和文本样本，估计两模态质心距离的百分位区间。"""
    # 先归一化一次；bootstrap 中反复归一化完整矩阵会产生大量重复计算。
    img = _normalize_rows(img)
    txt_all = _normalize_rows(txt_all)
    rng = np.random.default_rng(seed)
    estimates = np.empty(n, dtype=np.float64)
    for i in range(n):
        # 两种模态独立抽样；文本侧按 caption 观测抽样，不假设与图像逐行配对。
        img_idx = rng.integers(0, len(img), size=len(img))
        txt_idx = rng.integers(0, len(txt_all), size=len(txt_all))
        img_center = img[img_idx].mean(axis=0)
        txt_center = txt_all[txt_idx].mean(axis=0)
        estimates[i] = np.linalg.norm(img_center - txt_center)
    tail = (1 - confidence) / 2
    return tuple(np.quantile(estimates, [tail, 1 - tail]))


def _normalize_rows(x):
    """按行 L2 归一化，供 bootstrap 预处理复用。"""
    x = np.asarray(x, dtype=np.float64)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def _condition_paths(feature_root, model, setting, dataset, split):
    """返回该组合存在的条件文件；弱、强档只对 Flickr val 扫描。"""
    base = feature_root / model / setting
    conditions = list(MAIN_CONDS)
    if (dataset, split) == ("flickr", "val"):
        conditions += list(STRENGTH_CONDS)
    return {
        cond: base / f"{dataset}_{split}_{cond}.npz"
        for cond in conditions
        if (base / f"{dataset}_{split}_{cond}.npz").is_file()
    }


def run(feature_root="features", output_dir="results", *, ci_samples=200, seed=42):
    """计算存在的特征组合；缺失组合跳过，最终 CSV 仍包含固定表头。"""
    feature_root, output_dir = Path(feature_root), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    geometry_rows, prototype_rows = [], []

    for model in MODELS:
        for setting in SETTINGS:
            for dataset in DATASETS:
                for split in SPLITS:
                    paths = _condition_paths(feature_root, model, setting, dataset, split)
                    if not paths:
                        continue
                    # clean 文件保存文本特征；若 clean 缺失，该组合无法计算图文联合指标。
                    clean_path = paths.get("clean")
                    if clean_path is None:
                        print(f"skip {model}/{setting}/{dataset}_{split}: missing clean features")
                        continue
                    clean = _load(clean_path)
                    img_clean = clean["img_raw"]
                    txt_all = clean["txt_raw"]
                    n_img = len(img_clean)
                    if len(txt_all) != 5 * n_img:
                        raise ValueError(f"{clean_path}: expected five captions per image")
                    txt1 = txt_all[0::5]
                    if "txt2img" in clean:
                        expected = np.repeat(np.arange(n_img), 5)
                        if not np.array_equal(clean["txt2img"], expected):
                            raise ValueError(f"{clean_path}: txt2img does not match five-caption ordering")

                    for cond, path in paths.items():
                        data = clean if cond == "clean" else _load(path)
                        img = data["img_raw"]
                        if img.shape != img_clean.shape:
                            raise ValueError(f"{path}: image feature shape differs from clean")
                        # 扰动只改变图像；对齐、文本侧均匀性和模态间隙都复用 clean 文本。
                        lo, hi = _gap_bootstrap_ci(img, txt_all, n=ci_samples, seed=seed)
                        geometry_rows.append({
                            "dataset": dataset,
                            "model": model,
                            "setting": setting,
                            "cond": cond,
                            "align_cos": alignment_cos(img, txt1),
                            "align_w": alignment_w(img, txt1),
                            "unif_img": uniformity(img, seed=seed),
                            "unif_txt": uniformity(txt_all, seed=seed),
                            "gap": modality_gap(img, txt_all),
                            "gap_ci_low": lo,
                            "gap_ci_high": hi,
                            "gap_linsep": gap_linsep(img, txt_all),
                        })

                        if dataset == "coco":
                            labels = data.get("labels", clean.get("labels"))
                            if labels is None or len(labels) != n_img:
                                raise ValueError(f"{path}: COCO labels missing or misaligned")
                            proto = prototypes(img, txt_all, labels, min_count=20)
                            if proto:
                                ip = {label: pair[0] for label, pair in proto.items()}
                                tp = {label: pair[1] for label, pair in proto.items()}
                                counts = {label: int(np.sum(labels == label)) for label in proto}
                                dist = proto_distance(ip, tp)
                                acc = proto_nn_acc(ip, tp)
                                prototype_rows.append({
                                    "dataset": dataset,
                                    "model": model,
                                    "setting": setting,
                                    "cond": cond,
                                    "proto_dist_mean": dist["mean"],
                                    "proto_dist_median": dist["median"],
                                    "proto_nn_acc": acc["mean"],
                                })

    _write_csv(output_dir / "geometry_metrics.csv", GEOMETRY_FIELDS, geometry_rows)
    _write_csv(output_dir / "prototype_metrics.csv", PROTOTYPE_FIELDS, prototype_rows)
    print(f"wrote {len(geometry_rows)} geometry rows and {len(prototype_rows)} prototype rows to {output_dir}")
    return geometry_rows, prototype_rows


def _write_csv(path, fields, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="计算图文嵌入几何指标")
    parser.add_argument("--features", default="features", help="特征目录（默认：features）")
    parser.add_argument("--out", default="results", help="CSV 输出目录（默认：results）")
    parser.add_argument("--ci-samples", type=int, default=200, help="模态间隙 bootstrap 次数")
    parser.add_argument("--seed", type=int, default=42, help="抽样随机种子")
    args = parser.parse_args()
    if args.ci_samples < 1:
        parser.error("--ci-samples must be positive")
    run(args.features, args.out, ci_samples=args.ci_samples, seed=args.seed)


if __name__ == "__main__":
    main()
