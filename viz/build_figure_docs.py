"""Index existing group figures and produce source-based captions plus reproducible spot checks."""
from __future__ import annotations

import argparse
from functools import lru_cache
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from eval.p5_common import REPO_ROOT, sha256, write_csv, write_json, atomic_text
from viz.make_report_tables import format_value, read_csv

INDEX_COLUMNS = ["图号", "文件路径", "生成脚本", "数据来源 csv", "对应报告章节", "一句话说明", "核验状态"]


def projection_caption(meta, coordinates, category):
    config = meta["config"]
    frame = pd.read_csv(coordinates)
    caption = f"{config['model'].upper()} 在 {config['dataset']} {config['split']} 的 {config['group']} 联合 {config['method'].upper()} 投影。"
    if category:
        counts = frame.drop_duplicates("img_id").label.value_counts().head(3)
        observations = ["采样图像中最多的三个类别（COCO 类别 ID）及数量为：" + "、".join(f"{k}: {v}" for k, v in counts.items()) + "。",
                        "同一类别的图像与首条文本使用相同颜色；只给数量最多的十类着色，其余点为灰色。",
                        "类别标签随图像 ID 固定，所有面板使用同一批样本。"]
    else:
        distances = []
        for setting, cond in config["members"]:
            img = frame[(frame.setting == setting) & (frame.cond == cond) & (frame.modality == "image")]
            txt = frame[(frame.setting == setting) & (frame.modality == "text")]
            gap = np.linalg.norm(img[["x", "y"]].mean().to_numpy()-txt[["x", "y"]].mean().to_numpy())
            distances.append((f"{setting}/{cond}", gap))
        low, high = min(distances, key=lambda v: v[1]), max(distances, key=lambda v: v[1])
        observations = [f"本图二维图文质心距离最小的面板是 {low[0]}（{low[1]:.2f}），最大的为 {high[0]}（{high[1]:.2f}）；这些数值仅描述本投影。",
                        f"每个面板显示同一批 {config['n_images']} 张图像及其首条文本，灰线连接固定抽取的 {min(30, config['n_images'])} 对图文。",
                        "二维距离受降维影响，不能替代原始嵌入的模态间隙，也不能跨图或跨模型比较大小。"]
        if config["kind"] == "strength":
            observations[1] += "黑色箭头表示所选图像从 clean 到该扰动强度的位移；文本坐标在各面板完全一致。"
    return caption, observations


def upstream_source(path, root):
    name = path.name
    if name.startswith("p5_"):
        if "recall" in name: csv = f"retrieval_{'test' if '_test' in name else 'val'}.csv"
        elif "drop" in name or "strength" in name: csv = "retrieval_drop.csv"
        else: csv = "drift_cosine.csv"
        return "eval/plot_p5.py", "results/"+csv, "下游任务与嵌入漂移"
    if name.startswith("norm_"):
        base = path.parent.parent
        return "extract/norm_stats.py", (base/"norm_stats.csv").relative_to(root).as_posix(), "特征范数"
    if name in ("alignment_uniformity_scatter.png", "modality_gap_bars.png", "strength_alignment.png", "strength_modality_gap.png"):
        return "metrics/plot_geometry.py", "results/geometry_metrics.csv", "表征空间几何"
    if path.name == "loss.png" and "lit_selfcheck" in path.as_posix():
        return "train/selfcheck_lit.py", (path.parent/"epochs.csv").relative_to(root).as_posix(), "训练设置与收敛"
    if name.startswith("perturb_examples_"):
        return "tools/make_perturb_examples.py", "configs/perturb.yaml;configs/models.yaml", "数据与扰动设置"
    if name.startswith("gantt"):
        return "", "docs/分工方案.md", "附录：实验组织"
    return "", "", "待上游登记"


def upstream_caption(path, source, root):
    caption = f"{path.stem}：组内上游已交付的图。"
    observations = []
    sources = source.split(";")
    csv = root / sources[0]
    if csv.suffix == ".csv" and csv.exists():
        data = pd.read_csv(csv)
        for dataset in ("flickr", "coco"):
            if dataset in path.name and "dataset" in data: data = data[data.dataset == dataset]
        for split in ("val", "test"):
            if "_"+split in path.name and "split" in data: data = data[data.split == split]
        for model in ("clip", "siglip"):
            if model in path.name and "model" in data: data = data[data.model == model]
        for modality in ("image", "text"):
            if modality in path.name and "modality" in data: data = data[data.modality == modality]
        if "strength" in path.name and "cond" in data:
            if "noise" in path.name: data = data[data.cond.isin(("noise_lo", "noise", "noise_hi"))]
            elif "occl" in path.name: data = data[data.cond.isin(("occl_lo", "occl", "occl_hi"))]
            else: data = data[data.cond.isin(("noise_lo", "noise", "noise_hi", "occl_lo", "occl", "occl_hi"))]
        metrics = [c for c in ("i2t_R1", "t2i_R1", "gap", "align_cos", "mean", "loss", "drop_pp") if c in data]
        for metric in metrics[:2]:
            values = pd.to_numeric(data[metric], errors="coerce").dropna()
            if not values.empty:
                observations.append(f"对应来源记录中 {metric} 的范围为 {values.min():.4f} 至 {values.max():.4f}（沿用原始单位）。")
        if not observations: observations.append(f"对应来源表有 {len(data)} 条记录，具体绘图筛选以生成脚本为准。")
        observations.append("此处描述来源记录的数值范围；因果解释和假设判定由 P8/P10 汇总。")
    else:
        observations = ["该图已存在于组内仓库，登记其原有文件路径供报告引用。",
                        "未提供可核对的数值 CSV，不能在此确认数值一致性。"]
    return caption, observations


def build_docs(root, projection_dir, docs_dir):
    root, projection_dir, docs_dir = Path(root).resolve(), Path(projection_dir).resolve(), Path(docs_dir).resolve()
    rows, captions, metadata = [], [], {}
    for path in sorted((projection_dir / "coordinates").glob("*.json")):
        meta = json.loads(path.read_text(encoding="utf-8"))
        if meta["config"]["demo"]: continue
        for file in meta["figures"]:
            figure = projection_dir/file
            if figure.exists(): metadata[figure.resolve()] = (meta, path)
    figures = set(metadata)
    for base in (root/"docs", root/"results"):
        figures |= {p.resolve() for p in base.rglob("*.png")
                    if not any(part in ("p9_demo", "p9_smoke", "report_tables", ".ipynb_checkpoints") for part in p.parts)}
    previous = {}
    index = docs_dir/"figure_index.csv"
    if index.exists():
        old = pd.read_csv(index).fillna("")
        previous = dict(zip(old["文件路径"], old["图号"]))
    counter = max([int(v[1:]) for v in previous.values() if v.startswith("F") and v[1:].isdigit()] or [0])
    for figure in sorted(figures):
        path = figure.relative_to(root).as_posix()
        number = previous.get(path)
        if not number:
            counter += 1
            number = f"F{counter:03d}"
        if figure in metadata:
            meta, meta_path = metadata[figure]
            source = (projection_dir/meta["coordinates"]).relative_to(root).as_posix()
            script, section = "viz/tsne_umap.py", "表征空间可视化"
            caption, observations = projection_caption(meta, root/source, figure.stem.endswith("_classes"))
            status = "已登记：含参数、来源哈希与坐标"
        else:
            script, source, section = upstream_source(figure, root)
            caption, observations = upstream_caption(figure, source, root)
            status = "已登记上游来源，未重跑" if script and source else "待上游补生成脚本/数据来源"
        rows.append(dict(zip(INDEX_COLUMNS, (number, path, script, source, section, caption, status))))
        captions.append(f"## {number} — {figure.stem}\n\n图注：{caption}\n\n" + "\n\n".join(observations)
                        + f"\n\n文件：`{path}`；来源：`{source or '待补'}`。\n")
    write_csv(index, pd.DataFrame(rows, columns=INDEX_COLUMNS))
    atomic_text(docs_dir/"figure_captions.md", "# 图注与观察\n\n"
                "图号按文件路径稳定分配；PNG/SVG 是同一张图的两种格式，共用图号。P9 二维投影只作定性辅助，原空间指标以 P7 为准。\n\n"
                + "\n".join(captions))
    return rows, metadata


def audit(root, projection_dir, docs_dir, rows, metadata, seed=42):
    root, projection_dir, docs_dir = Path(root).resolve(), Path(projection_dir).resolve(), Path(docs_dir).resolve()
    rng = np.random.default_rng(seed)
    checks = []
    tables_path = root/"results/report_tables/manifest.json"
    tables = json.loads(tables_path.read_text(encoding="utf-8")) if tables_path.exists() else {"cells": [], "missing": ["all"]}
    @lru_cache(maxsize=None)
    def source_frame(source):
        parts = source.split("#")
        path = Path(parts[0])
        if not path.is_absolute(): path = root/path
        return pd.read_excel(path, sheet_name=parts[1]) if len(parts)>1 else pd.read_csv(path)
    for i in rng.choice(len(tables["cells"]), min(10, len(tables["cells"])), replace=False):
        cell = tables["cells"][int(i)]
        raw = source_frame(cell["source"]).iloc[cell["source_row"]][cell["source_column"]]
        display = pd.read_csv(root/f"results/report_tables/{cell['table']}.csv", dtype=str).iloc[cell["row"]][cell["column"]]
        passed = np.isclose(float(raw), cell["raw_value"], atol=1e-12, rtol=1e-10) and display == format_value(cell["column"], raw)
        checks.append({"kind": "number", "item": f"{cell['table']} row {cell['row']+1} {cell['column']}",
                       "source": f"{cell['source']} data row {cell['source_row']+1}", "raw": float(raw), "display": display, "pass": bool(passed)})
    # The figure sample covers actual projection artifacts; upstream figures are indexed without claiming regeneration.
    candidates = sorted(metadata)
    hashes = {}
    for i in rng.choice(len(candidates), min(10, len(candidates)), replace=False):
        figure = candidates[int(i)]
        meta, meta_path = metadata[figure]
        passed, details = True, []
        for name, expected in meta["artifacts"].items():
            path = projection_dir/name
            if path not in hashes: hashes[path] = sha256(path) if path.exists() else None
            passed &= hashes[path] == expected
        for record in meta["config"]["sources"]:
            path = root/"features"/record["path"]
            if path not in hashes: hashes[path] = sha256(path) if path.exists() else None
            passed &= hashes[path] == record["sha256"]
        frame = pd.read_csv(projection_dir/meta["coordinates"])
        passed &= len(frame) == meta["points"] and np.isfinite(frame[["x", "y"]]).all().all()
        passed &= (frame[frame.modality == "text"].source_row % 5 == 0).all()
        with Image.open(figure) as img:
            dpi = img.info.get("dpi", (0, 0))
            passed &= min(dpi) >= 299
        checks.append({"kind": "figure", "item": figure.relative_to(root).as_posix(),
                       "source": (projection_dir/meta["coordinates"]).relative_to(root).as_posix(),
                       "pass": bool(passed), "checks": "input/output SHA256, coordinate rows, finite values, first caption, 300 dpi, PNG/SVG"})
    number_count = sum(c["kind"]=="number" for c in checks)
    figure_count = sum(c["kind"]=="figure" for c in checks)
    result = {"seed": seed, "checks": checks, "numbers_checked": number_count, "figures_checked": figure_count,
              "passed": all(c["pass"] for c in checks) and number_count==10 and figure_count==10,
              "missing_tables": tables["missing"]}
    write_json(docs_dir/"consistency_check.json", result)
    atomic_text(docs_dir/"consistency_check.md", "# 图表一致性抽查\n\n"
                f"种子 {seed}；抽查 {number_count} 个数字、{figure_count} 张图；结果：{'通过' if result['passed'] else '未完成或有失败'}。\n\n"
                "数字回查原始 CSV/XLSX 与报告 CSV 的格式化值；图回查输入/输出 SHA-256、坐标文件、首条 caption、300 dpi 和 SVG。"
                "这不是对全部上游实验的科学结论验收。\n\n"
                + "\n".join(f"- {'PASS' if c['pass'] else 'FAIL'} {c['item']} ← `{c['source']}`" for c in checks)
                + f"\n\n仍缺上游表格：{', '.join(tables['missing']) or '无'}。后续到齐后重新运行本脚本。\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--projection-dir", type=Path, default=REPO_ROOT/"results/p9")
    parser.add_argument("--docs-dir", type=Path, default=REPO_ROOT/"docs")
    args = parser.parse_args()
    rows, metadata = build_docs(args.root, args.projection_dir, args.docs_dir)
    result = audit(args.root, args.projection_dir, args.docs_dir, rows, metadata)
    print(f"Indexed {len(rows)} figures. Audit: {result['numbers_checked']} numbers, {result['figures_checked']} figures; passed={result['passed']}")


if __name__ == "__main__":
    main()
