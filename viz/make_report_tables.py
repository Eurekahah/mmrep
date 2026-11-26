"""Build traceable report tables without changing upstream CSVs or inventing missing results."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from eval.p5_common import REPO_ROOT, sha256, write_json, atomic_text

KEYS = ["dataset", "split", "model", "setting", "cond", "modality", "ref", "seed", "k"]


def path_reference(path):
    path = Path(path).resolve()
    return path.relative_to(REPO_ROOT).as_posix() if path.is_relative_to(REPO_ROOT) else path.as_posix()


def resolve_reference(source):
    path = Path(source.split("#")[0])
    return path if path.is_absolute() else REPO_ROOT/path


SPECS = {
    "retrieval": (["retrieval_*.csv"], lambda c: bool(re.fullmatch(r"(?:i2t|t2i)_R(?:1|5|10)", c))),
    "cifar100": (["cifar100_summary.csv", "cifar100_top1.csv"], lambda c: c in ("top1", "top1_mean", "top1_std", "top5", "top5_mean", "top5_std")),
    "alignment_uniformity": (["geometry_metrics.csv"], lambda c: c.startswith(("align_", "unif_"))),
    "modality_gap": (["geometry_metrics.csv"], lambda c: c == "gap" or c.startswith("gap_")),
    "prototypes": (["prototype_metrics.csv"], lambda c: c.startswith("proto_")),
    "drift": (["drift_cosine.csv"], lambda c: c in ("mean", "std", "p5", "p50", "p95", "n")),
    "svd": (["svd_energy.csv"], lambda c: bool(re.fullmatch(r"top\d+_energy", c)) or c in ("dims90", "dims95", "eff_rank", "dims90_ratio", "eff_rank_ratio")),
}


def is_percent(column):
    return bool(re.fullmatch(r"(?:i2t|t2i)_R(?:1|5|10)", column)) or column in {
        "top1", "top1_mean", "top1_std", "top5", "top5_mean", "top5_std",
        "gap_linsep", "proto_nn_acc", "proto_nn_acc_weighted", "Recall@1", "Recall@5", "Recall@10"}


def is_recall(column):
    # P5 explicitly exports 0..100, whereas P6/P7 accuracy columns are 0..1.
    return bool(re.fullmatch(r"(?:i2t|t2i)_R(?:1|5|10)", column)) or column in {"Recall@1", "Recall@5", "Recall@10"}


def format_value(column, value):
    if pd.isna(value):
        return "—"
    if column in KEYS or isinstance(value, str):
        return str(value)
    if is_percent(column):
        number = float(value)
        upper = 100 if is_recall(column) else 1
        if not 0 <= number <= upper:
            raise ValueError(f"{column} must be in [0,{upper}], got {number}; do not guess percent units")
        return f"{number if is_recall(column) else 100 * number:.1f}%"
    if column in ("n", "dims90", "dims95"):
        return str(int(value))
    if not np.isfinite(float(value)):
        raise ValueError(f"Non-finite {column}: {value}")
    return f"{float(value):.4f}"


def markdown(frame):
    def escape(v):
        return str(v).replace("|", "\\|").replace("\n", " ")
    lines = ["| " + " | ".join(map(escape, frame.columns)) + " |", "| " + " | ".join("---" for _ in frame.columns) + " |"]
    return "\n".join(lines + ["| " + " | ".join(map(escape, row)) + " |" for row in frame.itertuples(index=False, name=None)]) + "\n"


def read_csv(path):
    frame = pd.read_csv(path)
    if "split" not in frame and "dataset" in frame:
        frame["split"] = "test" if "test" in path.stem else "val"
    return frame


def emit(name, frame, columns, source, out, provenance):
    selected = frame[columns].copy()
    selected = selected.dropna(subset=[c for c in columns if c not in KEYS], how="all")
    if selected.empty:
        return None
    keys = [k for k in KEYS if k in selected]
    if keys:
        selected = selected.sort_values(keys, kind="stable")
        if selected.duplicated(keys).any():
            raise ValueError(f"{name}: duplicate logical keys in {source}")
    formatted = selected.copy().astype(object)
    for row_num, (idx, row) in enumerate(selected.iterrows()):
        for column in columns:
            formatted.loc[idx, column] = format_value(column, row[column])
            if column not in KEYS and pd.notna(row[column]):
                provenance.append({"table": name, "row": row_num, "column": column,
                                   "raw_value": float(row[column]), "display": formatted.loc[idx, column],
                                   "source": source, "source_row": int(idx), "source_column": column})
    atomic_text(out / f"{name}.csv", formatted.to_csv(index=False, lineterminator="\n"), "utf-8-sig")
    atomic_text(out / f"{name}.md", f"<!-- Source: {source}; ratios are formatted as percentages only for recall/accuracy. -->\n\n" + markdown(formatted))
    return {"name": name, "rows": len(selected), "source": source, "columns": columns}


def build_tables(results, out):
    results, out = Path(results), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    workbook = results / "summary_table.xlsx"
    sheets = pd.read_excel(workbook, sheet_name=None) if workbook.exists() else {}
    provenance, tables, missing, sources, conflicts = [], [], [], {}, []
    for name, (patterns, predicate) in SPECS.items():
        paths = []
        for pattern in patterns:
            found = sorted(p for p in results.glob(pattern) if p.is_file())
            paths.extend(found)
            if name == "cifar100" and found:
                break
        candidates = [(path_reference(p), read_csv(p)) for p in paths]
        if not candidates:
            candidates = [(f"{path_reference(workbook)}#{sheet}", f.copy()) for sheet, f in sheets.items()
                          if sheet != "strength" and any(predicate(c) for c in f.columns)]
        produced = False
        for source, frame in candidates:
            metrics = [c for c in frame if predicate(c)]
            if not metrics:
                continue
            base = resolve_reference(source)
            sources[source] = sha256(base)
            columns = [k for k in KEYS if k in frame] + metrics
            output_name = name if len(candidates) == 1 else name + "_" + re.sub(r"[^\w-]", "_", source.split("#")[-1] if "#" in source else base.stem)
            record = emit(output_name, frame, columns, source, out, provenance)
            if record:
                tables.append(record)
                produced = True
            # A summary workbook is a cross-check, not permission to overwrite conflicting measurements.
            for sheet, other in sheets.items():
                common_metrics = [c for c in metrics if c in other]
                common_keys = [k for k in KEYS if k in frame and k in other]
                if not common_metrics or not {"model", "setting"}.issubset(common_keys):
                    continue
                left = frame[common_keys + common_metrics].drop_duplicates()
                right = other[common_keys + common_metrics].drop_duplicates()
                if left.duplicated(common_keys).any() or right.duplicated(common_keys).any():
                    continue  # Different aggregation grain; do not compare unrelated rows.
                merged = left.merge(right, on=common_keys, suffixes=("_raw", "_summary"))
                for c in common_metrics:
                    a, b = merged[c+"_raw"], merged[c+"_summary"]
                    valid = a.notna() & b.notna()
                    if valid.any() and not np.allclose(a[valid].astype(float), b[valid].astype(float), rtol=1e-7, atol=1e-8):
                        conflicts.append(f"{source} vs summary_table.xlsx#{sheet}: {c}")
        if not produced:
            missing.append(name)
    if "strength" in sheets:
        frame = sheets["strength"]
        source = f"{path_reference(workbook)}#strength"
        sources[source] = sha256(workbook)
        record = emit("strength", frame, list(frame.columns), source, out, provenance)
        if record: tables.append(record)
        else: missing.append("strength")
    else:
        missing.append("strength")
    manifest = {"tables": tables, "missing": missing, "conflicts": conflicts,
                "summary_workbook_available": workbook.exists(), "sources": sources,
                "complete": not missing and not conflicts and workbook.exists(), "cells": provenance}
    write_json(out / "manifest.json", manifest)
    status = "全部表格已生成" if manifest["complete"] else "当前为部分交付"
    atomic_text(out / "README.md", f"# P9 报告表格\n\n{status}。缺少：{', '.join(missing) or '无'}。总表：{'已收到' if workbook.exists() else '未收到'}。\n\n"
                "CSV 与 Markdown 使用相同的显示数值；原始数值、来源文件和行号保存在 manifest.json。源文件不被改写。\n\n"
                + "\n".join(f"- [{t['name']}]({t['name']}.md)：{t['rows']} 行" for t in tables) + "\n"
                + ("\n检测到总表与原始 CSV 冲突：\n" + "\n".join(conflicts) if conflicts else ""))
    if conflicts:
        raise ValueError("Workbook/CSV conflicts: " + "; ".join(conflicts))
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=REPO_ROOT / "results")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "results" / "report_tables")
    args = parser.parse_args()
    result = build_tables(args.results_dir, args.out_dir)
    print(f"Tables: {len(result['tables'])}; missing: {result['missing']}")


if __name__ == "__main__":
    main()
