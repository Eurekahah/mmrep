"""One command for validation, retrieval, drift, figures and numeric notes."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import platform
import sys
import time

import numpy as np
import pandas as pd

from .drift import evaluate_drift
from .eval_retrieval import evaluate
from .p5_common import (MAIN_CONDS, SETTINGS, add_input_arguments, atomic_text,
                        cli_main, discover_features, inspect_features, resolve_inputs, write_json)


def _markdown_table(frame, columns, digits=2):
    # Avoid pandas.to_markdown's optional tabulate dependency.
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for record in frame[columns].to_dict("records"):
        values = []
        for column in columns:
            value = record[column]
            values.append(f"{value:.{digits}f}" if isinstance(value, (float, np.floating)) else str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_observations(out_dir, report, retrieval, drops, drift):
    present = set(retrieval["setting"])
    lines = ["# P5 实验数字记录", "", f"生成时间（UTC）：{datetime.now(timezone.utc).isoformat()}", "",
             f"输入特征：{report['available_files']} / {report['expected_files']} 份；"
             f"当前设置：{', '.join(s for s in SETTINGS if s in present)}。", "",
             "本文只记录实际计算结果，不自动判定实验假设。", ""]
    if report.get("scope", "all") != "all":
        model = report["scope"]
        deferred = "SigLIP2" if model == "clip" else "CLIP"
        lines += [f"本次为 {model.upper()} 单模型阶段，完整覆盖按该模型的 64 份特征检查。"
                  f"{deferred} 留待后续补充，本次不进行跨模型比较。", ""]
    if present == {"pretrained"}:
        lines += ["当前仅有 pretrained 特征，尚不能比较 LiT/LoRA 或 LoRA 秩的影响。", ""]
    lines += ["## 检索主条件", "", "Recall 为百分数，val 为主要结果，test 用于复核。", "",
              _markdown_table(retrieval[retrieval["cond"].isin(MAIN_CONDS)],
                              ["dataset", "split", "model", "setting", "cond", "i2t_R1", "t2i_R1"]), "",
              "## 扰动掉点", "", "drop_pp 为 clean 减扰动的百分点差，drop_relative_pct 为相对百分比下降。"
              "负数表示该扰动条件的 Recall 高于 clean；clean 为零时相对下降留空。", ""]
    for direction in ("i2t", "t2i"):
        sub = drops[(drops["metric"] == f"{direction}_R1") & drops["cond"].isin(("noise", "occl"))]
        lines += [f"### {direction} R@1", "", _markdown_table(sub,
                  ["dataset", "split", "model", "setting", "cond", "drop_pp", "drop_relative_pct"]), ""]
    baseline = drift[(drift["ref"] == "self_clean") & drift["cond"].isin(("noise", "occl"))]
    lines += ["## pretrained 扰动相似度", "", _markdown_table(baseline,
              ["dataset", "split", "model", "cond", "mean", "std", "p5", "p50", "p95"], digits=4), ""]
    tuned = drift[(drift["setting"] != "pretrained") & (drift["cond"] == "clean")]
    if not tuned.empty:
        lines += ["## 微调后 clean 表征相似度", "", _markdown_table(tuned,
                  ["dataset", "split", "model", "setting", "modality", "ref", "mean", "std"], digits=4), ""]
    lines += ["## 口径与局限", "", "- 全部五条 caption 用于检索及文本漂移，向量先 L2 归一化。",
              "- std 使用总体标准差（ddof=0），分位数为逐样本余弦相似度的 5/50/95 百分位。",
              "- 同条件比较与 pretrained clean 比较分开记录，文本在图像扰动下复用 clean 特征。",
              "- CLIP 与 SigLIP2 的输入尺寸、预处理、训练来源和维度不同。跨模型讨论应围绕变化量，"
              "不能把差异单独归因于损失函数。", "- 缺失组不填零、不模拟。完整缺失清单见 p5_feature_check.json。", ""]
    atomic_text(Path(out_dir) / "p5_observations.md", "\n".join(lines))


def run_pipeline(features_root, out_dir, *, splits_root=None, block_size=512, threads=2,
                 require_complete=False, validate_only=False, no_plots=False, model="all"):
    started = time.perf_counter()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    state = {"started_utc": datetime.now(timezone.utc).isoformat(), "status": "running",
             "features_root": str(Path(features_root).resolve()), "output_dir": str(out_dir.resolve()),
             "python": sys.version, "python_executable": sys.executable, "platform": platform.platform(),
             "numpy": np.__version__, "pandas": pd.__version__, "block_size": block_size,
             "threads": threads, "split_check": splits_root is not None, "model_scope": model}
    write_json(out_dir / "p5_run.json", state)
    try:
        if block_size < 1 or threads < 1:
            raise ValueError("block_size and threads must be positive")
        keys = discover_features(features_root, model)
        print(f"Validating {len(keys)} feature files ...", flush=True)
        report = inspect_features(features_root, keys, splits_root, model=model)
        write_json(out_dir / "p5_feature_check.json", report)
        state["coverage"] = {k: report[k] for k in ("available_files", "expected_files", "complete")}
        print(f"Coverage: {report['available_files']}/{report['expected_files']} "
              f"({'complete' if report['complete'] else 'partial; missing groups will not be fabricated'})", flush=True)
        if require_complete and not report["complete"]:
            raise ValueError(f"Complete evaluation for scope '{model}' needs {report['expected_files']} files; "
                             f"{len(report['missing'])} are missing. "
                             "See p5_feature_check.json, or omit --require-complete for available features.")
        if validate_only:
            state["status"] = "validated"
        else:
            stage = time.perf_counter()
            retrieval, drops = evaluate(features_root, out_dir, keys=keys, splits_root=splits_root,
                                        block_size=block_size, threads=threads)
            state["retrieval_seconds"] = time.perf_counter() - stage
            stage = time.perf_counter()
            drift, missing = evaluate_drift(features_root, out_dir, keys=keys, splits_root=splits_root)
            state["drift_seconds"] = time.perf_counter() - stage
            figures = []
            if not no_plots:
                from .plot_p5 import plot_all
                figures = plot_all(out_dir)
            write_observations(out_dir, report, retrieval, drops, drift)
            state.update(status="completed", retrieval_val_rows=int((retrieval["split"] == "val").sum()),
                         retrieval_test_rows=int((retrieval["split"] == "test").sum()),
                         drop_rows=len(drops), drift_rows=len(drift), figure_files=len(figures),
                         missing_drift_references=len(missing))
    except Exception as exc:
        state.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        state["elapsed_seconds"] = time.perf_counter() - started
        state["finished_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(out_dir / "p5_run.json", state)
    print(f"P5 {state['status']}: {out_dir.resolve()} ({state['elapsed_seconds']:.1f}s)", flush=True)
    return state


def main():
    parser = add_input_arguments(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--block-size", type=int, default=512)
    parser.add_argument("--require-complete", action="store_true",
                        help="Require every NPZ in the selected scope: 64 for one model, 128 for all")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    root, out, splits = resolve_inputs(args)
    run_pipeline(root, out, splits_root=splits, block_size=args.block_size, threads=args.threads,
                 require_complete=args.require_complete, validate_only=args.validate_only,
                 no_plots=args.no_plots, model=args.model)


if __name__ == "__main__":
    cli_main(main)
