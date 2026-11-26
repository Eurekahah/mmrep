"""Summarize submitted RunLogger JSONL and checkpoint metadata without double-counting runs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from eval.p5_common import REPO_ROOT, write_csv, write_json, atomic_text, sha256
from viz.make_report_tables import markdown, path_reference

FIELDS = ["name", "model", "setting", "wall_sec", "gpu_mem_peak_MB", "status", "epoch", "batch",
          "time", "seed", "dataset", "split", "cond", "source", "meta_source", "record_kind"]


def first(record, names):
    return next((record[k] for k in names if k in record and record[k] is not None), None)


def summarize(logs, ckpt, out):
    logs, ckpt, out = Path(logs), Path(ckpt), Path(out)
    paths = sorted(logs.glob("*_runs.jsonl"))
    if not paths and (logs / "runs.jsonl").exists():
        paths = [logs / "runs.jsonl"]
    rows, seen, duplicate_count, sources = [], {}, 0, {}
    for path in paths:
        sources[path_reference(path)] = sha256(path)
        for line_no, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
                if not isinstance(rec, dict):
                    raise ValueError("expected JSON object")
            except (ValueError, TypeError) as exc:
                raise ValueError(f"{path}:{line_no}: {exc}") from exc
            identity = json.dumps(rec, sort_keys=True, ensure_ascii=False)
            reference = f"{path_reference(path)}:{line_no}"
            if identity in seen:
                rows[seen[identity]]["source"] += ";" + reference
                duplicate_count += 1
                continue
            row = {k: rec.get(k) for k in FIELDS}
            row.update(epoch=first(rec, ("epoch", "epochs", "probe_epochs")),
                       batch=first(rec, ("batch", "batch_size", "probe_batch_size")),
                       source=reference, record_kind="run", status=rec.get("status", "unknown"))
            for key in ("wall_sec", "gpu_mem_peak_MB"):
                if row[key] is not None and (not isinstance(row[key], (int, float)) or row[key] < 0 or not __import__('math').isfinite(row[key])):
                    raise ValueError(f"{reference}: invalid {key}")
            seen[identity] = len(rows)
            rows.append(row)
    metas = sorted(ckpt.glob("*/*/meta.json"))
    for path in metas:
        meta = json.loads(path.read_text(encoding="utf-8-sig"))
        sources[path_reference(path)] = sha256(path)
        model, setting = path.parent.parent.name, path.parent.name
        matches = [row for row in rows if row["model"] == model and row["setting"] == setting
                   and str(row["name"]).startswith("train")
                   and row["status"] in ("ok", "success", "completed")
                   and (meta.get("seed") is None or row["seed"] == meta["seed"])]
        if len(matches) == 1:
            row = matches[0]
            row["meta_source"] = path_reference(path)
            # Existing run measurements take precedence; metadata complements training config only.
            if row["epoch"] is None: row["epoch"] = first(meta, ("epochs", "epoch"))
            if row["batch"] is None: row["batch"] = first(meta, ("batch_size", "batch"))
        else:
            rows.append(dict(name="checkpoint_meta", model=model, setting=setting,
                             wall_sec=first(meta, ("wall_sec", "total_time_sec", "total_time")),
                             gpu_mem_peak_MB=first(meta, ("gpu_mem_peak_MB", "peak_memory_mb")),
                             status="metadata_only", epoch=first(meta, ("epochs", "epoch")),
                             batch=first(meta, ("batch_size", "batch")), seed=meta.get("seed"),
                             source=path_reference(path), meta_source=path_reference(path), record_kind="metadata"))
    frame = pd.DataFrame(rows, columns=FIELDS)
    write_csv(out / "compute_log.csv", frame)
    received = sorted(p.stem.removesuffix("_runs") for p in paths if p.name != "runs.jsonl")
    missing = sorted(set(("P2", "P3", "P4", "P6")) - set(received))
    metadata_keys = {f"{p.parent.parent.name}/{p.parent.name}" for p in metas}
    expected = {f"{m}/{s}" for m in ("clip", "siglip") for s in ("lit", "lora_r8", "lora_r16")}
    info = {"sources": sources, "rows": len(rows), "exact_duplicates_removed": duplicate_count,
            "missing_log_owners": missing, "missing_checkpoint_meta": sorted(expected-metadata_keys),
            "complete": not missing and expected.issubset(metadata_keys)}
    write_json(out / "compute_log_manifest.json", info)
    display = frame.fillna("—").copy()
    for key in ("wall_sec", "gpu_mem_peak_MB"):
        display[key] = display[key].map(lambda v: f"{float(v):.2f}" if v != "—" else v)
    atomic_text(out / "compute_log.md", "# 实验日志汇总\n\n"
                f"记录 {len(rows)} 条；去除完全相同的副本 {duplicate_count} 条。缺少人员日志：{', '.join(missing) or '无'}。\n\n"
                f"缺少 checkpoint 元数据：{', '.join(info['missing_checkpoint_meta']) or '无'}。\n\n"
                "`metadata_only` 是训练元数据，不作为另一条运行累计耗时；失败、重试与未知状态保留。峰值显存沿用 RunLogger 的 MiB 口径（列名按约定保留 MB），不能相加。空值表示上游未记录。\n\n"
                + markdown(display))
    return info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logs-dir", type=Path, default=REPO_ROOT / "logs")
    parser.add_argument("--ckpt-dir", type=Path, default=REPO_ROOT / "ckpt")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "results")
    args = parser.parse_args()
    print(json.dumps(summarize(args.logs_dir, args.ckpt_dir, args.out_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
