"""One-command P9 rebuild; missing upstream tables/logs are reported explicitly."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd

from eval.p5_common import REPO_ROOT, write_json, write_csv, sha256
from tools.summarize_logs import summarize
from viz.make_report_tables import build_tables
from viz.build_figure_docs import build_docs, audit
from viz.tsne_umap import groups
from viz.validate_inputs import validate


def summarize_projections(root):
    out = Path(root)/"results/p9"
    expected = {f"{g.model}_{m}_{g.name}_{g.dataset}" for g in groups(("clip", "siglip"), ("flickr", "coco")) for m in ("tsne", "umap")}
    records, names, invalid = [], set(), []
    for path in sorted((out/"coordinates").glob("*.json")):
        if path.stem not in expected: continue
        meta = json.loads(path.read_text(encoding="utf-8"))
        if meta["config"]["demo"] or meta["config"]["n_images"] != 1000:
            invalid.append(path.name)
            continue
        if not all((out/p).exists() and sha256(out/p)==h for p, h in meta["artifacts"].items()):
            invalid.append(path.name)
            continue
        names.add(path.stem)
        records.append(meta)
    result = {"expected_projections": len(expected), "completed_projections": len(names),
              "figure_count": sum(len(m["figures"]) for m in records), "missing": sorted(expected-names),
              "invalid": invalid, "complete": names==expected and not invalid, "completed": records}
    write_json(out/"manifest.json", result)
    write_csv(out/"projection_runtime.csv", pd.DataFrame([
        {"model": r["config"]["model"], "dataset": r["config"]["dataset"], "method": r["config"]["method"],
         "group": r["config"]["group"], "wall_sec": r["wall_sec"], "points": r["points"]} for r in records]))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-projections", action="store_true", help="Only rebuild report material from existing projections")
    args = parser.parse_args()
    validate()
    if not args.skip_projections:
        subprocess.run([sys.executable, "-m", "viz.tsne_umap", "--resume"], cwd=REPO_ROOT, check=True)
    projection = summarize_projections(REPO_ROOT)
    tables = build_tables(REPO_ROOT/"results", REPO_ROOT/"results/report_tables")
    logs = summarize(REPO_ROOT/"logs", REPO_ROOT/"ckpt", REPO_ROOT/"results")
    rows, metadata = build_docs(REPO_ROOT, REPO_ROOT/"results/p9", REPO_ROOT/"docs")
    checks = audit(REPO_ROOT, REPO_ROOT/"results/p9", REPO_ROOT/"docs", rows, metadata)
    result = {"projections_complete": projection["complete"], "figures": projection["figure_count"],
              "indexed_figures": len(rows), "table_count": len(tables["tables"]), "tables_missing": tables["missing"],
              "summary_workbook_available": tables["summary_workbook_available"],
              "missing_logs": logs["missing_log_owners"], "consistency_passed": checks["passed"],
              "complete": projection["complete"] and tables["complete"] and logs["complete"] and checks["passed"]}
    write_json(REPO_ROOT/"results/p9/delivery_status.json", result)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
