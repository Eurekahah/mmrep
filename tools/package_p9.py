"""Package P9 code/results and the small upstream sources needed to inspect the handoff."""
import argparse
import json
from pathlib import Path
import zipfile

from eval.p5_common import REPO_ROOT, sha256, write_json


def package(out):
    root, out = REPO_ROOT, Path(out).resolve()
    status = json.loads((root/"results/p9/delivery_status.json").read_text(encoding="utf-8"))
    if not status["projections_complete"] or not status["consistency_passed"]:
        raise ValueError("Finish and validate the P9 projections before packaging")
    files = set()
    for folder in ("viz", "results/p9", "results/report_tables", "results/figs", "results/p2_stages", "results/lit_selfcheck"):
        files |= {p for p in (root/folder).rglob("*") if p.is_file() and "__pycache__" not in p.parts
                  and ".ipynb_checkpoints" not in p.parts and p.suffix not in (".pyc",)}
    files |= set((root/"results").glob("*.csv"))
    files |= set((root/"results").glob("*.json"))
    files |= {root/p for p in ("tools/summarize_logs.py", "tools/package_p9.py", "tools/logger.py", "tools/__init__.py",
                               "tools/make_dummy_features.py", "tests/test_p9.py", "pytest.ini",
                               "eval/__init__.py", "eval/p5_common.py", "docs/p9_handoff.md",
                               "docs/figure_captions.md", "docs/figure_index.csv", "docs/consistency_check.md",
                               "docs/consistency_check.json", "docs/CONTRACT.md", "results/compute_log.md",
                               "configs/models.yaml", "configs/perturb.yaml", "splits/flickr_split.json", "splits/coco_split.json")}
    files |= set((root/"docs").glob("*.png"))
    files |= set((root/"docs").glob("p9_*.json"))
    files |= set((root/"logs").glob("*_runs.jsonl"))
    files |= set((root/"ckpt").glob("*/*/meta.json"))
    files = sorted(p for p in files if p.exists())
    out.parent.mkdir(parents=True, exist_ok=True)
    manifest = {"status": status, "files": [{"path": p.relative_to(root).as_posix(), "bytes": p.stat().st_size,
                                             "sha256": sha256(p)} for p in files]}
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in files: z.write(p, p.relative_to(root).as_posix())
        z.writestr("P9_MANIFEST.json", json.dumps(manifest, ensure_ascii=False, indent=2))
        z.writestr("P9_README.txt", "P9 交付包（2026-11-26）\n先阅读 docs/p9_handoff.md，再阅读 viz/README.md。\n"
                   "完整特征需从组内云盘 features/ 获取，本包不重复包含特征和模型权重。\n"
                   "P8 表/图及 P3/P6 JSONL 未到齐，当前整体交付状态为部分完成；正式降维图已齐。\n")
    # Read every member back and compare the digest, not just ZIP's central directory.
    import hashlib
    with zipfile.ZipFile(out) as z:
        for record in manifest["files"]:
            if hashlib.sha256(z.read(record["path"])).hexdigest() != record["sha256"]:
                raise ValueError("ZIP verification failed: "+record["path"])
    receipt = {"file": out.name, "bytes": out.stat().st_size, "sha256": sha256(out), "members": len(files)+2, "verified": True}
    write_json(out.with_suffix(".json"), receipt)
    print(json.dumps(receipt, indent=2))
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=REPO_ROOT.parent/"P9_交付_2026-11-26.zip")
    package(parser.parse_args().out)
