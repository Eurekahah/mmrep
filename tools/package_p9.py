"""Package only P9's required handoff outputs; code is delivered through GitHub."""
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
    # Whitelist final artifacts. Reproducibility caches and upstream originals stay
    # at their existing locations; copying them again is not a P9 deliverable.
    files = {root/p for p in ("docs/p9_handoff.md", "docs/p9_submission_checklist.md",
                              "docs/figure_captions.md", "docs/figure_index.csv", "docs/consistency_check.md",
                              "results/compute_log.csv", "results/compute_log.md")}
    files |= {p for p in (root/"results/p9/figures").glob("*") if p.suffix in (".png", ".svg")}
    files |= {p for p in (root/"results/report_tables").glob("*") if p.suffix in (".md", ".csv")}
    missing = [str(p.relative_to(root)) for p in files if not p.is_file()]
    if missing:
        raise ValueError("Missing required handoff files: "+", ".join(missing))
    pngs = {p.stem for p in files if p.suffix == ".png"}
    svgs = {p.stem for p in files if p.suffix == ".svg"}
    if len(pngs) != status["figures"] or pngs != svgs:
        raise ValueError("Formal PNG/SVG coverage differs from the validated result")
    files = sorted(files)
    out.parent.mkdir(parents=True, exist_ok=True)
    manifest = {"status": status, "files": [{"path": p.relative_to(root).as_posix(), "bytes": p.stat().st_size,
                                             "sha256": sha256(p)} for p in files]}
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in files: z.write(p, p.relative_to(root).as_posix())
        z.writestr("P9_README.txt", "P9 精简交付包\n先阅读 docs/p9_submission_checklist.md 与 docs/p9_handoff.md。\n"
                   "代码与复现说明：https://github.com/Eurekahah/mmrep/pull/2\n"
                   "本包只放正式 PNG/SVG、报告表格、图注、全组图号索引、核查记录和日志汇总。\n"
                   "原始特征、权重、他人原始结果、坐标缓存及机器检查 JSON 不重复上交。\n"
                   "P3/P6 日志已补齐；仍待 P8 的 SVD 表/图和总表，补齐后再更新报告材料。\n")
    # Read every member back and compare the digest, not just ZIP's central directory.
    import hashlib
    with zipfile.ZipFile(out) as z:
        for record in manifest["files"]:
            if hashlib.sha256(z.read(record["path"])).hexdigest() != record["sha256"]:
                raise ValueError("ZIP verification failed: "+record["path"])
    receipt = {"file": out.name, "bytes": out.stat().st_size, "sha256": sha256(out), "members": len(files)+1, "verified": True}
    write_json(out.with_suffix(".json"), receipt)
    print(json.dumps(receipt, indent=2))
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=REPO_ROOT.parent/"P9_交付_精简版_2026-11-27.zip")
    package(parser.parse_args().out)
