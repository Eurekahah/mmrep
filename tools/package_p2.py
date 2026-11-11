"""Make local P2 handoffs with per-file SHA-256 and verify the resulting ZIP."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import zipfile

import numpy as np

from data.config import REPO_ROOT, get_paths, load_models_cfg
from extract.extract_features import checkpoint_hashes
from extract.norm_stats import generate, stage_jobs
from models.checkpoint_receipt import read_training_receipt


ROOT = Path(REPO_ROOT)
BATCH_NAMES = {"pretrained": "P2_01_pretrained", "clip-finetuned": "P2_02_CLIP_finetuned",
               "siglip-finetuned": "P2_03_SigLIP2_finetuned", "all": "P2_全量交接"}
SOURCE_DIRS = ("analysis", "configs", "data", "docs", "eval", "extract", "metrics",
               "models", "splits", "tests", "tools", "train", "viz")


def sha256_stream(stream):
    digest = hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(block)
    return digest.hexdigest()


def verify_zip(path):
    with zipfile.ZipFile(path) as archived:
        manifest = json.loads(archived.read("MANIFEST.json"))
        inventory = manifest["files"]
        expected = set(inventory) | {"MANIFEST.json"}
        if set(archived.namelist()) != expected or len(archived.namelist()) != len(expected):
            raise ValueError(f"Invalid member list in {path}")
        for name, declared in inventory.items():
            with archived.open(name) as stream:
                if (archived.getinfo(name).file_size != declared["bytes"]
                        or sha256_stream(stream) != declared["sha256"]):
                    raise ValueError(f"{path}: checksum mismatch for {name}")
    return manifest


def write_zip(target, files, description, expected_hashes=None):
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=target.parent, suffix=".tmp", delete=False) as tmp:
        temporary = Path(tmp.name)
    try:
        inventory = {}
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
            for name, path in sorted(files.items()):
                with path.open("rb") as stream:
                    inventory[name] = {"bytes": path.stat().st_size, "sha256": sha256_stream(stream)}
                if expected_hashes and name in expected_hashes and inventory[name]["sha256"] != expected_hashes[name]:
                    raise ValueError(f"{name} changed after the feature/statistics validation")
                compressed = zipfile.ZIP_STORED if path.suffix in (".npz", ".png") else zipfile.ZIP_DEFLATED
                archive.write(path, name, compress_type=compressed)
            archive.writestr("MANIFEST.json", json.dumps(
                {**description, "files": inventory}, indent=2, ensure_ascii=False).encode("utf-8"))
        verify_zip(temporary)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    with target.open("rb") as stream:
        digest = sha256_stream(stream)
    target.with_suffix(".sha256").write_text(f"{digest}  {target.name}\n", encoding="utf-8")
    return {"path": str(target), "bytes": target.stat().st_size, "sha256": digest,
            "members": len(inventory), "verified": True}


def package_source(destination):
    names = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"], cwd=ROOT
    ).decode("utf-8").split("\0")
    top = {"README.md", "pytest.ini", "requirements.txt", "run_extract_all.sh", ".gitignore"}
    paths = [Path(n) for n in names if n and (n in top or Path(n).parts[0] in SOURCE_DIRS)]
    files = {p.as_posix(): ROOT / p for p in paths if (ROOT / p).is_file()}
    required = {"models/loader.py", "extract/extract_features.py", "extract/norm_stats.py",
                "run_extract_all.sh", "tools/package_p2.py"}
    if not required <= files.keys():
        raise FileNotFoundError(f"Missing P2 source: {sorted(required - files.keys())}")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    return write_zip(destination / "P2_源码及复现.zip", files, {
        "scope": "project source with P2 working-tree changes and fixed split indices",
        "base_or_current_commit": head,
        "setup": "See README.md, models/README.md and extract/README.md",
        "entrypoint": "python -m extract.extract_features --help",
    })


def formal_logs(log_dir, stage_paths):
    """Copy the cumulative formal P2 log. Keep errors/skips, account written jobs once."""
    records, written = [], {}
    canonical = {(ROOT / "features" / m / s / f"{d}_{p}_{c}.npz").resolve()
                 for m, s, d, p, c in stage_jobs("all")}
    log_path = log_dir / "runs.jsonl"
    for line in log_path.read_text(encoding="utf-8").splitlines():
        item = json.loads(line)
        if not item.get("output") or item.get("limit") is not None:
            continue
        path = (ROOT / item["output"]).resolve()
        if path not in canonical:
            continue
        records.append(item)
        if item["status"] == "ok" and item.get("cache_status") == "written":
            written[path] = item
    expected = {p.resolve() for p in stage_paths}
    missing = expected - written.keys()
    if missing:
        raise ValueError(f"No successful extraction timing record for {next(iter(missing))}")
    rows = []
    for path in sorted(expected):
        item = written[path]
        if (not str(item.get("device", "")).startswith("cuda")
                or item.get("precision") != "cuda_fp16_autocast"
                or not item.get("batch_size") or item["gpu_mem_peak_MB"] <= 0):
            raise ValueError(f"Incomplete formal GPU log: {item['output']}")
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(data["_meta"].item())
        for key in ("model", "setting", "dataset", "split", "cond", "device", "device_name",
                    "precision", "batch_size", "text_batch_size", "image_bytes_sha256"):
            if key not in meta or key not in item or meta[key] != item[key]:
                raise ValueError(f"{path.name}: extraction log differs from NPZ for {key}")
        if (item["n_images"] != meta["count"]
                or item["n_texts"] != (5 * meta["count"] if meta["cond"] == "clean" else 0)):
            raise ValueError(f"{path.name}: extraction log counts differ from NPZ")
        rows.append(item)
    log_copy = log_dir / "P2_runs.jsonl"
    with tempfile.NamedTemporaryFile("w", dir=log_dir, delete=False, encoding="utf-8") as temp:
        for item in records:
            temp.write(json.dumps(item, ensure_ascii=False) + "\n")
    os.replace(temp.name, log_copy)
    return log_copy, {"written_outputs": len(rows),
                      "sum_written_wall_sec": round(sum(r["wall_sec"] for r in rows), 3),
                      "max_gpu_mem_peak_MB": max(r["gpu_mem_peak_MB"] for r in rows),
                      "cumulative_log_rows": len(records)}


def review_frozen_images(stats, features_root):
    """Replay flagged pairs on the original hardware and retain the measured differences."""
    from extract.norm_stats import Job
    from extract.recheck_lit import recheck

    data_root, splits_dir = get_paths()
    table = stats / "lit_frozen_check.csv"
    with table.open(encoding="utf-8", newline="") as stream:
        pending = [r for r in csv.DictReader(stream) if r["status"] == "needs_review"]
    outputs, unresolved = [], []
    for row in pending:
        job = Job(row["model"], "lit", row["dataset"], row["split"], row["cond"])
        target = stats / "lit_rechecks" / (job.model + "_" + job.path(".").stem + ".json")
        report = recheck(job, review_csv=table, features_root=features_root, splits_dir=splits_dir,
                         data_root=data_root, ckpt_root=ROOT / "ckpt", cache_dir=ROOT.parent / ".p2-runtime/hf/hub",
                         device=row["pretrained_device"] or "cuda", out=target)
        outputs.append(target)
        if report["status"] != "measured":
            unresolved.append(target)
            continue
        measures = report["measurements"]
        repeat = measures["original_precision_pretrained_repeat"]["max_abs"]
        fp32 = measures["fp32_lit_vs_pretrained"]["max_abs"]
        state = report["vision_state_sha256"]
        if (state["lit"] != state["pretrained"] or fp32 != 0 or float(row["max_abs"]) > repeat
                or any(measures[k]["max_abs"] > repeat for k in ("lit_replay_vs_archive",
                                                                  "pretrained_replay_vs_archive"))):
            unresolved.append(target)
    if unresolved:
        raise ValueError("Frozen-image discrepancy needs investigation. Measured evidence: "
                         + ", ".join(str(p) for p in unresolved))
    return outputs


def package_stage(stage, destination):
    log_dir = ROOT / "logs"
    data_receipt = log_dir / "p2_data_validation.json"
    input_check = json.loads(data_receipt.read_text(encoding="utf-8"))
    if input_check["status"] != "passed":
        raise ValueError(f"Input dataset validation is pending: {data_receipt}")
    receipt = generate(stage, ROOT / "features", ROOT / "splits", ROOT / "results")
    stats = ROOT / "results" if stage == "all" else ROOT / "results" / "p2_stages" / stage
    replay_files = (review_frozen_images(stats, ROOT / "features")
                    if receipt["status"] == "needs_review" else [])
    for name, current_hash in receipt["split_sha256"].items():
        verified = input_check["datasets"][name]
        if (verified["status"] != "passed" or not verified["split_bytes_identical"]
                or verified["regenerated_split_sha256"] != current_hash
                or verified["checked_in_split_sha256"] != current_hash
                or any(p["images"] != p["decoded"] for p in verified["splits"].values())):
            raise ValueError(f"Raw data verification and current split disagree for {name}")
    jobs = stage_jobs(stage)
    feature_paths = [job.path(ROOT / "features") for job in jobs]
    copied_log, timings = formal_logs(log_dir, feature_paths)
    artifacts = receipt["artifacts"]
    files = [*feature_paths, copied_log, data_receipt, *replay_files, stats / "norm_stats_validation.json",
             stats / artifacts["norm_stats"], stats / artifacts["freeze_csv"],
             *(stats / name for name in artifacts["histograms"])]
    trained = {}
    for model_name, setting in sorted({(j.model, j.setting) for j in jobs if j.setting != "pretrained"}):
        training = read_training_receipt(ROOT / "ckpt", model_name, setting)
        trained[model_name, setting] = (training["meta_sha256"],
                                        checkpoint_hashes(ROOT / "ckpt", model_name, setting))
    val_ids = json.loads((ROOT / "splits" / "coco_split.json").read_text(encoding="utf-8"))["val"]
    for model in sorted({job.model for job in jobs}):
        smoke = log_dir / "p2_real_smoke" / "coco_val" / model / "report.json"
        result = json.loads(smoke.read_text(encoding="utf-8"))
        cfg = load_models_cfg()[model]
        count = result["batch_size"]
        expected = {"status": "passed", "model": model, "setting": "pretrained", "dataset": "coco",
                    "split": "val", "precision": "fp16 autocast with fp32 parameters",
                    "hf_name": cfg["hf_name"], "input_size": cfg["input_size"], "embed_dim": cfg["embed_dim"]}
        if (not str(result["device"]).startswith("cuda") or not 1 <= count <= len(val_ids)
                or any(result.get(k) != v for k, v in expected.items())
                or result["img_ids"] != val_ids[:count]
                or any(result[m]["shape"] != [count, cfg["embed_dim"]] for m in ("image", "text"))):
            raise ValueError(f"Real GPU single-batch receipt differs from the current model or COCO split: {smoke}")
        for job in jobs:
            if job.model == model:
                with np.load(job.path(ROOT / "features"), allow_pickle=False) as data:
                    meta = json.loads(data["_meta"].item())
                if meta["weights"]["resolved_revision"] != result["resolved_revision"]:
                    raise ValueError(f"{job}: base model revision differs from the verified model")
                source_digest = input_check["datasets"][job.dataset]["splits"][job.split]["image_bytes_sha256"]
                if meta.get("image_bytes_sha256") != source_digest:
                    raise ValueError(f"{job}: actual input image bytes differ from the validated raw images")
                if job.setting != "pretrained":
                    current = (meta["weights"].get("training_meta_sha256"), meta["weights"].get("checkpoint_sha256"))
                    if current != trained[job.model, job.setting]:
                        raise ValueError(f"{job}: training receipt or checkpoint bytes changed")
        files.append(smoke)
    if stage == "pretrained":
        selections = [("all", "pretrained")]
    elif stage == "all":
        selections = [("all", "all")]
    else:
        selections = [(jobs[0].model, s) for s in ("lit", "lora_r8", "lora_r16")]
    commands = [
        f"python -m extract.extract_features --model {m} --setting {s} --cond complete "
        "--cache-dir ../.p2-runtime/hf/hub --local-files-only" for m, s in selections
    ]
    expected = {"features/" + k: v for k, v in receipt["feature_sha256"].items()}
    expected.update({(stats / k).relative_to(ROOT).as_posix(): v for k, v in receipt["artifact_sha256"].items()})
    return write_zip(destination / (BATCH_NAMES[stage] + ".zip"),
                     {p.relative_to(ROOT).as_posix(): p for p in files}, {
        "stage": stage, "feature_files": len(jobs), "norm_rows": receipt["norm_rows"],
        "freeze_pairs": receipt["freeze_pairs"], "replayed_freeze_pairs": len(replay_files),
        "compute": timings,
        "reproduce": [*commands, f"python -m tools.package_p2 --stage {stage}"],
        "use": "See the accompanying source ZIP and extract/README.md for Windows commands and setup.",
    }, expected_hashes=expected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=(*BATCH_NAMES, "source"))
    parser.add_argument("--dest", type=Path, default=ROOT.parent / "P2_本地交接")
    parser.add_argument("--verify", type=Path, help="Read every member and compare against MANIFEST.json")
    args = parser.parse_args()
    if args.verify:
        manifest = verify_zip(args.verify)
        print(f"Verified {len(manifest['files'])} payload files in {args.verify}")
        return
    if not args.stage:
        parser.error("Choose --stage or --verify")
    result = (package_source(args.dest) if args.stage == "source"
              else package_stage(args.stage, args.dest))
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
