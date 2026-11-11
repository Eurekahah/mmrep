"""Compute P2 raw row norms and compare the frozen LiT image branch.

Run from the repository root with python -m extract.norm_stats --stage STAGE.
Exit 2 means the stage was measured but some LiT pairs need review. For those
pairs, check the saved IDs, preprocessing, model revision, device, precision and
image batch size. Rerun the affected images in matching FP32 conditions and
compare the frozen parameters before attributing differences to fine-tuning.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import NamedTuple

import numpy as np

from data.config import load_models_cfg
from extract.extract_features import (
    MAIN_CONDITIONS, STRENGTH_CONDITIONS, REPO_ROOT, _digest, build_jobs,
    validate_feature_file,
)


STAGES = ("pretrained", "clip-finetuned", "siglip-finetuned", "all")
STAT_COLUMNS = ("model", "setting", "cond", "dataset", "split", "modality",
                "mean", "std", "p5", "p50", "p95")
FREEZE_COLUMNS = ("model", "dataset", "split", "cond", "n_images", "max_abs",
                  "mean_abs", "max_abs_img_id", "status", "review_reason",
                  "pretrained_file_sha256", "lit_file_sha256",
                  "pretrained_precision", "lit_precision", "pretrained_device",
                  "lit_device", "pretrained_device_name", "lit_device_name",
                  "pretrained_batch_size", "lit_batch_size")


class Job(NamedTuple):
    model: str
    setting: str
    dataset: str
    split: str
    cond: str

    def path(self, root):
        return Path(root) / self.model / self.setting / f"{self.dataset}_{self.split}_{self.cond}.npz"


def stage_jobs(stage):
    if stage not in STAGES:
        raise ValueError(f"Unknown stage {stage!r}")
    jobs = [Job(*job) for job in build_jobs(cond="complete")]
    if stage == "all":
        return jobs
    if stage == "pretrained":
        return [job for job in jobs if job.setting == "pretrained"]
    model = stage.split("-")[0]
    return [job for job in jobs if job.model == model and job.setting != "pretrained"]


def _read_splits(splits_dir):
    sources, hashes = {}, {}
    for dataset in ("flickr", "coco"):
        raw = (Path(splits_dir) / f"{dataset}_split.json").read_bytes()
        manifest = json.loads(raw)
        hashes[dataset] = hashlib.sha256(raw).hexdigest()
        if manifest.get("dataset") != dataset:
            raise ValueError(f"Wrong dataset in {dataset}_split.json")
        seen = set(manifest.get("train", []))
        for split in ("val", "test"):
            ids = manifest[split]
            if not ids or len(set(ids)) != len(ids) or seen.intersection(ids):
                raise ValueError(f"Empty, duplicate or overlapping IDs in {dataset}/{split}")
            seen.update(ids)
            captions = [manifest["captions"][id_] for id_ in ids]
            if any(len(group) != 5 or not all(isinstance(s, str) for s in group)
                   for group in captions):
                raise ValueError(f"Invalid captions in {dataset}/{split}")
            labels = ([manifest["labels"][id_] for id_ in ids] if dataset == "coco" else None)
            source_hash = _digest({"ids": ids, "captions": captions, "labels": labels})
            sources[dataset, split] = (ids, labels, source_hash)
    return sources, hashes


def _checked_metadata(job, root, sources, config_hashes):
    path = job.path(root)
    ids, labels, source_hash = sources[job.dataset, job.split]
    cfg = load_models_cfg()[job.model]
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data["_meta"].item()))
    expected = dict(job._asdict(), schema=1, limit=None, count=len(ids), embed_dim=cfg["embed_dim"],
                    input_size=cfg["input_size"], mean=cfg["mean"], std=cfg["std"],
                    model_config=cfg, source_sha256=source_hash, config_sha256=config_hashes)
    for key, value in expected.items():
        if key not in meta or meta[key] != value:
            raise ValueError(f"{path}: incorrect or missing metadata {key!r}")
    if meta.get("precision") not in ("cuda_fp16_autocast", "fp32"):
        raise ValueError(f"{path}: unknown precision {meta.get('precision')!r}")
    if (meta["precision"] == "cuda_fp16_autocast" and meta.get("device") is not None
            and not str(meta["device"]).startswith("cuda")):
        raise ValueError(f"{path}: CUDA precision is incompatible with the recorded device")
    weights = meta.get("weights")
    if not isinstance(weights, dict) or not weights.get("resolved_revision"):
        raise ValueError(f"{path}: missing base weight revision")
    if job.setting != "pretrained" and not weights.get("checkpoint_sha256"):
        raise ValueError(f"{path}: missing fine-tuned checkpoint identity")
    validate_feature_file(path, ids=ids, embed_dim=cfg["embed_dim"], cond=job.cond,
                          labels=labels, metadata=meta)
    return meta


def _row_norms(matrix):
    # Float64 accumulation avoids overflow and rounds only on output formatting.
    return np.sqrt(np.einsum("ij,ij->i", matrix, matrix, dtype=np.float64))


def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _summarize(job, modality, norms):
    p5, p50, p95 = np.percentile(norms, (5, 50, 95))
    return dict(job._asdict(), modality=modality, mean=norms.mean(),
                std=norms.std(ddof=0), p5=p5, p50=p50, p95=p95)


def _freeze_row(job, root, metas, ids, hashes=None):
    ref = job._replace(setting="pretrained")
    before_meta, after_meta = metas[ref], metas[job]
    # Both files were validated against this split in its exact original order.
    with np.load(ref.path(root), allow_pickle=False) as before, np.load(job.path(root), allow_pickle=False) as after:
        diff = np.abs(np.subtract(after["img_raw"], before["img_raw"], dtype=np.float64))
    worst_row, _ = np.unravel_index(diff.argmax(), diff.shape)
    maximum = float(diff.max())
    row = dict(model=job.model, dataset=job.dataset, split=job.split, cond=job.cond,
               n_images=len(ids), max_abs=maximum, mean_abs=float(diff.mean()),
               max_abs_img_id=ids[worst_row] if maximum else "",
               pretrained_file_sha256=hashes[ref] if hashes is not None else file_sha256(ref.path(root)),
               lit_file_sha256=hashes[job] if hashes is not None else file_sha256(job.path(root)))
    reasons = []
    for key in ("precision", "device", "device_name", "batch_size"):
        a, b = before_meta.get(key), after_meta.get(key)
        row[f"pretrained_{key}"], row[f"lit_{key}"] = a, b
        if a is None or b is None:
            reasons.append(f"missing {key}")
        elif a != b:
            reasons.append(f"different {key}")
    if before_meta["weights"]["resolved_revision"] != after_meta["weights"]["resolved_revision"]:
        reasons.append("different base revision")
    if before_meta.get("image_bytes_sha256") != after_meta.get("image_bytes_sha256"):
        reasons.append("different input image bytes")
    if maximum:
        reasons.append("nonzero image difference")
    row["status"] = "needs_review" if reasons else "identical"
    row["review_reason"] = ", ".join(reasons)
    return row


def _write_csv(path, columns, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _write_histograms(out_dir, groups):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    paths = []
    (out_dir / "figs").mkdir(exist_ok=True)
    for (model, dataset, split, modality), conditions in groups.items():
        order = [cond for cond in MAIN_CONDITIONS + STRENGTH_CONDITIONS if cond in conditions]
        cols = min(3, len(order))
        rows = (len(order) + cols - 1) // cols
        fig = Figure(figsize=(4.0 * cols, 2.9 * rows), constrained_layout=True)
        FigureCanvasAgg(fig)
        axes = fig.subplots(rows, cols, squeeze=False).ravel()
        # Use the same bin edges for each condition/setting in this figure.
        edges = np.histogram_bin_edges(np.concatenate([
            values for settings in conditions.values() for values in settings.values()
        ]), bins=35)
        for ax, cond in zip(axes, order):
            for setting, values in conditions[cond].items():
                ax.hist(values, bins=edges, histtype="step", linewidth=1.5, label=setting)
            ax.set(title=cond, xlabel="Raw embedding L2 norm", ylabel="Samples")
            ax.legend(fontsize="small")
        for ax in axes[len(order):]:
            ax.set_visible(False)
        fig.suptitle(f"{model} / {dataset} {split} / {modality}")
        relative = f"figs/norm_{model}_{dataset}_{split}_{modality}.png"
        fig.savefig(out_dir / relative, dpi=150)
        fig.clear()
        paths.append(relative)
    return paths


def generate(stage, features_root, splits_dir, results_dir):
    jobs = stage_jobs(stage)
    lit_jobs = [job for job in jobs if job.setting == "lit"]
    references = [job._replace(setting="pretrained") for job in lit_jobs]
    required = list(dict.fromkeys(jobs + references))
    missing = [job.path(features_root) for job in required if not job.path(features_root).is_file()]
    if missing:
        raise FileNotFoundError(f"Stage {stage}: missing {len(missing)} feature files, first: {missing[0]}")

    sources, hashes = _read_splits(splits_dir)
    config_hashes = {name: hashlib.sha256((REPO_ROOT / "configs" / name).read_bytes()).hexdigest()
                     for name in ("models.yaml", "perturb.yaml")}
    metas, identities, input_hashes = {}, {}, {}
    for job in required:
        meta = _checked_metadata(job, features_root, sources, config_hashes)
        metas[job] = meta
        input_hashes[job] = file_sha256(job.path(features_root))
        weights = meta["weights"]
        identity = (weights["resolved_revision"], weights.get("checkpoint_sha256"),
                    weights.get("training_meta_sha256"))
        group = (job.model, job.setting)
        if identities.setdefault(group, identity) != identity:
            raise ValueError(f"{job.path(features_root)}: mixed weight revisions/checkpoints within {group}")

    stats, histograms = [], defaultdict(lambda: defaultdict(dict))
    for job in jobs:
        with np.load(job.path(features_root), allow_pickle=False) as data:
            for modality, key in (("image", "img_raw"), ("text", "txt_raw")):
                if key in data.files:
                    norms = _row_norms(data[key])
                    stats.append(_summarize(job, modality, norms))
                    histograms[job.model, job.dataset, job.split, modality][job.cond][job.setting] = norms
    freezes = [_freeze_row(job, features_root, metas, sources[job.dataset, job.split][0], input_hashes)
               for job in lit_jobs]
    reviews = sum(row["status"] == "needs_review" for row in freezes)
    out = Path(results_dir) if stage == "all" else Path(results_dir) / "p2_stages" / stage
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out, prefix=".norm_stats_") as tmp:
        staging = Path(tmp)
        _write_csv(staging / "norm_stats.csv", STAT_COLUMNS, stats)
        _write_csv(staging / "lit_frozen_check.csv", FREEZE_COLUMNS, freezes)
        figures = _write_histograms(staging, histograms)
        artifacts = ["norm_stats.csv", "lit_frozen_check.csv", *figures]
        receipt = {
            "stage": stage, "status": "needs_review" if reviews else "passed",
            "expected_files": len(jobs), "validated_files": len(jobs),
            "extra_reference_files": len(required) - len(jobs),
            "expected_norm_rows": len(jobs) + sum(job.cond == "clean" for job in jobs),
            "norm_rows": len(stats), "freeze_pairs": len(freezes), "freeze_reviews": reviews,
            "split_sha256": hashes, "config_sha256": config_hashes,
            "feature_sha256": {job.path(".").as_posix(): sha for job, sha in input_hashes.items()},
            "artifact_sha256": {rel: file_sha256(staging / rel) for rel in artifacts},
            "artifacts": {"norm_stats": "norm_stats.csv", "freeze_csv": "lit_frozen_check.csv",
                          "histograms": figures},
        }
        manifest = "norm_stats_validation.json"
        (staging / manifest).write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n",
                                       encoding="utf-8")
        for relative in [*artifacts, manifest]:
            target = out / relative
            target.parent.mkdir(exist_ok=True)
            os.replace(staging / relative, target)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=STAGES, required=True)
    parser.add_argument("--features-root", type=Path, default=REPO_ROOT / "features")
    parser.add_argument("--splits-dir", type=Path, default=REPO_ROOT / "splits")
    parser.add_argument("--results-dir", type=Path, default=REPO_ROOT / "results")
    args = parser.parse_args(argv)
    try:
        receipt = generate(args.stage, args.features_root, args.splits_dir, args.results_dir)
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(1, f"Feature validation failed: {exc}\n")
    print(json.dumps(receipt, ensure_ascii=False))
    return 2 if receipt["freeze_reviews"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
