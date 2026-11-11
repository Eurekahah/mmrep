"""Replay one flagged LiT image pair on its original batch and measure FP32.

Select a row in lit_frozen_check.csv by model, dataset, split and condition.
The model, checkpoint and source images must be available locally. The JSON
records the observed comparisons and any difference in the original inputs.
Exit 0 means the diagnosis ran. Exit 1 records a failure or incompatible
original inputs. The original frozen-image verdict is preserved in both cases.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
import csv
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import torch

from data.datasets import get_dataset
from extract.extract_features import MAIN_CONDITIONS, REPO_ROOT, STRENGTH_CONDITIONS, weight_identity
from extract.norm_stats import Job, _checked_metadata, _read_splits, file_sha256
from models.checkpoint_receipt import read_training_receipt
from models.loader import load_model
from tools.p2_image_identity import image_bytes_sha256


class OriginalInputsChanged(ValueError):
    """The archived measurements cannot be replayed under matching conditions."""


def _differences(actual, reference):
    delta = np.subtract(actual, reference, dtype=np.float64)
    if not np.isfinite(delta).all():
        raise ValueError("Replay produced a nonfinite difference")
    base_norm = np.linalg.norm(reference.astype(np.float64, copy=False))
    return {"max_abs": float(np.abs(delta).max()), "mean_abs": float(np.abs(delta).mean()),
            "relative_l2": float(np.linalg.norm(delta) / base_norm) if base_norm else None}


def _vision_state_hash(wrapper):
    digest = hashlib.sha256()
    for name, value in sorted(wrapper.model.state_dict().items()):
        if name.startswith(("vision_model.", "visual_projection.")):
            a = value.detach().cpu().contiguous().numpy()
            digest.update(f"{name}:{a.dtype}:{tuple(a.shape)}\n".encode())
            digest.update(a.tobytes())
    return digest.hexdigest()


@contextmanager
def _strict_fp32(device):
    if device.type != "cuda":
        yield
        return
    previous = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = previous


def _forward(wrapper, pixels, amp):
    autocast = torch.autocast("cuda", dtype=torch.float16) if amp else nullcontext()
    # Match the extractor's eval + no_grad and end autocast after this call.
    with torch.no_grad(), autocast:
        raw = wrapper.encode_image(pixels).detach().float().cpu().numpy()
    if not np.isfinite(raw).all() or raw.shape != (len(pixels), wrapper.embed_dim):
        raise ValueError("Invalid image embeddings in replay")
    return raw


def _load_review(review_csv, job):
    with Path(review_csv).open(encoding="utf-8-sig", newline="") as stream:
        rows = [r for r in csv.DictReader(stream)
                if (r["model"], r["dataset"], r["split"], r["cond"]) ==
                   (job.model, job.dataset, job.split, job.cond)]
    if len(rows) != 1 or rows[0]["status"] != "needs_review":
        raise ValueError("Select exactly one needs_review row from the frozen-image CSV")
    return rows[0]


def _diagnose(job, review_csv, features_root, splits_dir, data_root, ckpt_root,
              cache_dir, device, report):
    review = _load_review(review_csv, job)
    report["original_review"] = review
    sources, split_hashes = _read_splits(splits_dir)
    report["split_sha256"] = split_hashes[job.dataset]
    ids = sources[job.dataset, job.split][0]
    config_hashes = {name: file_sha256(REPO_ROOT / "configs" / name)
                     for name in ("models.yaml", "perturb.yaml")}
    metas = {}
    report["config_sha256"] = config_hashes
    report["original_weights"] = {}
    for setting in ("pretrained", "lit"):
        entry = job._replace(setting=setting)
        path = entry.path(features_root)
        if file_sha256(path) != review.get(f"{setting}_file_sha256"):
            raise OriginalInputsChanged(f"{setting} feature file differs from the reviewed file: {path}")
        try:
            metas[setting] = _checked_metadata(entry, features_root, sources, config_hashes)
        except (KeyError, ValueError) as exc:
            raise OriginalInputsChanged(str(exc)) from exc
        report["original_weights"][setting] = metas[setting]["weights"]
    previous, fine = metas["pretrained"], metas["lit"]
    for key in ("batch_size", "precision", "device", "device_name"):
        if previous.get(key) is None or previous[key] != fine.get(key):
            raise OriginalInputsChanged(f"Original {key} missing or different: "
                                        f"{previous.get(key)!r} / {fine.get(key)!r}")
    if previous["weights"]["resolved_revision"] != fine["weights"]["resolved_revision"]:
        raise OriginalInputsChanged("The files use different base revisions")
    original_device = torch.device(previous["device"])
    device = torch.device(device or original_device)
    if original_device.type != device.type:
        raise OriginalInputsChanged(f"Original device {original_device}, requested {device}")
    if device.type == "cuda":
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        if original_device.index is not None and original_device.index != device.index:
            raise OriginalInputsChanged(f"Original GPU index {original_device.index}, requested {device.index}")
    actual_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    if previous["device_name"] != actual_name:
        raise OriginalInputsChanged(f"Original device name {previous['device_name']!r}, current {actual_name!r}")
    size = previous["batch_size"]
    if type(size) is not int or size <= 0:
        raise OriginalInputsChanged(f"Invalid original batch size {size!r}")
    worst = review["max_abs_img_id"] or ids[0]
    index = ids.index(worst)
    start = index // size * size
    stop = min(start + size, len(ids))
    expected_ids = ids[start:stop]
    report["batch"] = {"start_inclusive": start, "stop_exclusive": stop, "original_batch_size": size,
                       "ids": expected_ids, "selected_id": worst, "selected_index": index,
                       "selection": "largest original difference" if review["max_abs_img_id"] else "first image for context review",
                       "source_precision": previous["precision"], "device": str(device), "device_name": actual_name}
    dataset = get_dataset(job.dataset, job.split, job.cond, job.model,
                          splits_dir=str(splits_dir), data_root=data_root)
    if any("image_bytes_sha256" in value for value in metas.values()):
        current_digest = image_bytes_sha256(dataset.img_dir, ids)
        report["image_bytes_sha256"] = current_digest
        if any(meta.get("image_bytes_sha256") != current_digest for meta in metas.values()):
            raise OriginalInputsChanged("Current image bytes differ from the recorded feature inputs")
    items = [dataset[i] for i in range(start, stop)]
    if [item["id"] for item in items] != expected_ids:
        raise OriginalInputsChanged("Loaded images do not match the original batch's ordered IDs")
    pixels = torch.stack([item["image"] for item in items]).to(device)
    report["preprocessed_batch_sha256"] = hashlib.sha256(pixels.cpu().contiguous().numpy().tobytes()).hexdigest()
    amp = previous["precision"] == "cuda_fp16_autocast"
    archived, outputs, vision_hash = {}, {}, {}
    for setting in ("pretrained", "lit"):
        meta = metas[setting]
        with np.load(job._replace(setting=setting).path(features_root), allow_pickle=False) as f:
            archived[setting] = f["img_raw"][start:stop]
        receipt = read_training_receipt(ckpt_root, job.model, setting)
        if receipt is not None and receipt["meta_sha256"] != meta["weights"].get("training_meta_sha256"):
            raise OriginalInputsChanged(f"{setting} training_meta_sha256 differs from original extraction")
        wrapper = load_model(job.model, setting, ckpt_root=ckpt_root, device=str(device),
                             cache_dir=cache_dir, local_files_only=True,
                             revision=meta["weights"]["resolved_revision"])
        try:
            identity = weight_identity(wrapper, ckpt_root, job.model, setting, None, receipt)
            for key in ("resolved_revision", "checkpoint_sha256", "training_meta_sha256"):
                if identity.get(key) != meta["weights"].get(key):
                    raise OriginalInputsChanged(f"{setting} {key} differs from original extraction")
            vision_hash[setting] = _vision_state_hash(wrapper)
            outputs[setting] = _forward(wrapper, pixels, amp)
            if setting == "pretrained":
                outputs["repeat"] = _forward(wrapper, pixels, amp)
            with _strict_fp32(device):
                outputs[setting + "_fp32"] = _forward(wrapper, pixels, False)
        finally:
            del wrapper
            if device.type == "cuda":
                torch.cuda.empty_cache()
    pairs = {
        "archived_lit_vs_pretrained": (archived["lit"], archived["pretrained"]),
        "original_precision_pretrained_repeat": (outputs["repeat"], outputs["pretrained"]),
        "original_precision_lit_vs_pretrained": (outputs["lit"], outputs["pretrained"]),
        "pretrained_replay_vs_archive": (outputs["pretrained"], archived["pretrained"]),
        "lit_replay_vs_archive": (outputs["lit"], archived["lit"]),
        "fp32_lit_vs_pretrained": (outputs["lit_fp32"], outputs["pretrained_fp32"]),
    }
    report["measurements"] = measured = {name: _differences(a, b) for name, (a, b) in pairs.items()}
    report["selected_sample"] = {name: _differences(a[index - start], b[index - start])
                                 for name, (a, b) in pairs.items()}
    report["vision_state_sha256"] = vision_hash
    report["relative_error_definition"] = "L2(actual-reference) / L2(reference); null if reference norm is zero"
    report["fp32_tf32_disabled"] = device.type == "cuda"
    if vision_hash["pretrained"] != vision_hash["lit"]:
        report["interpretation"] = "Visual state hashes differ. Inspect LiT parameters and base revision."
    elif measured["fp32_lit_vs_pretrained"]["max_abs"]:
        report["interpretation"] = "FP32 outputs differ with identical visual state. Inspect the operators and runtime."
    elif measured["original_precision_pretrained_repeat"]["max_abs"]:
        report["interpretation"] = "The repeated baseline varies under the original precision. FP32 results coincide."
    elif any(measured[key]["max_abs"] for key in ("pretrained_replay_vs_archive", "lit_replay_vs_archive")):
        report["interpretation"] = "The original saved features were not reproduced. Verify input bytes and runtime."
    else:
        report["interpretation"] = "The original difference is reproducible. The visual state and FP32 outputs coincide."


def recheck(job, *, review_csv, features_root, splits_dir, data_root, ckpt_root, cache_dir, device, out):
    report = {"model": job.model, "dataset": job.dataset, "split": job.split, "cond": job.cond,
              "torch": str(torch.__version__), "review_csv": str(review_csv)}
    try:
        _diagnose(job, review_csv, features_root, splits_dir, data_root, ckpt_root, cache_dir, device, report)
        report["status"] = "measured"
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        report["status"] = "original_inputs_changed" if isinstance(exc, OriginalInputsChanged) else "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=out.parent, mode="w", encoding="utf-8",
                                     delete=False, suffix=".tmp") as f:
        temp = Path(f.name)
        json.dump(report, f, ensure_ascii=False, indent=2)
        f.write("\n")
    try:
        os.replace(temp, out)
    finally:
        temp.unlink(missing_ok=True)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-csv", type=Path, required=True)
    parser.add_argument("--model", choices=("clip", "siglip"), required=True)
    parser.add_argument("--dataset", choices=("flickr", "coco"), required=True)
    parser.add_argument("--split", choices=("val", "test"), required=True)
    parser.add_argument("--cond", choices=MAIN_CONDITIONS + STRENGTH_CONDITIONS, required=True)
    parser.add_argument("--features-root", type=Path, default=REPO_ROOT / "features")
    parser.add_argument("--splits-dir", type=Path, default=REPO_ROOT / "splits")
    parser.add_argument("--data-root")
    parser.add_argument("--ckpt-root", type=Path, default=REPO_ROOT / "ckpt")
    parser.add_argument("--cache-dir")
    parser.add_argument("--device", help="Defaults to the device recorded in the feature files.")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    job = Job(args.model, "lit", args.dataset, args.split, args.cond)
    out = args.out or args.review_csv.parent / "lit_rechecks" / f"{job.model}_{job.dataset}_{job.split}_{job.cond}.json"
    report = recheck(job, review_csv=args.review_csv, features_root=args.features_root,
                     splits_dir=args.splits_dir, data_root=args.data_root, ckpt_root=args.ckpt_root,
                     cache_dir=args.cache_dir, device=args.device, out=out)
    print(f"{report['status']} {out}")
    return 0 if report["status"] == "measured" else 1


if __name__ == "__main__":
    raise SystemExit(main())
