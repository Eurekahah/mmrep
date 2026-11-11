"""P2 feature extraction. Run with ``python -m extract.extract_features``.

The NPZ arrays follow CONTRACT 3.7. A small ``_meta`` JSON string identifies
the inputs, precision and weights so an incompatible cache is never reused.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import gc
import hashlib
import itertools
import json
import os
from pathlib import Path
import tempfile
import zipfile

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from data.datasets import collate_fn, get_dataset
from tools.logger import RunLogger
from tools.p2_image_identity import image_bytes_sha256


REPO_ROOT = Path(__file__).resolve().parents[1]
MAIN_CONDITIONS = ("clean", "noise", "occl")
STRENGTH_CONDITIONS = ("noise_lo", "noise_hi", "occl_lo", "occl_hi")
MODELS = ("clip", "siglip")
SETTINGS = ("pretrained", "lit", "lora_r8", "lora_r16")
DEFAULT_BATCH = {"clip": 256, "siglip": 192}
RUNTIME_KEYS = {"device", "device_name", "batch_size", "text_batch_size",
                "attempted_batch_size", "attempted_text_batch_size", "oom_retries"}


class FeatureCacheError(ValueError):
    """An existing feature file cannot safely be used as this run's output."""


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _digest(value):
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def _is_within(path, root):
    return path.resolve().is_relative_to(root.resolve())


def _input_description(dataset, limit):
    if dataset.name not in ("flickr", "coco") or dataset.split not in ("val", "test"):
        raise ValueError("P2 extracts Flickr/COCO val/test only; CIFAR belongs to P6")
    if dataset.cond not in MAIN_CONDITIONS + STRENGTH_CONDITIONS:
        raise ValueError(f"Unknown condition: {dataset.cond}")
    if dataset.cond in STRENGTH_CONDITIONS and (dataset.name, dataset.split) != ("flickr", "val"):
        raise ValueError("Weak/strong conditions are only defined for Flickr val")
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")
    if len(dataset.ids) != len(dataset):
        raise ValueError("dataset.ids and dataset length differ")
    ids = list(dataset.ids[:limit])
    if not ids or not all(isinstance(x, str) for x in ids) or len(set(ids)) != len(ids):
        raise ValueError("dataset.ids must be nonempty, unique strings in split order")
    captions = [dataset.meta["captions"][id_] for id_ in ids]
    if any(len(caps) != 5 or not all(isinstance(x, str) for x in caps) for caps in captions):
        raise ValueError("Every image must have exactly five string captions")
    labels = None
    if dataset.name == "coco":
        labels = np.asarray([dataset.meta["labels"][id_] for id_ in ids], dtype=np.int64)
    return ids, captions, labels


def validate_feature_file(path, *, ids, embed_dim, cond, labels=None, metadata):
    """Validate all arrays and input identity before allowing a resume skip."""
    expected_keys = {"img_raw", "img_ids", "_meta"}
    if cond == "clean":
        expected_keys.update(("txt_raw", "txt2img"))
    if labels is not None:
        expected_keys.add("labels")
    try:
        with np.load(path, allow_pickle=False) as data:
            if set(data.files) != expected_keys:
                raise ValueError(f"fields {sorted(data.files)} != {sorted(expected_keys)}")
            if data["_meta"].shape != ():
                raise ValueError("expected scalar metadata")
            original = json.loads(str(data["_meta"].item()))
            if isinstance(original, dict) and "image_bytes_sha256" in metadata:
                if "image_bytes_sha256" not in original:
                    raise ValueError("Legacy file lacks image_bytes_sha256. Archive it and re-extract from "
                                     "the verified original images; a current hash cannot certify old embeddings")
                if original["image_bytes_sha256"] != metadata["image_bytes_sha256"]:
                    raise ValueError("Original image bytes have changed (image_bytes_sha256 mismatch)")
            if not isinstance(original, dict) or (
                {k: v for k, v in original.items() if k not in RUNTIME_KEYS} !=
                {k: v for k, v in metadata.items() if k not in RUNTIME_KEYS}
            ):
                raise ValueError("input/model/precision metadata changed or missing")
            if data["img_ids"].dtype.kind != "U" or not np.array_equal(data["img_ids"], np.asarray(ids)):
                raise ValueError("img_ids do not match this dataset/split in exact order")
            for key, rows in (("img_raw", len(ids)), ("txt_raw", len(ids) * 5)):
                if key == "txt_raw" and cond != "clean":
                    continue
                array = data[key]
                if array.dtype != np.float32 or array.shape != (rows, embed_dim):
                    raise ValueError(f"{key} must be float32 [{rows}, {embed_dim}]")
                if not np.isfinite(array).all():
                    raise ValueError(f"{key} contains NaN or infinity")
            if cond == "clean":
                mapping = data["txt2img"]
                if mapping.dtype != np.int64 or not np.array_equal(mapping, np.repeat(np.arange(len(ids)), 5)):
                    raise ValueError("txt2img must map every consecutive five captions to their image")
            if labels is not None:
                if data["labels"].dtype != np.int64 or not np.array_equal(data["labels"], labels):
                    raise ValueError("COCO labels differ from the split metadata")
    except (ValueError, OSError, EOFError, KeyError, TypeError, zipfile.BadZipFile) as exc:
        raise FeatureCacheError(f"Refusing cache {path}: {exc}. Move the invalid file aside before retrying.") from exc
    return original


def _raw_array(tensor, rows, dimension, kind):
    if not isinstance(tensor, torch.Tensor) or tensor.shape != (rows, dimension):
        raise ValueError(f"{kind} encoder must return [{rows}, {dimension}] raw tensor")
    array = tensor.detach().float().cpu().numpy()
    if not np.isfinite(array).all():
        raise ValueError(f"{kind} encoder produced NaN or infinity")
    return array


def _retry_oom(operation, batch_size, text_batch_size, *, device, clean=True):
    """A file is retried from its first sample so batch shapes stay consistent."""
    retries = 0
    while True:
        try:
            return operation(batch_size, text_batch_size, retries)
        except RuntimeError as exc:
            is_oom = isinstance(exc, torch.cuda.OutOfMemoryError) or any(
                msg in str(exc).lower() for msg in ("cuda out of memory", "cuda error: out of memory"))
            if device.type != "cuda" or not is_oom or (batch_size == 1 and (not clean or text_batch_size == 1)):
                raise
        # Do this outside the exception handler, after the frame holding tensors is released.
        gc.collect()
        torch.cuda.empty_cache()
        batch_size = max(1, batch_size // 2)
        text_batch_size = max(1, text_batch_size // 2)
        retries += 1
        print(f"CUDA OOM, retry {retries} from the first image with batches {batch_size}/{text_batch_size}",
              flush=True)


def extract_dataset(wrapper, dataset, out_path, *, batch_size=None, text_batch_size=None,
                    num_workers=0, amp=True, limit=None, log_dir=None,
                    model_name=None, setting=None, cache_identity=None):
    """Extract one condition without shuffling or normalizing the raw embeddings.

    ``dataset`` supplies P1's ``name/split/cond/model/ids/meta`` attributes and
    item dictionaries. Limited runs require a ``.limitN.npz`` filename outside
    the canonical repository ``features/`` tree.
    When called directly with custom/modified weights, the caller must provide
    a distinct ``cache_identity`` (e.g. checkpoint SHA256). CLI calls do this.
    """
    device = torch.device(wrapper.device)
    if batch_size is None:
        batch_size = DEFAULT_BATCH[dataset.model] if device.type == "cuda" else 16
    if batch_size <= 0 or num_workers < 0:
        raise ValueError("batch_size must be positive and num_workers nonnegative")
    text_batch_size = batch_size if text_batch_size is None else text_batch_size
    if text_batch_size <= 0:
        raise ValueError("text_batch_size must be positive")
    out_path = Path(out_path)
    if out_path.suffix != ".npz":
        raise ValueError("Output must be an .npz file")
    if limit is not None:
        if not out_path.name.endswith(f".limit{limit}.npz") or _is_within(out_path, REPO_ROOT / "features"):
            raise ValueError("Limited output requires a .limitN.npz name outside the formal features/ directory")
    ids, captions, labels = _input_description(dataset, limit)
    actual_model = getattr(wrapper, "model_name", dataset.model)
    actual_setting = getattr(wrapper, "setting", "pretrained")
    model_name = model_name or actual_model
    setting = setting or actual_setting
    if model_name != actual_model or dataset.model != actual_model or setting != actual_setting:
        raise ValueError("Requested model/setting and dataset do not match the loaded wrapper")
    if cache_identity is None:
        native_config = getattr(getattr(wrapper, "model", None), "config", None)
        cache_identity = {"model": actual_model, "setting": actual_setting,
                          "resolved_revision": getattr(native_config, "_commit_hash", None)}
    use_amp = bool(amp and device.type == "cuda")
    metadata = {
        "schema": 1, "model": model_name, "setting": setting,
        "dataset": dataset.name, "split": dataset.split, "cond": dataset.cond,
        "limit": limit, "count": len(ids), "embed_dim": int(wrapper.embed_dim),
        "input_size": int(wrapper.input_size),
        "mean": list(wrapper.mean), "std": list(wrapper.std),
        "model_config": getattr(wrapper, "model_config", None),
        "source_sha256": _digest({"ids": ids, "captions": captions,
                                  "labels": labels.tolist() if labels is not None else None}),
        "precision": "cuda_fp16_autocast" if use_amp else "fp32",
        "config_sha256": {name: hashlib.sha256((REPO_ROOT / "configs" / name).read_bytes()).hexdigest()
                          for name in ("models.yaml", "perturb.yaml")},
        "weights": cache_identity,
    }
    if getattr(dataset, "img_dir", None) is not None:
        # P1 supplies the actual image tree. Read bytes anew on every independent
        # extraction/resume so another tree with the same IDs cannot reuse this file.
        metadata["image_bytes_sha256"] = image_bytes_sha256(dataset.img_dir, ids)
    validation = dict(ids=ids, embed_dim=wrapper.embed_dim, cond=dataset.cond,
                      labels=labels, metadata=metadata)
    def attempt(image_bs, text_bs, retries):
        # Record the largest microbatch actually available in this file.
        run = {"device": str(device),
               "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
               "batch_size": min(image_bs, len(ids)),
               "text_batch_size": min(text_bs, 5 * min(len(ids), image_bs)) if dataset.cond == "clean" else None,
               "attempted_batch_size": batch_size, "attempted_text_batch_size": text_batch_size,
               "oom_retries": retries}
        return _extract_attempt(wrapper, dataset, out_path, metadata, validation, captions,
                                batch_size=image_bs, text_batch_size=text_bs, run=run,
                                num_workers=num_workers, use_amp=use_amp, log_dir=log_dir)
    # RunLogger's default CUDA device must match the model when an index was supplied.
    with torch.cuda.device(device) if device.type == "cuda" else nullcontext():
        return _retry_oom(attempt, min(batch_size, len(ids)), min(text_batch_size, 5 * len(ids)),
                          device=device, clean=dataset.cond == "clean")


def _extract_attempt(wrapper, dataset, out_path, metadata, validation, captions, *,
                     batch_size, text_batch_size, run, num_workers, use_amp, log_dir):
    ids, labels = validation["ids"], validation["labels"]
    model_name, setting, limit = metadata["model"], metadata["setting"], metadata["limit"]
    device = torch.device(wrapper.device)
    texts_count = 5 * len(ids) if dataset.cond == "clean" else 0
    with RunLogger(f"extract_{dataset.name}_{dataset.split}_{dataset.cond}",
                   model=model_name, setting=setting, seed=42, log_dir=log_dir,
                   track_cuda=device.type == "cuda",
                   output=str(out_path), limit=limit, precision=metadata["precision"],
                   image_bytes_sha256=metadata.get("image_bytes_sha256"),
                   dataset=dataset.name, split=dataset.split, cond=dataset.cond,
                   cache_status="error", **run) as log:
        if out_path.exists():
            original = validate_feature_file(out_path, **validation)
            recorded = {key: original.get(key) for key in RUNTIME_KEYS}
            complete = (all(isinstance(original.get(k), str) and original[k] for k in ("device", "device_name"))
                        and type(original.get("batch_size")) is int and original["batch_size"] > 0
                        and "text_batch_size" in original and (
                            type(original["text_batch_size"]) is int and original["text_batch_size"] > 0
                            if dataset.cond == "clean" else original["text_batch_size"] is None))
            # Original producer data stay in the cache and its written log row.
            # This additional row times only file validation.
            log.metrics.update(recorded, runtime_metadata_complete=complete,
                               n_images=len(ids), n_texts=texts_count, work="cache_validation")
            log.metric("cache_status", "validated_skip")
            return {"path": str(out_path), "status": "skipped", "n_images": len(ids),
                    "n_texts": texts_count, "runtime_metadata_complete": complete, **recorded}

        source = dataset if limit is None else Subset(dataset, range(len(ids)))
        loader = DataLoader(source, batch_size=batch_size, shuffle=False, drop_last=False,
                            num_workers=num_workers, collate_fn=collate_fn,
                            pin_memory=device.type == "cuda")
        img_chunks, txt_chunks = [], []
        position = 0
        wrapper.eval()
        with torch.no_grad():
            for batch in loader:
                count = len(batch["id"])
                expected_ids = ids[position:position + count]
                if batch["id"] != expected_ids:
                    raise ValueError("DataLoader image order differs from dataset.ids")
                if batch["captions"] != captions[position:position + count]:
                    raise ValueError("Dataset captions differ from split metadata")
                if not torch.equal(batch["idx"], torch.arange(position, position + count)):
                    raise ValueError("Dataset idx must be the original split position")
                if labels is not None and not np.array_equal(batch["label"].numpy(), labels[position:position + count]):
                    raise ValueError("Dataset labels differ from split metadata")
                pixels = batch["image"].to(device, non_blocking=device.type == "cuda")
                if tuple(pixels.shape[1:]) != (3, wrapper.input_size, wrapper.input_size):
                    raise ValueError("Dataset image shape differs from the model configuration")
                context = torch.autocast("cuda", dtype=torch.float16) if use_amp else nullcontext()
                with context:
                    encoded = wrapper.encode_image(pixels)
                img_chunks.append(_raw_array(encoded, count, wrapper.embed_dim, "image"))
                if dataset.cond == "clean":
                    texts = [caption for group in batch["captions"] for caption in group]
                    for start in range(0, len(texts), text_batch_size):
                        chunk = texts[start:start + text_batch_size]
                        tokens = {key: value.to(device) for key, value in wrapper.tokenize(chunk).items()}
                        context = torch.autocast("cuda", dtype=torch.float16) if use_amp else nullcontext()
                        with context:
                            encoded = wrapper.encode_text(**tokens)
                        txt_chunks.append(_raw_array(encoded, len(chunk), wrapper.embed_dim, "text"))
                position += count
        if position != len(ids):
            raise ValueError("Extraction did not produce every expected image")
        arrays = {"img_raw": np.concatenate(img_chunks), "img_ids": np.asarray(ids),
                  "_meta": np.asarray(_json(dict(metadata, **run)))}
        if dataset.cond == "clean":
            arrays.update(txt_raw=np.concatenate(txt_chunks),
                          txt2img=np.repeat(np.arange(len(ids), dtype=np.int64), 5))
        if labels is not None:
            arrays["labels"] = labels
        out_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="wb", dir=out_path.parent,
                                             prefix=f".{out_path.name}.", suffix=".tmp", delete=False) as stream:
                temporary = Path(stream.name)
                np.savez_compressed(stream, **arrays)
                stream.flush()
                os.fsync(stream.fileno())
            validate_feature_file(temporary, **validation)
            if out_path.exists():
                raise FileExistsError(f"Output appeared during extraction; refusing to overwrite {out_path}")
            os.replace(temporary, out_path)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()
        log.metric("cache_status", "written")
        log.metric("runtime_metadata_complete", True)
        log.metric("n_images", len(ids))
        log.metric("n_texts", 5 * len(ids) if dataset.cond == "clean" else 0)
        return {"path": str(out_path), "status": "written", "n_images": len(ids),
                "n_texts": texts_count, "runtime_metadata_complete": True, **run}


def build_jobs(model="all", setting="all", dataset="all", split="all", cond="all"):
    """Expand CLI selections; ``complete`` includes all 128 feature files."""
    choices = ((MODELS if model == "all" else (model,)),
               (SETTINGS if setting == "all" else (setting,)),
               (("flickr", "coco") if dataset == "all" else (dataset,)),
               (("val", "test") if split == "all" else (split,)))
    jobs = []
    for m, s, d, p in itertools.product(*choices):
        if cond == "strength":
            conditions = STRENGTH_CONDITIONS if (d, p) == ("flickr", "val") else ()
        elif cond == "complete":
            conditions = MAIN_CONDITIONS + (STRENGTH_CONDITIONS if (d, p) == ("flickr", "val") else ())
        else:
            conditions = MAIN_CONDITIONS if cond == "all" else (cond,)
        jobs.extend((m, s, d, p, condition) for condition in conditions)
    if not jobs:
        raise ValueError("No applicable jobs: strength requires Flickr val")
    return jobs


def checkpoint_hashes(checkpoint_root, model, setting):
    """Hash checkpoint files using the relative keys stored in feature metadata."""
    if setting == "pretrained":
        return {}
    if setting not in SETTINGS:
        raise ValueError(f"Unknown checkpoint setting: {setting}")
    checkpoint_root = Path(checkpoint_root)
    if not checkpoint_root.is_absolute():
        checkpoint_root = REPO_ROOT / checkpoint_root
    location = checkpoint_root / model / setting
    candidates = [location / "trainable_state.pt"] if setting == "lit" else sorted((location / "adapter").rglob("*"))
    hashes = {}
    for path in candidates:
        if path.is_file():
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            hashes[str(path.relative_to(location))] = digest.hexdigest()
    if not hashes:
        raise FileNotFoundError(f"No checkpoint files in {location}")
    return hashes


def weight_identity(wrapper, ckpt_root, model, setting, revision, receipt):
    """Build the same feature weight identity for formal extraction and LiT replay."""
    native = getattr(wrapper, "model", None)
    config = getattr(native, "config", None)
    identity = {"requested_revision": revision, "resolved_revision": getattr(config, "_commit_hash", None)}
    if setting != "pretrained":
        identity["checkpoint_sha256"] = checkpoint_hashes(ckpt_root, model, setting)
        identity["training_meta_sha256"] = receipt["meta_sha256"]
    return identity


def choose_batches(args, model, reference, cap, precision):
    """Use the producer's corresponding pretrained batch sizes when available."""
    previous = {}
    if reference is not None and reference.is_file():
        with np.load(reference, allow_pickle=False) as data:
            if "_meta" in data.files:
                prev = json.loads(str(data["_meta"].item()))
                if (prev.get("model"), prev.get("setting"), prev.get("precision")) == (
                        model, "pretrained", precision):
                    previous = prev
    def valid(value):
        return type(value) is int and value > 0
    fallback = DEFAULT_BATCH[model] if torch.device(args.device).type == "cuda" else 16
    image = args.batch_size or (previous.get("batch_size") if valid(previous.get("batch_size")) else fallback)
    text = args.text_batch_size or (None if args.batch_size else previous.get("text_batch_size"))
    text = text if valid(text) else image
    if cap:
        image, text = min(image, cap[0]), min(text, cap[1])
    return image, text


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=(*MODELS, "all"), default="all")
    parser.add_argument("--setting", choices=(*SETTINGS, "all"), default="pretrained")
    parser.add_argument("--dataset", choices=("flickr", "coco", "all"), default="all")
    parser.add_argument("--split", choices=("val", "test", "all"), default="all")
    parser.add_argument("--cond", choices=(*MAIN_CONDITIONS, "all", "strength", "complete"), default="all")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int,
                        help="Image batch size (CUDA defaults: CLIP 256, SigLIP2 192; CPU 16)")
    parser.add_argument("--text-batch-size", type=int)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--out", type=Path, help="Default features/, or features_smoke/ when --limit is used")
    parser.add_argument("--limit", type=int, help="Smoke run only; isolated path and .limitN.npz suffix")
    parser.add_argument("--data-root")
    parser.add_argument("--splits-dir")
    parser.add_argument("--ckpt-root", type=Path, default=REPO_ROOT / "ckpt")
    parser.add_argument("--cache-dir")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--revision")
    parser.add_argument("--log-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true", help="List planned outputs without model/data loading")
    args = parser.parse_args(argv)
    if any(x is not None and x <= 0 for x in (args.batch_size, args.text_batch_size)):
        parser.error("Batch sizes must be positive")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    try:
        jobs = build_jobs(args.model, args.setting, args.dataset, args.split, args.cond)
    except ValueError as exc:
        parser.error(str(exc))
    root = args.out or REPO_ROOT / ("features_smoke" if args.limit is not None else "features")
    if args.limit is not None and _is_within(root, REPO_ROOT / "features"):
        parser.error("--limit output must be outside the formal features/ directory")
    if args.limit is not None:
        root = root / f"limit_{args.limit}"
    suffix = f".limit{args.limit}.npz" if args.limit is not None else ".npz"
    if args.dry_run:
        for m, s, d, p, c in jobs:
            print(root / m / s / f"{d}_{p}_{c}{suffix}")
        print(f"{len(jobs)} feature files")
        return 0
    from models.loader import load_model
    from models.checkpoint_receipt import read_training_receipt

    caps = {}
    precision = "cuda_fp16_autocast" if torch.device(args.device).type == "cuda" and not args.no_amp else "fp32"
    for (model, setting), group in itertools.groupby(jobs, key=lambda job: job[:2]):
        receipt = read_training_receipt(args.ckpt_root, model, setting)
        wrapper = load_model(model, setting, ckpt_root=str(args.ckpt_root), device=args.device,
                             cache_dir=args.cache_dir, local_files_only=args.local_files_only,
                             revision=args.revision)
        identity = weight_identity(wrapper, args.ckpt_root, model, setting, args.revision, receipt)
        try:
            for _, _, name, split, condition in group:
                dataset = get_dataset(name, split, condition, model, data_root=args.data_root,
                                      splits_dir=args.splits_dir)
                filename = f"{name}_{split}_{condition}{suffix}"
                ref = root / model / "pretrained" / filename if setting == "lit" else None
                image_bs, text_bs = choose_batches(args, model, ref, caps.get(model), precision)
                result = extract_dataset(wrapper, dataset, root / model / setting / filename,
                                         batch_size=image_bs, text_batch_size=text_bs,
                                         num_workers=args.num_workers, amp=not args.no_amp, limit=args.limit,
                                         model_name=model, setting=setting, log_dir=args.log_dir,
                                         cache_identity=identity)
                if result.get("oom_retries"):
                    caps[model] = (min(image_bs, result["batch_size"]),
                                   min(text_bs, result["text_batch_size"] or
                                       max(1, text_bs // 2 ** result["oom_retries"])))
        finally:
            del wrapper
            if torch.device(args.device).type == "cuda":
                torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
