"""Shared P5 input validation, file discovery and output conventions."""
from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS = {"clip": 512, "siglip": 768}
SETTINGS = ("pretrained", "lit", "lora_r8", "lora_r16")
MAIN_CONDS = ("clean", "noise", "occl")
STRENGTH_CONDS = ("noise_lo", "noise_hi", "occl_lo", "occl_hi")
CONDITIONS = MAIN_CONDS + STRENGTH_CONDS
METRICS = tuple(f"{direction}_R{k}" for direction in ("i2t", "t2i") for k in (1, 5, 10))
RETRIEVAL_COLUMNS = ["dataset", "model", "setting", "cond", *METRICS]
DROP_COLUMNS = ["dataset", "model", "setting", "cond", "split", "metric",
                "clean_recall", "perturbed_recall", "drop_pp", "drop_relative_pct"]
DRIFT_COLUMNS = ["dataset", "model", "setting", "cond", "modality", "ref",
                 "mean", "std", "p5", "p50", "p95", "split", "n"]


@dataclass(frozen=True)
class FeatureKey:
    model: str
    setting: str
    dataset: str
    split: str
    cond: str

    @property
    def relative_path(self):
        return Path(self.model) / self.setting / f"{self.dataset}_{self.split}_{self.cond}.npz"

    @property
    def group(self):
        return self.model, self.setting, self.dataset, self.split

    def with_(self, **kwargs):
        return replace(self, **kwargs)

    def columns(self):
        return dict(dataset=self.dataset, model=self.model, setting=self.setting,
                    cond=self.cond, split=self.split)


@dataclass
class Features:
    img: np.ndarray
    ids: np.ndarray
    txt: np.ndarray | None = None
    mapping: np.ndarray | None = None
    labels: np.ndarray | None = None


def sort_key(key):
    return (list(MODELS).index(key.model), SETTINGS.index(key.setting), key.dataset,
            ("val", "test").index(key.split), CONDITIONS.index(key.cond))


def selected_models(model="all"):
    if model == "all":
        return tuple(MODELS)
    if model not in MODELS:
        raise ValueError(f"Unknown model scope: {model}")
    return (model,)


def expected_keys(model="all"):
    return {
        FeatureKey(model_name, setting, dataset, split, cond)
        for model_name in selected_models(model) for setting in SETTINGS
        for dataset in ("flickr", "coco") for split in ("val", "test")
        for cond in MAIN_CONDS + (STRENGTH_CONDS if (dataset, split) == ("flickr", "val") else ())
    }


def discover_features(root, model="all"):
    root = Path(root).resolve()
    if not root.is_dir():
        raise ValueError(f"Feature directory does not exist: {root}")
    known = {key.relative_path.as_posix(): key for key in expected_keys(model)}
    search_root = root if model == "all" else root / model
    files = list(search_root.rglob("*.npz"))
    if not files:
        raise ValueError(f"No NPZ features found in {search_root}; .gitkeep is only a placeholder.")
    unknown = [p.relative_to(root).as_posix() for p in files if p.relative_to(root).as_posix() not in known]
    if unknown:
        raise ValueError(f"Unexpected NPZ paths (check for an extra features/ layer): {unknown[:5]}")
    return sorted((known[p.relative_to(root).as_posix()] for p in files), key=sort_key)


def coverage(keys, model="all"):
    present = set(keys)
    expected = expected_keys(model)
    if present - expected:
        raise ValueError("Feature keys contain models outside the selected scope")
    missing = sorted(expected - present, key=sort_key)
    return {"scope": model, "expected_models": list(selected_models(model)),
            "expected_files": len(expected), "available_files": len(present),
            "complete": not missing,
            "available": [k.relative_path.as_posix() for k in sorted(present, key=sort_key)],
            "missing": [k.relative_path.as_posix() for k in missing]}


def as_numpy(value):
    # Accept torch tensors without making PyTorch or CUDA a P5 runtime requirement.
    if hasattr(value, "detach"):
        value = value.detach().float().cpu().numpy()
    return np.asarray(value)


def unit_rows(value, name="features", dtype=np.float32):
    array = as_numpy(value)
    if array.ndim != 2 or not all(array.shape) or array.dtype.kind not in "fiu":
        raise ValueError(f"{name} must be a nonempty real [N, D] matrix")
    array = array.astype(np.float64, copy=False)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or infinity")
    norms = np.linalg.norm(array, axis=1)
    if not np.isfinite(norms).all() or np.any(norms == 0):
        raise ValueError(f"{name} contains zero vectors or invalid norms")
    return np.ascontiguousarray(array / norms[:, None], dtype=dtype)


@lru_cache(maxsize=8)
def _read_split(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_features(root, key, splits_root=None):
    path = Path(root) / key.relative_path
    try:
        with np.load(path, allow_pickle=False) as data:
            img = data["img_raw"]
            ids = data["img_ids"]
            if img.dtype != np.float32 or img.ndim != 2 or img.shape[1] != MODELS[key.model]:
                raise ValueError(f"img_raw must be float32 [N, {MODELS[key.model]}]")
            unit_rows(img, "img_raw")
            if ids.shape != (len(img),) or ids.dtype.kind not in "US":
                raise ValueError("img_ids must contain one string per image")
            ids = ids.astype(str)
            if len(set(ids.tolist())) != len(ids):
                raise ValueError("img_ids contains duplicate IDs")
            txt = mapping = None
            if key.cond == "clean":
                txt, mapping = data["txt_raw"], data["txt2img"]
                if txt.dtype != np.float32 or txt.shape != (5 * len(img), MODELS[key.model]):
                    raise ValueError("txt_raw must be float32 [5N, D]")
                unit_rows(txt, "txt_raw")
                if mapping.dtype != np.int64 or not np.array_equal(mapping, np.repeat(np.arange(len(img)), 5)):
                    raise ValueError("txt2img must be int64 with five consecutive captions per image")
            elif "txt_raw" in data.files or "txt2img" in data.files:
                raise ValueError("Perturbed files should not store text arrays; reuse the clean file")
            labels = data["labels"] if "labels" in data.files else None
            if key.dataset == "coco" and (labels is None or labels.dtype != np.int64 or labels.shape != (len(img),)):
                raise ValueError("COCO requires int64 labels [N]")
            if "_meta" in data.files:
                meta = json.loads(str(data["_meta"].item()))
                if not isinstance(meta, dict):
                    raise ValueError("_meta must contain a JSON object")
                for name, value in key.columns().items():
                    if name in meta and meta[name] != value:
                        raise ValueError(f"_meta.{name} disagrees with the file path")
                if meta.get("limit") is not None:
                    raise ValueError("Limited smoke features cannot be used as full evaluation data")
                if "count" in meta and meta["count"] != len(img):
                    raise ValueError("_meta.count disagrees with img_raw")
            if splits_root is not None:
                split = _read_split(str(Path(splits_root).resolve() / f"{key.dataset}_split.json"))
                if not np.array_equal(ids, np.asarray(split[key.split], dtype=str)):
                    raise ValueError("img_ids do not exactly match the official split order")
                if key.dataset == "coco":
                    expected_labels = np.asarray([split["labels"][id_] for id_ in ids], dtype=np.int64)
                    if not np.array_equal(labels, expected_labels):
                        raise ValueError("COCO labels do not match the split")
    except (OSError, ValueError, KeyError, EOFError, TypeError) as exc:
        raise ValueError(f"Invalid feature file {path}: {exc}") from exc
    return Features(img, ids, txt, mapping, labels)


def assert_aligned(left, right, description):
    if left.img.shape != right.img.shape or not np.array_equal(left.ids, right.ids):
        raise ValueError(f"Image IDs/order or shapes differ: {description}")
    if (left.labels is None) != (right.labels is None) or (
        left.labels is not None and not np.array_equal(left.labels, right.labels)
    ):
        raise ValueError(f"Labels differ: {description}")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_features(root, keys, splits_root=None, model="all"):
    refs, records = {}, []
    present = set(keys)
    for key in keys:
        if key.with_(cond="clean") not in present:
            raise ValueError(f"Missing clean file required for text: {key.with_(cond='clean').relative_path}")
        data = load_features(root, key, splits_root)
        identity = (key.dataset, key.split)
        if identity in refs and not np.array_equal(refs[identity], data.ids):
            raise ValueError(f"IDs/order differ across models/settings: {key.relative_path}")
        refs[identity] = data.ids
        path = Path(root) / key.relative_path
        records.append({"path": key.relative_path.as_posix(), "n_images": len(data.img),
                        "dimension": data.img.shape[1], "n_texts": 0 if data.txt is None else len(data.txt),
                        "bytes": path.stat().st_size, "sha256": sha256(path)})
    return {**coverage(keys, model), "split_check": splits_root is not None, "files": records}


def atomic_text(path, text, encoding="utf-8"):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding=encoding, newline="", dir=path.parent,
                                     prefix=f".{path.name}.", suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(text)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_json(path, value):
    atomic_text(path, json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def write_csv(path, frame):
    atomic_text(path, frame.to_csv(index=False, float_format="%.10g", lineterminator="\n"), encoding="utf-8-sig")


def add_input_arguments(parser):
    parser.add_argument("--model", choices=("all", *MODELS), default="all",
                        help="Evaluate only this model; complete coverage is 64 files per model")
    parser.add_argument("--features-root", type=Path, default=REPO_ROOT / "features")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "results")
    parser.add_argument("--no-split-check", action="store_true", help="For isolated synthetic tests only")
    parser.add_argument("--threads", type=int, default=2, help="BLAS CPU threads")
    return parser


def resolve_inputs(args):
    root, out = args.features_root.resolve(), args.out_dir.resolve()
    formal_outputs = {(REPO_ROOT / "results").resolve(),
                      (REPO_ROOT / "results" / "p5_clip").resolve(),
                      (REPO_ROOT / "results" / "p5_siglip").resolve()}
    if args.no_split_check and out in formal_outputs:
        raise ValueError("--no-split-check requires a separate --out-dir, e.g. results/p5_dummy")
    if args.model != "all" and out == (REPO_ROOT / "results").resolve():
        out = out / f"p5_{args.model}"
    if root == out or out.is_relative_to(root) or root.is_relative_to(out):
        raise ValueError("Input features and output results must be separate directories")
    if args.threads < 1:
        raise ValueError("--threads must be positive")
    return root, out, None if args.no_split_check else REPO_ROOT / "splits"


def cli_main(main):
    try:
        main()
    except (ValueError, FileNotFoundError) as exc:
        raise SystemExit(f"P5 error: {exc}") from exc
