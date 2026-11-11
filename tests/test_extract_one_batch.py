"""Run one real pretrained-weight batch on synthetic or P1 Flickr/COCO inputs.

Run with ``python -m tests.test_extract_one_batch --model all --device cuda``.
The ordinary pytest suite skips downloads unless MMREP_RUN_REAL_MODEL_TESTS=1.
This checks integration and export, not dataset quality or retrieval accuracy.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import gc
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import time

import numpy as np
from PIL import Image
import pytest
import torch
import torch.nn.functional as F

from data.config import REPO_ROOT, load_models_cfg
from data.datasets import collate_fn, get_dataset
from data.preprocess import preprocess_pil


class SyntheticDataset(torch.utils.data.Dataset):
    """Deterministic RGB gradients and captions, following the P1 dataset API."""

    def __init__(self, model, count=2, cond="clean"):
        self.name, self.split, self.cond, self.model = "flickr", "val", cond, model
        self.ids = [f"synthetic_{i:03d}.png" for i in range(count)]
        self.meta = {
            "captions": {
                image_id: [f"A synthetic COLOR gradient number {i}, caption {j}."
                           for j in range(5)]
                for i, image_id in enumerate(self.ids)
            },
            "labels": {image_id: -1 for image_id in self.ids},
        }

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        # A non-square input exercises CLIP crop versus SigLIP squash.
        yy, xx = np.indices((193, 271), dtype=np.int32)
        array = np.stack([(xx + 31 * index) % 256,
                          (yy * 2 + 17 * index) % 256,
                          (xx + yy + 59 * index) % 256], axis=-1).astype(np.uint8)
        image_id = self.ids[index]
        return {
            "image": preprocess_pil(Image.fromarray(array), self.model,
                                    cond=self.cond, idx=index),
            "captions": self.meta["captions"][image_id],
            "idx": index, "id": image_id, "label": -1,
        }


def _versions():
    result = {"python": platform.python_version(), "torch": torch.__version__,
              "cuda_runtime": torch.version.cuda}
    for package in ("transformers", "peft", "numpy", "huggingface-hub"):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = None
    return result


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run_one_batch(model_name, *, device="cuda", cache_dir=None,
                  local_files_only=False, out_dir=None, batch_size=2, amp=True,
                  dataset_name="synthetic", split="val", data_root=None, splits_dir=None,
                  revision=None):
    """Raise on any failed check; persist a report and isolated test NPZ files."""
    from models.loader import load_model
    from extract.extract_features import extract_dataset

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if dataset_name not in ("synthetic", "flickr", "coco") or split not in ("val", "test"):
        raise ValueError("Select synthetic, flickr or coco, and a val/test split")
    real_inputs = dataset_name != "synthetic"
    output = Path(out_dir or Path(REPO_ROOT) / "logs" / "p2_smoke")
    if real_inputs:
        output = output / f"{dataset_name}_{split}"
    output = output / model_name
    for formal_root in ("features", "features_cifar"):
        if output.resolve().is_relative_to((Path(REPO_ROOT) / formal_root).resolve()):
            raise ValueError("Single-batch smoke outputs must stay outside formal feature directories")
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; choose --device cpu explicitly")
    use_amp = bool(amp and device.type == "cuda")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    report = {
        "model": model_name, "setting": "pretrained", "status": "running",
        "input_kind": (f"real {dataset_name} images and their split captions" if real_inputs
                       else "deterministic synthetic RGB images and synthetic captions"),
        "dataset": dataset_name, "split": split if real_inputs else None,
        "weights_kind": "real public pretrained weights",
        "scope": "integration and numerical checks only; not a dataset metric",
        "batch_size": batch_size, "device": str(device),
        "precision": "fp16 autocast with fp32 parameters" if use_amp else "fp32",
        "versions": _versions(), "tolerances": {"atol": 1e-5, "rtol": 1e-5},
    }
    wrapper = None
    try:
        wrapper = load_model(model_name, "pretrained", device=str(device),
                             cache_dir=cache_dir, local_files_only=local_files_only,
                             revision=revision)
        wrapper.eval()
        cfg = load_models_cfg()[model_name]
        report["hf_name"] = cfg["hf_name"]
        report["resolved_revision"] = getattr(wrapper.model.config, "_commit_hash", None)
        report["input_size"] = wrapper.input_size
        report["embed_dim"] = wrapper.embed_dim
        report["model_type"] = wrapper.model.config.model_type
        report["tokenizer_class"] = type(wrapper.tokenizer).__name__
        report["tokenizer_policy"] = cfg["tokenizer"]
        report["logit_parameters"] = {
            key: {"value": float(getattr(wrapper.model, key).detach().cpu()),
                  "dtype": str(getattr(wrapper.model, key).dtype)}
            for key in ("logit_scale", "logit_bias") if hasattr(wrapper.model, key)
        }
        report["parameter_count"] = sum(p.numel() for p in wrapper.model.parameters())
        if device.type == "cuda":
            report["gpu"] = torch.cuda.get_device_name(device)
        def inputs(condition):
            if real_inputs:
                return get_dataset(dataset_name, split, condition, model_name,
                                   data_root=data_root, splits_dir=splits_dir)
            return SyntheticDataset(model_name, batch_size, condition)

        dataset = inputs("clean")
        if len(dataset) < batch_size:
            raise ValueError(f"Requested {batch_size} images but split contains {len(dataset)}")
        batch = collate_fn([dataset[i] for i in range(batch_size)])
        report["img_ids"] = batch["id"]
        pixels = batch["image"].to(device)
        captions = [group[0] for group in batch["captions"]]
        token_cpu = wrapper.tokenize(captions)
        assert tuple(token_cpu["input_ids"].shape) == (
            batch_size, cfg["tokenizer"]["max_length"])
        assert all(value.device.type == "cpu" for value in token_cpu.values())
        if cfg["tokenizer"]["lowercase"]:
            lower = wrapper.tokenize([caption.lower() for caption in captions])
            assert set(token_cpu) == set(lower)
            assert all(torch.equal(token_cpu[key], lower[key]) for key in token_cpu)
        tokens = {key: value.to(device) for key, value in token_cpu.items()}
        context = lambda: (torch.autocast("cuda", dtype=torch.float16) if use_amp else nullcontext())

        def forward(call, *args, **kwargs):
            # End each autocast scope to release cached casts between towers.
            # This matters when another application occupies most GPU memory.
            with torch.inference_mode(), context():
                return call(*args, **kwargs)

        img = forward(wrapper.encode_image, pixels)
        txt = forward(wrapper.encode_text, **tokens)
        img_again = forward(wrapper.encode_image, pixels)
        txt_again = forward(wrapper.encode_text, **tokens)
        img_native = forward(wrapper.model.get_image_features, pixel_values=pixels)
        txt_native = forward(wrapper.model.get_text_features, **tokens)
        for modality, value, again, native in (
                ("image", img, img_again, img_native), ("text", txt, txt_again, txt_native)):
            assert tuple(value.shape) == (batch_size, wrapper.embed_dim)
            assert torch.isfinite(value).all(), f"{modality} contains NaN/Inf"
            norms = value.float().norm(dim=-1)
            assert (norms > 0).all(), f"{modality} has zero norm"
            torch.testing.assert_close(value, again, **report["tolerances"])
            # The wrapper must not silently normalize or use a classifier output.
            torch.testing.assert_close(value, native, **report["tolerances"])
            torch.testing.assert_close(F.normalize(value.float(), dim=-1).norm(dim=-1),
                                       torch.ones(batch_size, device=device), atol=1e-6, rtol=1e-6)
            report[modality] = {
                "shape": list(value.shape), "dtype": str(value.dtype),
                "raw_norms": norms.cpu().tolist(),
                "repeat_max_abs_diff": (value - again).abs().max().item(),
                "native_max_abs_diff": (value - native).abs().max().item(),
            }

        paths = {}
        # RunLogger resets its per-run peak, so retain peaks from each phase.
        phase_peaks = ([torch.cuda.max_memory_allocated(device)]
                       if device.type == "cuda" else [])
        for condition in ("clean", "noise"):
            condition_ds = dataset if condition == "clean" else inputs(condition)
            suffix = f".limit{batch_size}.npz" if real_inputs else ".npz"
            target = output / f"{dataset_name}_{condition}{suffix}"
            extraction = extract_dataset(
                wrapper, condition_ds, target, batch_size=batch_size,
                text_batch_size=batch_size, num_workers=0, amp=use_amp,
                log_dir=output, model_name=model_name, setting="pretrained",
                limit=batch_size if real_inputs else None,
                cache_identity={"input_kind": report["input_kind"] if real_inputs else "p2_synthetic_smoke_v1",
                                "resolved_revision": report["resolved_revision"],
                                "batch_size": batch_size})
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                phase_peaks.append(torch.cuda.max_memory_allocated(device))
            with np.load(target, allow_pickle=False) as features:
                assert features["img_ids"].tolist() == condition_ds.ids[:batch_size]
                if dataset_name == "coco":
                    np.testing.assert_array_equal(features["labels"], batch["label"].numpy())
                assert features["img_raw"].shape == (batch_size, wrapper.embed_dim)
                assert features["img_raw"].dtype == np.float32
                assert np.isfinite(features["img_raw"]).all()
                if condition == "clean":
                    np.testing.assert_allclose(features["img_raw"], img.float().cpu().numpy(),
                                               atol=1e-5, rtol=1e-5)
                    assert features["txt_raw"].shape == (5 * batch_size, wrapper.embed_dim)
                    assert features["txt_raw"].dtype == np.float32
                    assert np.isfinite(features["txt_raw"]).all()
                    np.testing.assert_array_equal(features["txt2img"], np.repeat(np.arange(batch_size), 5))
                    # All captions use the same text microbatch size as the direct call.
                    flat_captions = [caption for group in batch["captions"] for caption in group]
                    direct_text = []
                    for start in range(0, len(flat_captions), batch_size):
                        chunk = {k: v.to(device) for k, v in
                                 wrapper.tokenize(flat_captions[start:start + batch_size]).items()}
                        direct_text.append(forward(wrapper.encode_text, **chunk).float().cpu())
                    np.testing.assert_allclose(features["txt_raw"], torch.cat(direct_text).numpy(),
                                               atol=1e-5, rtol=1e-5)
                else:
                    assert "txt_raw" not in features and "txt2img" not in features
                    disturbed = collate_fn([condition_ds[i] for i in range(batch_size)])
                    expected = forward(wrapper.model.get_image_features,
                                       pixel_values=disturbed["image"].to(device))
                    np.testing.assert_allclose(features["img_raw"], expected.float().cpu().numpy(),
                                               atol=1e-5, rtol=1e-5)
            paths[condition] = {"path": str(target.resolve()), "export": extraction}
        report["exports"] = paths
        report["checks"] = ["image/text shape", "finite and nonzero raw embeddings",
                            "L2 normalization", "repeat inference", "raw native-model parity",
                            "fixed tokenizer length and case policy", "clean 5-caption export",
                            "caption ordering", "noise image-only export", "NPZ reread"]
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            phase_peaks.append(torch.cuda.max_memory_allocated(device))
            report["gpu_mem_peak_MB"] = max(phase_peaks) / 2**20
        report["status"] = "passed"
    except Exception as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        report["wall_sec"] = time.perf_counter() - started
        _write_json(output / "report.json", report)
        del wrapper
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return report


@pytest.mark.skipif(os.environ.get("MMREP_RUN_REAL_MODEL_TESTS") != "1",
                    reason="opt-in real pretrained weights; run the module CLI explicitly")
@pytest.mark.parametrize("model_name", ["clip", "siglip"])
def test_real_pretrained_one_batch(model_name, tmp_path):
    report = run_one_batch(
        model_name, device=os.environ.get("MMREP_TEST_DEVICE", "cuda"),
        cache_dir=os.environ.get("HF_HUB_CACHE"),
        local_files_only=os.environ.get("HF_HUB_OFFLINE") == "1", out_dir=tmp_path)
    assert report["status"] == "passed"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["clip", "siglip", "all"], default="all")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cache-dir")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=Path(REPO_ROOT) / "logs" / "p2_smoke")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--dataset", choices=["synthetic", "coco", "flickr"], default="synthetic")
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--data-root")
    parser.add_argument("--splits-dir")
    parser.add_argument("--revision", help="Optional HF model revision for this run")
    parser.add_argument("--fp32", action="store_true", help="Disable CUDA autocast")
    args = parser.parse_args()
    for name in (["clip", "siglip"] if args.model == "all" else [args.model]):
        result = run_one_batch(name, device=args.device, cache_dir=args.cache_dir,
                               local_files_only=args.local_files_only, out_dir=args.out_dir,
                               batch_size=args.batch_size, amp=not args.fp32,
                               dataset_name=args.dataset, split=args.split,
                               data_root=args.data_root, splits_dir=args.splits_dir,
                               revision=args.revision)
        location = args.out_dir if args.dataset == "synthetic" else args.out_dir / f"{args.dataset}_{args.split}"
        print(json.dumps({"model": name, "status": result["status"],
                          "report": str(location / name / "report.json")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
