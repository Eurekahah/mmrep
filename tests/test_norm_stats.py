"""P2 norm tables and frozen-image comparisons use small temporary NPZ fixtures."""
import csv
import hashlib
import json

import numpy as np
import pytest

from data.config import load_models_cfg
from extract import norm_stats as ns
from extract.extract_features import _digest, _json


@pytest.fixture
def inputs(tmp_path):
    splits, features = tmp_path / "splits", tmp_path / "features"
    splits.mkdir()
    for name in ("flickr", "coco"):
        manifest = {"dataset": name, "train": [], "val": ["z.jpg", "a.jpg"],
                    "test": ["c.jpg", "b.jpg"]}
        ids = manifest["val"] + manifest["test"]
        manifest["captions"] = {i: [i + str(j) for j in range(5)] for i in ids}
        manifest["labels"] = {i: j + 1 for j, i in enumerate(ids)}
        (splits / f"{name}_split.json").write_text(json.dumps(manifest), encoding="utf-8")
    sources, _ = ns._read_splits(splits)
    config_hashes = {name: hashlib.sha256((ns.REPO_ROOT / "configs" / name).read_bytes()).hexdigest()
                     for name in ("models.yaml", "perturb.yaml")}
    for job in ns.stage_jobs("all"):
        ids, labels, source_hash = sources[job.dataset, job.split]
        cfg = load_models_cfg()[job.model]
        weights = {"resolved_revision": f"{job.model}-fixture-revision"}
        if job.setting != "pretrained":
            weights["checkpoint_sha256"] = {"file": hashlib.sha256(job.setting.encode()).hexdigest()}
            weights["training_meta_sha256"] = "b" * 64
        meta = dict(job._asdict(), schema=1, limit=None, count=len(ids),
                    embed_dim=cfg["embed_dim"], input_size=cfg["input_size"],
                    mean=cfg["mean"], std=cfg["std"], model_config=cfg,
                    source_sha256=source_hash, precision="fp32", device="cpu",
                    device_name="CPU", batch_size=2, text_batch_size=2 if job.cond == "clean" else None,
                    config_sha256=config_hashes, weights=weights)
        raw = np.zeros((len(ids), cfg["embed_dim"]), dtype=np.float32)
        raw[:, :2] = [[3, 4], [0, 12]]
        arrays = dict(_meta=np.asarray(_json(meta)), img_raw=raw, img_ids=np.asarray(ids))
        if job.cond == "clean":
            txt = np.zeros((len(ids) * 5, cfg["embed_dim"]), dtype=np.float32)
            txt[:, 0] = np.arange(1, len(txt) + 1)
            arrays.update(txt_raw=txt, txt2img=np.repeat(np.arange(len(ids), dtype=np.int64), 5))
        if labels is not None:
            arrays["labels"] = np.asarray(labels, dtype=np.int64)
        path = job.path(features)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **arrays)
    return features, splits, tmp_path / "results"


def _change(path, change):
    with np.load(path, allow_pickle=False) as data:
        arrays = {k: data[k] for k in data.files}
    meta = json.loads(arrays["_meta"].item())
    change(arrays, meta)
    arrays["_meta"] = np.asarray(_json(meta))
    np.savez_compressed(path, **arrays)


def _csv(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def test_pretrained_stage_exports_known_row_norms_and_valid_figures(inputs):
    features, splits, results = inputs
    receipt = ns.generate("pretrained", features, splits, results)
    out = results / "p2_stages" / "pretrained"
    assert not (results / "norm_stats.csv").exists()
    assert receipt["status"] == "passed" and receipt["validated_files"] == 32
    assert receipt["freeze_pairs"] == receipt["extra_reference_files"] == 0
    rows = _csv(out / "norm_stats.csv")
    assert len(rows) == receipt["norm_rows"] == 40
    assert tuple(rows[0]) == ns.STAT_COLUMNS
    row = next(r for r in rows if r["modality"] == "image")
    np.testing.assert_allclose([float(row[k]) for k in ("mean", "std", "p5", "p50", "p95")],
                               [8.5, 3.5, 5.35, 8.5, 11.65])
    texts = [row for row in rows if row["modality"] == "text"]
    assert len(texts) == 8 and all(row["cond"] == "clean" for row in texts)
    assert all(float(row["mean"]) == 5.5 for row in texts)
    assert _csv(out / "lit_frozen_check.csv") == []
    assert json.loads((out / "norm_stats_validation.json").read_text(encoding="utf-8")) == receipt
    assert len(receipt["artifacts"]["histograms"]) == 16
    for figure in receipt["artifacts"]["histograms"]:
        assert (out / figure).read_bytes().startswith(b"\x89PNG\r\n")
        assert receipt["artifact_sha256"][figure] == ns.file_sha256(out / figure)
    for path, sha in receipt["feature_sha256"].items():
        assert sha == ns.file_sha256(features / path)


def test_all_and_incremental_stages_report_lit_mismatch_and_provenance(inputs, monkeypatch):
    features, splits, results = inputs
    monkeypatch.setattr(ns, "_write_histograms", lambda out, groups: [])
    mismatch = ns.Job("clip", "lit", "flickr", "val", "noise")
    missing = mismatch._replace(cond="occl")
    changed_device = mismatch._replace(cond="clean")
    changed_batch = mismatch._replace(cond="noise_lo")
    _change(mismatch.path(features), lambda arrays, meta: arrays["img_raw"].__setitem__((1, 0), 0.0625))
    _change(missing.path(features), lambda arrays, meta: meta.pop("batch_size"))
    _change(changed_device.path(features), lambda arrays, meta: meta.update(device_name="different CPU"))
    _change(changed_batch.path(features), lambda arrays, meta: meta.update(batch_size=1))
    assert ns.main(["--stage", "all", "--features-root", str(features), "--splits-dir", str(splits),
                    "--results-dir", str(results)]) == 2
    all_results = json.loads((results / "norm_stats_validation.json").read_text(encoding="utf-8"))
    assert all_results["validated_files"] == 128 and all_results["norm_rows"] == 160
    assert all_results["freeze_pairs"] == 32 and all_results["freeze_reviews"] == 4
    assert all_results["status"] == "needs_review" and all_results["extra_reference_files"] == 0
    rows = _csv(results / "lit_frozen_check.csv")
    bad = next(row for row in rows if row["max_abs_img_id"])
    assert float(bad["max_abs"]) == 0.0625 and bad["max_abs_img_id"] == "a.jpg"
    assert "nonzero image difference" in bad["review_reason"]
    assert {row["review_reason"] for row in rows if row["status"] == "needs_review"} == {
        "nonzero image difference", "missing batch_size", "different device_name", "different batch_size"}
    for stage, expected_reviews in [("clip-finetuned", 4), ("siglip-finetuned", 0)]:
        receipt = ns.generate(stage, features, splits, results)
        assert receipt["expected_files"] == receipt["validated_files"] == 48
        assert receipt["extra_reference_files"] == 16 and receipt["freeze_pairs"] == 16
        assert receipt["norm_rows"] == 60 and receipt["freeze_reviews"] == expected_reviews


def test_missing_full_file_or_reference_aborts_before_publishing(inputs):
    features, splits, results = inputs
    ns.Job("siglip", "pretrained", "coco", "test", "clean").path(features).unlink()
    for stage in ("all", "siglip-finetuned"):
        with pytest.raises(FileNotFoundError, match="missing 1 feature"):
            ns.generate(stage, features, splits, results)
    assert not results.exists()


@pytest.mark.parametrize("damage", [
    "id_order", "mapping", "labels", "dtype", "nan", "precision", "caption_source",
    "wrong_path", "smoke_limit", "mixed_weight", "false_cuda",
])
def test_corruption_or_foreign_inputs_rejected(inputs, damage):
    features, splits, results = inputs
    job = ns.Job("clip", "pretrained", "coco", "val", "clean")
    def change(arrays, meta):
        if damage == "id_order":
            arrays["img_ids"] = arrays["img_ids"][::-1]
        elif damage == "mapping":
            arrays["txt2img"][0] = 1
        elif damage == "labels":
            arrays["labels"] = arrays["labels"][::-1]
        elif damage == "dtype":
            arrays["txt_raw"] = arrays["txt_raw"].astype(np.float16)
        elif damage == "nan":
            arrays["img_raw"][0, 0] = np.nan
        elif damage == "precision":
            meta["precision"] = "other"
        elif damage == "caption_source":
            meta["source_sha256"] = _digest({"wrong": "caption order"})
        elif damage == "wrong_path":
            meta["cond"] = "noise"
        elif damage == "smoke_limit":
            meta["limit"] = len(arrays["img_ids"])
        elif damage == "mixed_weight":
            meta["weights"]["resolved_revision"] = "different-revision"
        elif damage == "false_cuda":
            meta["precision"] = "cuda_fp16_autocast"
    _change(job.path(features), change)
    with pytest.raises(ValueError):
        ns.generate("pretrained", features, splits, results)
    assert not results.exists()


def test_replay_uses_real_tiny_clip_checkpoint_and_training_metadata(inputs, monkeypatch, tmp_path):
    from copy import deepcopy
    from PIL import Image
    import torch
    from transformers import CLIPConfig, CLIPModel, CLIPTokenizer
    from data.datasets import get_dataset
    from extract import recheck_lit as replay
    from extract.extract_features import weight_identity
    from models import load_model
    from models.checkpoint_receipt import read_training_receipt

    features, splits, results = inputs
    job = ns.Job("clip", "lit", "flickr", "val", "noise")
    ref = job._replace(setting="pretrained")
    config = load_models_cfg()["clip"]
    small = dict(hidden_size=16, intermediate_size=24, num_hidden_layers=1, num_attention_heads=2)
    torch.manual_seed(42)
    native = CLIPModel(CLIPConfig(text_config=dict(small, vocab_size=32),
                                  vision_config=dict(small, image_size=config["input_size"],
                                                     patch_size=config["patch_size"]),
                                  projection_dim=config["embed_dim"]))
    native.config._commit_hash = "clip-fixture-revision"
    monkeypatch.setattr(CLIPModel, "from_pretrained", lambda *a, **kw: deepcopy(native))
    monkeypatch.setattr(CLIPTokenizer, "from_pretrained", lambda *a, **kw: object())
    checkpoint = tmp_path / "ckpt" / "clip" / "lit"
    checkpoint.mkdir(parents=True)
    state = {key: value.detach().clone() + 0.01 for key, value in native.named_parameters()
             if key.startswith(("text_model.", "text_projection."))}
    torch.save(state, checkpoint / "trainable_state.pt")
    meta = {"lr": 0.0001, "epochs": 1, "batch_size": 1, "seed": 42, "wall_sec": 1.25,
            "gpu_mem_peak_MB": 0, "trainable_params": sum(x.numel() for x in state.values()),
            "epoch_loss": [1.0], "model": "clip", "setting": "lit"}
    (checkpoint / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    raw_root = tmp_path / "raw"
    images = raw_root / "flickr30k" / "flickr30k-images"
    images.mkdir(parents=True)
    for i, image_id in enumerate(["z.jpg", "a.jpg"]):
        Image.new("RGB", (47, 61), color=(100, i * 70, 255)).save(images / image_id)
    ds = get_dataset("flickr", "val", "noise", "clip", data_root=raw_root, splits_dir=splits)
    metas = {}
    for entry in (ref, job):
        receipt = read_training_receipt(tmp_path / "ckpt", "clip", entry.setting)
        wrapper = load_model("clip", entry.setting, ckpt_root=tmp_path / "ckpt", device="cpu", local_files_only=True)
        ident = weight_identity(wrapper, tmp_path / "ckpt", "clip", entry.setting, None, receipt)
        with torch.no_grad():
            raw = torch.cat([wrapper.encode_image(ds[i]["image"].unsqueeze(0)) for i in range(len(ds))]).numpy()
        if entry == job:
            raw[1, 0] += 0.0625
        def update(arrays, metadata):
            arrays["img_raw"] = raw
            metadata.update(batch_size=1, weights=ident)
            metas[entry] = metadata
        _change(entry.path(features), update)
    results.mkdir()
    archive = results / "lit_frozen_check.csv"
    row = ns._freeze_row(job, features, metas, ["z.jpg", "a.jpg"])
    ns._write_csv(archive, ns.FREEZE_COLUMNS, [row])
    kwargs = dict(review_csv=archive, features_root=features, splits_dir=splits, data_root=raw_root,
                  ckpt_root=tmp_path / "ckpt", cache_dir=None, device="cpu", out=results / "replay.json")
    report = replay.recheck(job, **kwargs)
    assert report["status"] == "measured", report
    assert report["batch"]["ids"] == ["a.jpg"] and report["batch"]["start_inclusive"] == 1
    values = report["measurements"]
    assert values["archived_lit_vs_pretrained"]["max_abs"] == pytest.approx(0.0625)
    assert values["lit_replay_vs_archive"]["max_abs"] == pytest.approx(0.0625)
    assert values["fp32_lit_vs_pretrained"] == {"max_abs": 0.0, "mean_abs": 0.0, "relative_l2": 0.0}
    assert values["original_precision_pretrained_repeat"]["max_abs"] == 0.0
    assert report["vision_state_sha256"]["pretrained"] == report["vision_state_sha256"]["lit"]
    saved = json.loads((results / "replay.json").read_text(encoding="utf-8"))
    assert saved == report and "not reproduced" in report["interpretation"]
    # The tensor checkpoint is unchanged here; alone a changed training receipt
    # must be reported explicitly rather than giving another measured verdict.
    meta["wall_sec"] = 3
    (checkpoint / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    report = replay.recheck(job, **kwargs)
    assert report["status"] == "original_inputs_changed" and "training_meta_sha256" in report["error"]
    _change(job.path(features), lambda arr, meta: arr["img_raw"].__setitem__((0, 0), 7))
    report = replay.recheck(job, **kwargs)
    assert report["status"] == "original_inputs_changed" and "feature file" in report["error"]


def test_replay_records_incompatible_original_batches_before_loading_models(inputs, monkeypatch):
    from extract import recheck_lit as replay

    features, splits, results = inputs
    job = ns.Job("siglip", "lit", "coco", "test", "clean")
    _change(job.path(features), lambda arr, meta: meta.update(batch_size=1))
    monkeypatch.setattr(ns, "_write_histograms", lambda out, groups: [])
    ns.generate("siglip-finetuned", features, splits, results)
    monkeypatch.setattr(replay, "load_model", lambda *a, **kw: pytest.fail("should not load weights"))
    folder = results / "p2_stages" / "siglip-finetuned"
    report = replay.recheck(job, review_csv=folder / "lit_frozen_check.csv", features_root=features,
                            splits_dir=splits, data_root=None, ckpt_root=None, cache_dir=None,
                            device="cpu", out=folder / "replay.json")
    assert report["status"] == "original_inputs_changed"
    assert "batch_size" in report["error"] and "measurements" not in report
