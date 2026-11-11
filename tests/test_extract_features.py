"""CPU-only contract tests; no downloaded weights or real image data needed."""
import json
import hashlib
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from extract.extract_features import FeatureCacheError, RUNTIME_KEYS, _retry_oom, build_jobs, choose_batches, extract_dataset


class FakeDataset:
    model = "clip"
    split = "val"

    def __init__(self, name="flickr", cond="clean"):
        self.name, self.cond = name, cond
        self.ids = ["z.jpg", "a.jpg", "m.jpg"]  # Intentionally not sorted.
        self.meta = {
            "captions": {id_: [str(10 * i + j + 1) for j in range(5)] for i, id_ in enumerate(self.ids)},
            "labels": {id_: i + 7 for i, id_ in enumerate(self.ids)},
        }

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        id_ = self.ids[idx]
        return {"image": torch.full((3, 2, 2), float(idx + 2)),
                "captions": self.meta["captions"][id_], "idx": idx,
                "id": id_, "label": self.meta["labels"][id_]}


class FakeWrapper(torch.nn.Module):
    embed_dim = 2
    input_size = 2
    mean = [0.5] * 3
    std = [0.5] * 3
    device = torch.device("cpu")

    def __init__(self):
        super().__init__()
        self.image_calls = 0
        self.text_calls = 0
        self.text_batch_sizes = []

    def encode_image(self, pixels):
        assert not self.training and not torch.is_grad_enabled()
        self.image_calls += 1
        values = pixels[:, 0, 0, 0]
        return torch.stack((values, values * 2), dim=1)

    def tokenize(self, captions):
        return {"input_ids": torch.tensor([int(cap) for cap in captions])}

    def encode_text(self, input_ids):
        assert not self.training and not torch.is_grad_enabled()
        self.text_calls += 1
        self.text_batch_sizes.append(len(input_ids))
        return torch.stack((input_ids.float(), input_ids.float() * 3), dim=1)


@pytest.fixture(autouse=True)
def cpu_logger(monkeypatch):
    monkeypatch.setattr("tools.logger._cuda", lambda: None)


def run_extract(tmp_path, dataset=None, wrapper=None, **kwargs):
    dataset = dataset or FakeDataset()
    wrapper = wrapper or FakeWrapper()
    path = tmp_path / "result.npz"
    result = extract_dataset(wrapper, dataset, path, batch_size=2, text_batch_size=3,
                             num_workers=0, amp=False, log_dir=tmp_path / "logs", **kwargs)
    return path, result, wrapper


def test_clean_preserves_split_order_five_captions_and_raw_scale(tmp_path):
    path, result, wrapper = run_extract(tmp_path)
    assert result["status"] == "written"
    assert result["n_images"] == 3 and result["n_texts"] == 15
    with np.load(path, allow_pickle=False) as data:
        assert data["img_ids"].tolist() == ["z.jpg", "a.jpg", "m.jpg"]
        np.testing.assert_array_equal(data["img_raw"], [[2, 4], [3, 6], [4, 8]])
        np.testing.assert_array_equal(data["txt_raw"][:, 0], [1, 2, 3, 4, 5, 11, 12, 13, 14, 15, 21, 22, 23, 24, 25])
        np.testing.assert_array_equal(data["txt2img"], np.repeat(np.arange(3), 5))
        assert data["img_raw"].dtype == data["txt_raw"].dtype == np.float32
        assert data["txt2img"].dtype == np.int64
        assert "labels" not in data.files
    assert max(wrapper.text_batch_sizes) <= 3


@pytest.mark.parametrize("cond", ["noise", "occl", "noise_lo", "noise_hi", "occl_lo", "occl_hi"])
def test_perturbed_files_do_not_duplicate_text(tmp_path, cond):
    path, _, wrapper = run_extract(tmp_path, FakeDataset(cond=cond))
    with np.load(path, allow_pickle=False) as data:
        assert set(data.files) == {"img_raw", "img_ids", "_meta"}
    assert wrapper.text_calls == 0


def test_coco_labels_follow_same_image_order(tmp_path):
    path, _, _ = run_extract(tmp_path, FakeDataset(name="coco", cond="noise"))
    with np.load(path, allow_pickle=False) as data:
        np.testing.assert_array_equal(data["labels"], [7, 8, 9])
        assert data["labels"].dtype == np.int64


def test_resume_validates_then_skips_without_encoding(tmp_path):
    path, _, _ = run_extract(tmp_path)
    original = path.read_bytes()
    _, result, wrapper = run_extract(tmp_path)
    assert result["status"] == "skipped"
    assert wrapper.image_calls == wrapper.text_calls == 0
    assert path.read_bytes() == original
    records = [json.loads(line) for line in (tmp_path / "logs" / "runs.jsonl").read_text(encoding="utf-8").splitlines()]
    assert records[-1]["cache_status"] == "validated_skip"


@pytest.mark.parametrize("damage", ["id_order", "nan", "text_mapping", "dtype", "missing_field"])
def test_invalid_existing_cache_is_rejected_without_overwrite(tmp_path, damage):
    path, _, _ = run_extract(tmp_path)
    with np.load(path, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    if damage == "id_order":
        arrays["img_ids"] = arrays["img_ids"][::-1]
    elif damage == "nan":
        arrays["img_raw"][0, 0] = np.nan
    elif damage == "text_mapping":
        arrays["txt2img"][0] = 1
    elif damage == "dtype":
        arrays["img_raw"] = arrays["img_raw"].astype(np.float16)
    elif damage == "missing_field":
        del arrays["_meta"]
    np.savez(path, **arrays)
    original = path.read_bytes()
    with pytest.raises(FeatureCacheError):
        run_extract(tmp_path)
    assert path.read_bytes() == original


def test_changed_caption_or_weight_identity_cannot_reuse_cache(tmp_path):
    run_extract(tmp_path, cache_identity={"revision": "first"})
    with pytest.raises(FeatureCacheError, match="metadata"):
        run_extract(tmp_path, cache_identity={"revision": "second"})
    changed = FakeDataset()
    changed.meta["captions"]["z.jpg"][0] = "99"
    with pytest.raises(FeatureCacheError, match="metadata"):
        run_extract(tmp_path, changed, cache_identity={"revision": "first"})


def test_direct_api_derives_actual_setting_and_rejects_wrong_label(tmp_path):
    wrapper = FakeWrapper()
    wrapper.model_name = "clip"
    wrapper.setting = "lit"
    path, _, _ = run_extract(tmp_path, wrapper=wrapper)
    with np.load(path) as data:
        metadata = json.loads(data["_meta"].item())
    assert metadata["setting"] == metadata["weights"]["setting"] == "lit"
    with pytest.raises(ValueError, match="loaded wrapper"):
        run_extract(tmp_path, wrapper=wrapper, setting="pretrained")


def test_changed_wrapper_model_config_rejects_cache(tmp_path):
    wrapper = FakeWrapper()
    wrapper.model_config = {"tokenizer": {"lowercase": False}}
    run_extract(tmp_path, wrapper=wrapper)
    wrapper.model_config["tokenizer"]["lowercase"] = True
    with pytest.raises(FeatureCacheError, match="metadata"):
        run_extract(tmp_path, wrapper=wrapper)


def test_truncated_npz_is_rejected_without_overwrite(tmp_path):
    path, _, _ = run_extract(tmp_path)
    truncated = path.read_bytes()[:64]
    path.write_bytes(truncated)
    with pytest.raises(FeatureCacheError):
        run_extract(tmp_path)
    assert path.read_bytes() == truncated


def test_limit_requires_separate_filename_and_produces_only_prefix(tmp_path):
    wrapper, dataset = FakeWrapper(), FakeDataset()
    with pytest.raises(ValueError, match="Limited output"):
        extract_dataset(wrapper, dataset, tmp_path / "flickr_val_clean.npz", limit=2)
    path = tmp_path / "flickr_val_clean.limit2.npz"
    result = extract_dataset(wrapper, dataset, path, limit=2, amp=False, log_dir=tmp_path / "logs")
    assert result["n_images"] == 2 and result["n_texts"] == 10
    with np.load(path) as data:
        assert data["img_ids"].tolist() == dataset.ids[:2]


def test_failed_encoder_leaves_no_partial_output(tmp_path):
    class BrokenWrapper(FakeWrapper):
        def encode_image(self, pixels):
            raise RuntimeError("simulated inference error")

    with pytest.raises(RuntimeError, match="simulated"):
        run_extract(tmp_path, wrapper=BrokenWrapper())
    assert not (tmp_path / "result.npz").exists()
    assert not list(tmp_path.glob("*.tmp"))
    record = json.loads((tmp_path / "logs" / "runs.jsonl").read_text(encoding="utf-8").splitlines()[-1])
    assert record["status"] == "error"


def test_missing_caption_is_rejected_before_inference(tmp_path):
    dataset = FakeDataset()
    dataset.meta["captions"]["z.jpg"].pop()
    wrapper = FakeWrapper()
    with pytest.raises(ValueError, match="five"):
        run_extract(tmp_path, dataset, wrapper)
    assert wrapper.image_calls == 0


def test_job_matrix_covers_main_and_flickr_val_strength_only():
    main = build_jobs()
    strength = build_jobs(cond="strength")
    complete = build_jobs(cond="complete")
    assert len(main) == 96 and len(strength) == 32 and len(complete) == 128
    assert len(set(complete)) == 128
    assert set(complete) == set(main) | set(strength)
    assert all(job[2:4] == ("flickr", "val") for job in strength)
    with pytest.raises(ValueError, match="Flickr val"):
        build_jobs(dataset="coco", cond="strength")


def test_batch_log_and_cache_skip_use_original_producer(tmp_path):
    file, produced, _ = run_extract(tmp_path)
    with np.load(file) as archive:
        meta = json.loads(archive["_meta"].item())
    fields = dict(device="cpu", device_name="CPU", batch_size=2, text_batch_size=3)
    assert all(meta[k] == v == produced[k] for k, v in fields.items())
    resumed = extract_dataset(FakeWrapper(), FakeDataset(), file, batch_size=1,
                              log_dir=tmp_path / "logs")
    assert resumed["status"] == "skipped"
    records = [json.loads(row) for row in
               (tmp_path / "logs" / "runs.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [row["cache_status"] for row in records] == ["written", "validated_skip"]
    assert all(all(record[k] == v for k, v in fields.items()) for record in records)
    assert records[1]["work"] == "cache_validation" and resumed["runtime_metadata_complete"]


def test_legacy_cache_is_preserved_and_identified_as_incomplete(tmp_path):
    file, _, _ = run_extract(tmp_path)
    with np.load(file, allow_pickle=False) as archive:
        arrays = dict(archive)
    meta = json.loads(arrays["_meta"].item())
    arrays["_meta"] = np.asarray(json.dumps({k: v for k, v in meta.items() if k not in RUNTIME_KEYS}))
    np.savez(file, **arrays)
    original = file.read_bytes()
    _, resumed, wrapper = run_extract(tmp_path)
    assert resumed["status"] == "skipped" and resumed["batch_size"] is None
    assert not resumed["runtime_metadata_complete"] and wrapper.image_calls == 0
    assert file.read_bytes() == original


def test_cuda_oom_retry_halves_and_restarts_without_gpu(monkeypatch):
    calls = []
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    def operation(images, texts, retry):
        calls.append((images, texts, retry))
        if images > 2:
            raise RuntimeError("CUDA out of memory")
        return "finished"
    assert _retry_oom(operation, 8, 12, device=torch.device("cuda")) == "finished"
    assert calls == [(8, 12, 0), (4, 6, 1), (2, 3, 2)]
    def fail(_images, _texts, _retry):
        raise RuntimeError("CUDA out of memory")
    with pytest.raises(RuntimeError, match="out of memory"):
        _retry_oom(fail, 1, 1, device=torch.device("cuda"))
    with pytest.raises(RuntimeError, match="out of memory"):
        _retry_oom(fail, 8, 8, device=torch.device("cpu"))


def test_lit_reuses_pretrained_batches_and_respects_override(tmp_path):
    args = SimpleNamespace(device="cuda", batch_size=None, text_batch_size=None)
    assert choose_batches(args, "clip", None, None, "cuda_fp16_autocast") == (256, 256)
    assert choose_batches(args, "siglip", None, None, "cuda_fp16_autocast") == (192, 192)
    path = tmp_path / "source.npz"
    np.savez(path, _meta=np.asarray(json.dumps(dict(model="clip", setting="pretrained",
             precision="cuda_fp16_autocast", batch_size=64, text_batch_size=32))))
    assert choose_batches(args, "clip", path, None, "cuda_fp16_autocast") == (64, 32)
    assert choose_batches(args, "clip", path, (16, 16), "cuda_fp16_autocast") == (16, 16)
    args.batch_size = 8
    assert choose_batches(args, "clip", path, None, "cuda_fp16_autocast") == (8, 8)


def test_same_caption_and_id_cannot_reuse_features_from_another_jpeg(tmp_path):
    from PIL import Image
    from data.config import load_models_cfg
    from data.datasets import get_dataset

    splits = tmp_path / "splits"
    splits.mkdir()
    image_id = "same.jpg"
    split = {"val": [image_id], "captions": {image_id: ["1", "2", "3", "4", "5"]},
             "labels": {image_id: 1}}
    (splits / "coco_split.json").write_text(json.dumps(split), encoding="utf-8")
    datasets = []
    expected = []
    for shade in (0, 255):
        root = tmp_path / str(shade)
        image_dir = root / "coco" / "val2017"
        image_dir.mkdir(parents=True)
        source = image_dir / image_id
        Image.new("RGB", (40, 24), color=(shade, shade, shade)).save(source)
        file_digest = hashlib.sha256(source.read_bytes()).digest()
        expected.append(hashlib.sha256(image_id.encode("utf-8") + b"\x00" + file_digest).hexdigest())
        datasets.append(get_dataset("coco", "val", "clean", "clip", data_root=root, splits_dir=splits))
    wrapper = FakeWrapper()
    wrapper.input_size = load_models_cfg()["clip"]["input_size"]
    def extract(index, path):
        return extract_dataset(wrapper, datasets[index], path, log_dir=tmp_path / "logs", amp=False)

    first = tmp_path / "first.npz"
    second = tmp_path / "second.npz"
    extract(0, first)
    original = first.read_bytes()
    with pytest.raises(FeatureCacheError, match="image_bytes_sha256 mismatch"):
        extract(1, first)
    assert first.read_bytes() == original
    extract(1, second)
    with np.load(first, allow_pickle=False) as black, np.load(second, allow_pickle=False) as white:
        assert [json.loads(a["_meta"].item())["image_bytes_sha256"] for a in (black, white)] == expected
        assert expected[0] != expected[1]
        assert not np.array_equal(black["img_raw"], white["img_raw"])
        legacy = dict(black)
    old_meta = json.loads(legacy["_meta"].item())
    del old_meta["image_bytes_sha256"]
    legacy["_meta"] = np.asarray(json.dumps(old_meta))
    np.savez(first, **legacy)
    with pytest.raises(FeatureCacheError, match="Legacy file lacks image_bytes_sha256"):
        extract(0, first)
