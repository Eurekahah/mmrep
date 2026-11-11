"""Offline loader-contract tests with small real Transformers architectures."""
from copy import deepcopy
from pathlib import Path

import pytest
import torch
from transformers import CLIPConfig, CLIPModel, SiglipConfig, SiglipModel

from data.config import load_models_cfg


class RecordingTokenizer:
    def __init__(self):
        self.last_captions = None
        self.last_options = None

    def __call__(self, captions, *, padding, max_length, truncation, return_tensors):
        self.last_captions = list(captions)
        self.last_options = dict(padding=padding, max_length=max_length,
                                 truncation=truncation, return_tensors=return_tensors)
        rows = []
        for caption in captions:
            tokens = [1] + [3 + ord(c) % 28 for c in caption[:max_length - 2]] + [31]
            rows.append(tokens + [0] * (max_length - len(tokens)))
        ids = torch.tensor(rows, dtype=torch.long)
        return {"input_ids": ids, "attention_mask": ids.ne(0).long()}


@pytest.fixture(params=["clip", "siglip"])
def offline_model(request, monkeypatch):
    from models import loader

    name = request.param
    configs = deepcopy(load_models_cfg())
    configs[name].update(embed_dim=16, input_size=16, patch_size=4)
    configs[name]["tokenizer"]["max_length"] = 8
    config = configs[name]
    text = dict(vocab_size=32, hidden_size=16, intermediate_size=32,
                num_hidden_layers=1, num_attention_heads=2,
                max_position_embeddings=8, bos_token_id=1, eos_token_id=31,
                pad_token_id=0)
    vision = dict(hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                  num_attention_heads=2, image_size=16, patch_size=4)
    torch.manual_seed(42)
    base = (CLIPModel(CLIPConfig(text_config=text, vision_config=vision, projection_dim=16))
            if name == "clip" else SiglipModel(SiglipConfig(text_config=text, vision_config=vision)))
    tokenizer = RecordingTokenizer()
    calls = []

    def from_pretrained(identifier, **kwargs):
        calls.append((identifier, kwargs))
        return deepcopy(base)

    monkeypatch.setattr(loader, "load_models_cfg", lambda: configs)
    model_class = CLIPModel if name == "clip" else SiglipModel
    monkeypatch.setattr(model_class, "from_pretrained", from_pretrained)
    # Patch both supported tokenizer entrypoints without loading files or contacting HF.
    import transformers
    monkeypatch.setattr(transformers.CLIPTokenizer, "from_pretrained", lambda *a, **kw: tokenizer)
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **kw: tokenizer)
    return name, base, tokenizer, config, calls


def test_pretrained_raw_outputs_and_gradients(offline_model):
    from models.loader import load_model

    name, _, tokenizer, cfg, calls = offline_model
    wrapper = load_model(name, "pretrained", device="cpu", local_files_only=True,
                         cache_dir="test-cache", revision="test-revision")
    assert wrapper.embed_dim == 16 and wrapper.input_size == 16
    assert list(wrapper.mean) == cfg["mean"] and list(wrapper.std) == cfg["std"]
    assert not wrapper.training and not wrapper.model.training
    assert next(wrapper.parameters()).dtype == torch.float32
    assert calls[0][0] == cfg["hf_name"]
    assert calls[0][1]["local_files_only"] is True
    assert calls[0][1]["revision"] == "test-revision"
    tokens = wrapper.tokenize(["UPPER case", "Other sample"])
    assert tokens["input_ids"].shape == (2, 8)
    expected = ["upper case", "other sample"] if cfg["tokenizer"]["lowercase"] else ["UPPER case", "Other sample"]
    assert tokenizer.last_captions == expected
    assert tokenizer.last_options == dict(padding="max_length", max_length=8,
                                         truncation=True, return_tensors="pt")
    pixels = torch.randn(2, 3, 16, 16)
    image = wrapper.encode_image(pixels)
    text = wrapper.encode_text(**tokens)
    torch.testing.assert_close(image, wrapper.model.get_image_features(pixel_values=pixels))
    torch.testing.assert_close(text, wrapper.model.get_text_features(**tokens))
    assert image.shape == text.shape == (2, 16)
    assert image.requires_grad and text.requires_grad
    (image.square().mean() + text.square().mean()).backward()
    assert any(p.grad is not None for p in wrapper.model.vision_model.parameters())
    assert any(p.grad is not None for p in wrapper.model.text_model.parameters())


def _save_lit(root, name, state):
    path = Path(root) / name / "lit" / "trainable_state.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def test_lit_loads_only_text_and_preserves_image(offline_model, tmp_path):
    from models.loader import load_model

    name, base, _, _, _ = offline_model
    initial = base.state_dict()
    state = {key: value.clone() + 0.01 for key, value in initial.items()
             if key.startswith(("text_model.", "text_projection."))}
    _save_lit(tmp_path, name, state)
    wrapper = load_model(name, "lit", ckpt_root=tmp_path, device="cpu")
    loaded = wrapper.model.state_dict()
    assert set(loaded) == set(initial)
    for key in loaded:
        torch.testing.assert_close(loaded[key], state.get(key, initial[key]), atol=0, rtol=0)


@pytest.mark.parametrize("bad_kind", ["missing_text", "unexpected", "image_parameter"])
def test_lit_rejects_incomplete_or_foreign_checkpoint(offline_model, tmp_path, bad_kind):
    from models.loader import load_model

    name, base, _, _, _ = offline_model
    full = base.state_dict()
    state = {key: value.clone() for key, value in full.items()
             if key.startswith(("text_model.", "text_projection."))}
    if bad_kind == "missing_text":
        state.pop(next(iter(state)))
    elif bad_kind == "unexpected":
        state["unrecognized.weight"] = torch.zeros(1)
    else:
        key = next(key for key in full if key.startswith("vision_model."))
        state[key] = full[key]
    _save_lit(tmp_path, name, state)
    with pytest.raises((ValueError, RuntimeError)):
        load_model(name, "lit", ckpt_root=tmp_path, device="cpu")


def test_missing_checkpoint_fails(offline_model, tmp_path):
    from models.loader import load_model

    name = offline_model[0]
    with pytest.raises((FileNotFoundError, ValueError)):
        load_model(name, "lit", ckpt_root=tmp_path, device="cpu")
    with pytest.raises((FileNotFoundError, ValueError)):
        load_model(name, "lora_r8", ckpt_root=tmp_path, device="cpu")


def test_lora_adapter_reload_merge_matches(offline_model, tmp_path):
    from peft import LoraConfig, get_peft_model
    from models.loader import load_model

    name, base, _, _, _ = offline_model
    adapter_model = get_peft_model(deepcopy(base), LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.05, bias="none",
        target_modules=r"(vision_model|text_model)\.encoder\.layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj|out_proj)"))
    with torch.no_grad():
        for key, parameter in adapter_model.named_parameters():
            if "lora_B" in key:
                parameter.fill_(0.01)
    adapter_model.eval()
    target = tmp_path / name / "lora_r8" / "adapter"
    adapter_model.save_pretrained(target, safe_serialization=True)
    wrapper = load_model(name, "lora_r8", ckpt_root=tmp_path, device="cpu")
    assert not any("lora_" in key for key, _ in wrapper.model.named_parameters())
    pixels = torch.randn(2, 3, 16, 16)
    tokens = wrapper.tokenize(["cat", "dog"])
    with torch.no_grad():
        torch.testing.assert_close(wrapper.encode_image(pixels),
                                   adapter_model.get_image_features(pixel_values=pixels), atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(wrapper.encode_text(**tokens),
                                   adapter_model.get_text_features(**tokens), atol=1e-5, rtol=1e-5)


def test_invalid_names_fail_before_network():
    from models.loader import load_model

    with pytest.raises(ValueError):
        load_model("unknown", "pretrained", device="cpu")
    with pytest.raises(ValueError):
        load_model("clip", "unknown", device="cpu")


def test_real_input_cli_uses_only_first_two_local_images(offline_model, tmp_path, monkeypatch):
    """Exercise the P1 COCO path with tiny offline weights and two fixture JPEGs."""
    import json
    import numpy as np
    from PIL import Image
    from data import datasets, preprocess
    from tests import test_extract_one_batch as smoke

    name, _, _, config, _ = offline_model
    for module in (datasets, preprocess, smoke):
        monkeypatch.setattr(module, "load_models_cfg", lambda: {name: config})
    ids = ["first.jpg", "second.jpg", "intentionally_absent.jpg"]
    raw, splits = tmp_path / "raw", tmp_path / "splits"
    folder = raw / "coco" / "val2017"
    folder.mkdir(parents=True)
    splits.mkdir()
    for color, file in enumerate(ids[:2]):
        Image.new("RGB", (19, 25), color=(67 + color, 128, 202)).save(folder / file)
    split = {"val": ids, "captions": {id_: [f"{j} image {i}" for j in range(5)] for i, id_ in enumerate(ids)},
             "labels": {id_: i + 4 for i, id_ in enumerate(ids)}}
    (splits / "coco_split.json").write_text(json.dumps(split), encoding="utf-8")
    result = smoke.run_one_batch(name, device="cpu", dataset_name="coco", batch_size=2,
                                 data_root=raw, splits_dir=splits, out_dir=tmp_path / "smoke",
                                 local_files_only=True)
    assert result["status"] == "passed" and result["img_ids"] == ids[:2]
    with np.load(result["exports"]["clean"]["path"]) as output:
        assert output["img_raw"].shape == (2, 16)
        assert output["txt_raw"].shape == (10, 16)
        np.testing.assert_array_equal(output["labels"], [4, 5])
