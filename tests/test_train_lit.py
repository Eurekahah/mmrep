"""合成图片 + 小型真实 CLIP/SigLIP 的 CPU 测试，不下载预训练权重。

只缩小网络宽度与层数，保留实际模型结构、项目图像预处理及训练循环。
这些测试不能代替预训练 tokenizer、CUDA 混合精度和真实数据的试跑。
"""
import json
import math
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
import yaml
from transformers import CLIPConfig, CLIPModel, SiglipConfig, SiglipModel

from data.config import REPO_ROOT, load_models_cfg
from data.datasets import get_dataset
from tests.test_data import mini
from train.losses import info_nce, sigmoid_loss
from train.train_lit import configure_lit, run_training


@pytest.fixture(params=["clip", "siglip"])
def tiny_model(request):
    """直接用配置创建随机模型；不调用 from_pretrained，不访问网络。"""
    name = request.param
    config = load_models_cfg()[name]
    torch.manual_seed(42)
    text = dict(vocab_size=32, hidden_size=16, intermediate_size=32,
                num_hidden_layers=1, num_attention_heads=2,
                max_position_embeddings=config["tokenizer"]["max_length"],
                bos_token_id=1, eos_token_id=31, pad_token_id=0)
    vision = dict(hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                  num_attention_heads=2, image_size=config["input_size"],
                  patch_size=config["patch_size"])
    if name == "clip":
        model = CLIPModel(CLIPConfig(text_config=text, vision_config=vision,
                                    projection_dim=16))
    else:
        model = SiglipModel(SiglipConfig(text_config=text, vision_config=vision))
    return name, model


class DummyTokenizer:
    """仅提供合法、随文本变化的 token；真实分词器下载留给真实数据试跑。"""
    def __call__(self, captions, *, padding, max_length, truncation, return_tensors):
        assert padding == "max_length" and truncation and return_tensors == "pt"
        rows = []
        for caption in captions:
            tokens = [1] + [3 + ord(char) % 28 for char in caption[:max_length - 2]] + [31]
            rows.append(tokens + [0] * (max_length - len(tokens)))
        return {"input_ids": torch.tensor(rows, dtype=torch.long)}


def test_lit_freeze_groups(tiny_model):
    name, model = tiny_model
    trainable = configure_lit(model)
    # 显式列出两个模型各自的顶层分组，防止漏冻投影、温度或偏置。
    train_groups = {"text_model", "text_projection"} if name == "clip" else {"text_model"}
    frozen_groups = ({"vision_model", "visual_projection", "logit_scale"} if name == "clip"
                     else {"vision_model", "logit_scale", "logit_bias"})
    actual_train, actual_frozen = set(), set()
    for key, parameter in model.named_parameters():
        (actual_train if parameter.requires_grad else actual_frozen).add(key.split(".")[0])
        assert (key in trainable) == parameter.requires_grad
    assert actual_train == train_groups
    assert actual_frozen == frozen_groups
    assert model.text_model.training and not model.vision_model.training
    if name == "siglip":
        assert all(p.requires_grad for p in model.text_model.head.parameters())
        assert all(not p.requires_grad for p in model.vision_model.head.parameters())


def test_loss_matches_model_forward(tiny_model):
    """以真实模型的内置损失为参照，检查两个独立损失函数的数值。"""
    name, model = tiny_model
    model.eval()
    size = load_models_cfg()[name]["input_size"]
    pixels = torch.randn(3, 3, size, size)
    tokens = DummyTokenizer()(["a cat", "a dog", "a bird"],
                              **{key: load_models_cfg()[name]["tokenizer"][key]
                                 for key in ("padding", "max_length")},
                              truncation=True, return_tensors="pt")
    with torch.no_grad():
        expected = model(pixel_values=pixels, **tokens, return_loss=True).loss
        img = F.normalize(model.get_image_features(pixels), dim=-1)
        txt = F.normalize(model.get_text_features(**tokens), dim=-1)
        actual = (info_nce(img, txt, model.logit_scale) if name == "clip" else
                  sigmoid_loss(img, txt, model.logit_scale, model.logit_bias))
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


def test_training_smoke(tiny_model, mini, tmp_path, monkeypatch, capsys):
    name, model = tiny_model
    root, splits, _ = mini
    dataset = get_dataset("flickr", "train", "clean", name,
                          data_root=root, splits_dir=splits)
    with open(Path(REPO_ROOT) / "configs/lit.yaml", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    # 复用 P1 的两张合成训练图；跑两轮保证越过 warmup 首步的零学习率。
    config.update(epochs=2, batch_size=2)
    initial = {key: value.detach().clone() for key, value in model.named_parameters()}
    pixels = torch.stack([dataset[i]["image"] for i in range(len(dataset))])
    model.eval()
    with torch.no_grad():
        images_before = model.get_image_features(pixels)

    # 记录图像前向的梯度状态，直接检查训练循环确实使用了 no_grad。
    image_grad_enabled = []
    encode_image = model.get_image_features

    def record_image_forward(*args, **kwargs):
        image_grad_enabled.append(torch.is_grad_enabled())
        return encode_image(*args, **kwargs)

    monkeypatch.setattr(model, "get_image_features", record_image_forward)
    output_dir, log_dir = tmp_path / "ckpt", tmp_path / "logs"
    meta = run_training(name, model, DummyTokenizer(), dataset, config,
                        load_models_cfg()[name], device="cpu", output_dir=output_dir,
                        log_dir=log_dir)
    assert image_grad_enabled == [False, False]

    expected_names = {key for key in initial
                      if key.startswith(("text_model.", "text_projection."))}
    for key, parameter in model.named_parameters():
        if key not in expected_names:
            assert parameter.grad is None
            torch.testing.assert_close(parameter, initial[key], atol=0, rtol=0)
    assert any(not torch.equal(parameter, initial[key])
               for key, parameter in model.named_parameters() if key in expected_names)
    model.eval()
    with torch.no_grad():
        torch.testing.assert_close(model.get_image_features(pixels), images_before, atol=0, rtol=0)

    # checkpoint 必须仅含文本参数，且可与原始模型状态组合恢复训练后的全部权重。
    state = torch.load(output_dir / "trainable_state.pt", map_location="cpu", weights_only=True)
    assert set(state) == expected_names
    final = {key: value.detach().clone() for key, value in model.named_parameters()}
    model.load_state_dict(initial, strict=False)
    model.load_state_dict(state, strict=False)
    for key, parameter in model.named_parameters():
        torch.testing.assert_close(parameter, final[key], atol=0, rtol=0)

    assert json.loads((output_dir / "meta.json").read_text(encoding="utf-8")) == meta
    assert set(meta["trainable_names"]) == expected_names
    count = sum(initial[key].numel() for key in expected_names)
    total = sum(value.numel() for value in initial.values())
    assert meta["trainable_params"] == count and meta["total_params"] == total
    assert meta["trainable_ratio"] == pytest.approx(count / total)
    assert len(meta["epoch_loss"]) == len(meta["epoch_seconds"]) == config["epochs"]
    assert all(math.isfinite(loss) for loss in meta["epoch_loss"])
    assert all(seconds >= 0 for seconds in meta["epoch_seconds"])
    assert meta["epoch_samples"] == [len(dataset)] * config["epochs"]
    assert meta["precision"] == "fp32" and meta["final_lr"] == 0.0
    printed = capsys.readouterr().out
    assert all(key in printed for key in expected_names)
    log = (log_dir / f"lit_{name}.log").read_text(encoding="utf-8")
    assert "epoch 1/2" in log and "epoch 2/2" in log
