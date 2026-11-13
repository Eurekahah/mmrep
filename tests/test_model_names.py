"""Verify all 24 production encoder blocks, without allocating full-size models."""
import re
import pytest
import torch
import yaml
from pathlib import Path
from transformers import CLIPConfig, CLIPModel, SiglipConfig, SiglipModel
from data.config import REPO_ROOT
from models.loader import LORA_TARGET_PATTERN
from train.train_lora import configure_lora


@pytest.mark.parametrize("name", ["clip", "siglip"])
def test_exact_attention_targets(name):
    text = dict(vocab_size=32, hidden_size=16, intermediate_size=32,
                num_hidden_layers=12, num_attention_heads=2)
    vision = dict(hidden_size=16, intermediate_size=32, num_hidden_layers=12,
                  num_attention_heads=2, image_size=16, patch_size=4)
    model = (CLIPModel(CLIPConfig(text_config=text, vision_config=vision, projection_dim=16))
             if name == "clip" else SiglipModel(SiglipConfig(text_config=text, vision_config=vision)))
    targets = [n for n,m in model.named_modules() if re.fullmatch(LORA_TARGET_PATTERN,n)]
    assert len(targets) == 96
    assert sum(n.startswith("vision_model.") for n in targets) == 48
    assert sum(n.startswith("text_model.") for n in targets) == 48
    assert all(isinstance(dict(model.named_modules())[n], torch.nn.Linear) for n in targets)
    assert not any("head" in n for n in targets)
    config = yaml.safe_load((Path(REPO_ROOT)/"configs/lora.yaml").read_text(encoding="utf-8-sig"))
    _, actual = configure_lora(model, 8, config)
    assert set(actual) == set(targets)
