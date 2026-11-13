"""Offline P4 tests: actual architectures, training, freeze, save/reload/merge."""
from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch
import yaml

from data.config import REPO_ROOT
from models.loader import load_model
from tests.test_model_loader import offline_model
from train.train_lora import run_training


@pytest.mark.parametrize("rank", [8, 16])
@pytest.mark.parametrize("checkpointing", [False, True])
def test_train_and_handoff(offline_model, rank, checkpointing, tmp_path):
    name, base, _, cfg, _ = offline_model
    torch.set_num_threads(1)
    wrapper = load_model(name, "pretrained", device="cpu")
    original = deepcopy(base.state_dict())
    dataset = [{"image": torch.randn(3, 16, 16), "captions": [f"sample {i}"]*5,
                "idx": i, "id": str(i), "label": -1} for i in range(4)]
    config = yaml.safe_load((Path(REPO_ROOT)/"configs/lora.yaml").read_text(encoding="utf-8-sig"))
    config.update(epochs=2, batch_size=2)
    output = tmp_path / name / f"lora_r{rank}"
    meta = run_training(wrapper, dataset, rank, config, output, device="cpu",
                        gradient_checkpointing=checkpointing, workers=0,
                        log_dir=tmp_path/"logs")
    assert meta["optimizer_updates"] == 4
    assert meta["final_lr"] == 0
    assert meta["target_count"] == 8
    assert all(torch.isfinite(torch.tensor(meta["epoch_loss"])))
    assert json.loads((output/"meta.json").read_text()) == meta
    adapted = wrapper.model.get_base_model()
    for key, value in adapted.state_dict().items():
        if "lora_" not in key:
            torch.testing.assert_close(value, original[key.replace(".base_layer.", ".")], atol=0, rtol=0)
    for tower in ("vision_model", "text_model"):
        assert any(torch.count_nonzero(p) for n,p in adapted.named_parameters()
                   if n.startswith(tower) and "lora_B" in n)
    pixels = torch.stack([d["image"] for d in dataset])
    tokens = wrapper.tokenize([d["captions"][0] for d in dataset])
    wrapper.eval()
    with torch.no_grad():
        before = wrapper.encode_image(pixels), wrapper.encode_text(**tokens)
    restored = load_model(name, f"lora_r{rank}", ckpt_root=tmp_path, device="cpu")
    with torch.no_grad():
        after = restored.encode_image(pixels), restored.encode_text(**tokens)
    for a,b in zip(before,after):
        assert (a-b).abs().max() < 1e-4
    with pytest.raises(FileExistsError):
        run_training(wrapper, dataset, rank, config, output, device="cpu", workers=0)
