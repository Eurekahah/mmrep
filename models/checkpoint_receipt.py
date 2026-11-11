"""Validate upstream training metadata before a formal P2 extraction."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from data.config import REPO_ROOT, load_models_cfg


def _number(meta, aliases, *, integer=False, minimum=0):
    for name in aliases:
        if name in meta:
            value = meta[name]
            if (isinstance(value, bool) or not isinstance(value, (float, int))
                    or not math.isfinite(value) or value < minimum
                    or (integer and value != int(value))):
                raise ValueError(f"Invalid training metadata field {name}={value!r}")
            return int(value) if integer else value
    raise ValueError(f"Training metadata is missing {aliases[0]}")


def read_training_receipt(ckpt_root, model, setting):
    """Check received files and the fields agreed in CONTRACT 3.8.

    The model loader subsequently validates the actual tensors and adapter targets.
    """
    if setting == "pretrained":
        return None
    if setting not in ("lit", "lora_r8", "lora_r16"):
        raise ValueError(f"Unknown checkpoint setting: {setting}")
    root = Path(ckpt_root)
    if not root.is_absolute():
        root = Path(REPO_ROOT) / root
    folder = root / model / setting
    source = folder / "meta.json"
    if not source.is_file():
        raise FileNotFoundError(f"Awaiting training metadata: {source}")
    raw = source.read_bytes()
    meta = json.loads(raw.decode("utf-8-sig"))
    if not isinstance(meta, dict):
        raise ValueError(f"Expected a JSON object in {source}")
    hf_name = load_models_cfg()[model]["hf_name"]
    if meta.get("model", model) not in (model, hf_name) or meta.get("hf_name", hf_name) != hf_name:
        raise ValueError(f"Model identity differs from {model} in {source}")
    accepted = (setting, "lora") if setting.startswith("lora_") else (setting,)
    if meta.get("setting", setting) not in accepted:
        raise ValueError(f"Training setting differs from {setting} in {source}")
    receipt = {"model": model, "setting": setting,
               "meta_sha256": hashlib.sha256(raw).hexdigest(),
               "lr": _number(meta, ("lr", "learning_rate")),
               "epochs": _number(meta, ("epochs", "epoch"), integer=True, minimum=1),
               "batch_size": _number(meta, ("batch_size", "batch"), integer=True, minimum=1),
               "seed": _number(meta, ("seed",), integer=True),
               "trainable_params": _number(meta, ("trainable_params", "num_trainable_params"),
                                            integer=True, minimum=1),
               "wall_sec": _number(meta, ("wall_sec", "total_seconds", "elapsed_seconds")),
               "gpu_mem_peak_MB": _number(meta, ("gpu_mem_peak_MB", "peak_memory_mb", "peak_gpu_memory_mb"))}
    losses = meta.get("epoch_loss", meta.get("epoch_losses"))
    if (not isinstance(losses, list) or len(losses) != receipt["epochs"]
            or not all(isinstance(v, (float, int)) and not isinstance(v, bool)
                       and math.isfinite(v) for v in losses)):
        raise ValueError(f"Expected one finite loss per epoch in {source}")
    if "num_samples" in meta or "epoch_samples" in meta:
        split = Path(REPO_ROOT) / "splits" / "flickr_split.json"
        expected = len(json.loads(split.read_text(encoding="utf-8"))["train"])
        if "num_samples" in meta and meta["num_samples"] != expected:
            raise ValueError(f"{source}: received {meta['num_samples']} training samples, expected {expected}")
        if "epoch_samples" in meta and meta["epoch_samples"] != [expected] * receipt["epochs"]:
            raise ValueError(f"{source}: epoch sample counts differ from the official training split")
        receipt["expected_training_samples"] = expected
    if setting == "lit":
        weights = folder / "trainable_state.pt"
    else:
        rank = int(setting.removeprefix("lora_r"))
        adapter = folder / "adapter"
        cfg = json.loads((adapter / "adapter_config.json").read_text(encoding="utf-8-sig"))
        if cfg.get("r") != rank or meta.get("rank", meta.get("r", rank)) != rank:
            raise ValueError(f"Adapter rank does not match {setting}")
        weights = adapter / "adapter_model.safetensors"
        if not weights.exists():
            weights = adapter / "adapter_model.bin"
    if not weights.is_file() or weights.stat().st_size == 0:
        raise FileNotFoundError(f"Awaiting checkpoint weights: {weights}")
    return receipt
