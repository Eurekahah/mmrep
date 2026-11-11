"""P2: shared CLIP/SigLIP2 loader, returning unnormalised features.

The wrapper deliberately preserves autograd. Callers doing feature extraction
must use eval() and no_grad()/inference_mode(); training callers may use .model
to configure the native model's trainable parameters.
"""
from __future__ import annotations

import copy
import re
from pathlib import Path
from typing import Sequence

import torch
from torch import nn
from transformers import AutoTokenizer, CLIPModel, CLIPTokenizer, SiglipModel

from data.config import REPO_ROOT, load_models_cfg

SETTINGS = ("pretrained", "lit", "lora_r8", "lora_r16")
LORA_TARGET_PATTERN = (
    r"(vision_model|text_model)\.encoder\.layers\.\d+\.self_attn\."
    r"(q_proj|k_proj|v_proj|out_proj)"
)


class ModelWrapper(nn.Module):
    """Native model plus the input contract from configs/models.yaml.

    tokenize returns CPU tensors; callers move inputs to wrapper.device.
    encode_* return raw projected embeddings, not unit-length vectors.
    """

    def __init__(self, model, tokenizer, config, *, model_name, setting):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.model_config = copy.deepcopy(config)
        self.model_name = model_name
        self.setting = setting
        self.embed_dim = int(config["embed_dim"])
        self.input_size = int(config["input_size"])
        self.mean = tuple(config["mean"])
        self.std = tuple(config["std"])

    @property
    def device(self):
        return next(self.model.parameters()).device

    def tokenize(self, texts: Sequence[str]):
        if isinstance(texts, str) or not texts:
            raise ValueError("tokenize expects a non-empty sequence of strings")
        if not all(isinstance(text, str) for text in texts):
            raise TypeError("Every caption must be a string")
        cfg = self.model_config["tokenizer"]
        captions = [text.lower() if cfg["lowercase"] else text for text in texts]
        return self.tokenizer(
            captions, padding=cfg["padding"], max_length=int(cfg["max_length"]),
            truncation=True, return_tensors="pt",
        )

    def encode_image(self, pixel_values):
        return self.model.get_image_features(pixel_values=pixel_values)

    def encode_text(self, **tokens):
        return self.model.get_text_features(**tokens)


def _checkpoint_path(ckpt_root, model_name, setting):
    root = Path(ckpt_root)
    if not root.is_absolute():
        root = Path(REPO_ROOT) / root
    folder = root / model_name / setting
    if setting == "lit":
        path = folder / "trainable_state.pt"
        if not path.is_file():
            raise FileNotFoundError(f"LiT checkpoint has not been supplied: {path}")
        return path
    if setting.startswith("lora_"):
        path = folder / "adapter"
        if not (path / "adapter_config.json").is_file():
            raise FileNotFoundError(f"LoRA adapter has not been supplied: {path}")
        if not any((path / filename).is_file() for filename in
                   ("adapter_model.safetensors", "adapter_model.bin")):
            raise FileNotFoundError(f"LoRA adapter weights are missing: {path}")
        return path
    return None


def _validate_config(model, cfg, model_name):
    actual_size = int(model.config.vision_config.image_size)
    if model_name == "clip":
        dims = [int(model.config.projection_dim)]
    else:
        text_cfg = model.config.text_config
        dims = [int(model.config.vision_config.hidden_size),
                int(getattr(text_cfg, "projection_size", None) or text_cfg.hidden_size)]
    if actual_size != int(cfg["input_size"]) or any(d != int(cfg["embed_dim"]) for d in dims):
        raise ValueError(
            f"Loaded {model_name} does not match models.yaml: "
            f"input_size={actual_size}, output_dims={dims}"
        )


def _load_lit(model, path):
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or not state:
        raise ValueError("LiT checkpoint must be a non-empty parameter dictionary")
    expected = {name: value for name, value in model.named_parameters()
                if name.startswith(("text_model.", "text_projection."))}
    actual_keys, expected_keys = set(state), set(expected)
    missing = expected_keys - actual_keys
    extra = actual_keys - expected_keys
    if missing or extra:
        raise ValueError(
            "LiT checkpoint must contain exactly the text-side trainable parameters; "
            f"missing={sorted(missing)[:5]}, unexpected={sorted(extra)[:5]}"
        )
    for name, value in state.items():
        if not isinstance(value, torch.Tensor) or value.shape != expected[name].shape:
            raise ValueError(f"Invalid LiT parameter shape/type: {name}")
        if not value.is_floating_point() or not torch.isfinite(value).all():
            raise ValueError(f"Invalid LiT parameter values: {name}")
    # Frozen vision/logit parameters are intentionally absent, and have already
    # been checked above to prevent strict=False silently hiding a bad checkpoint.
    model.load_state_dict(state, strict=False)


def _load_lora(model, path, setting):
    from peft import PeftConfig, PeftModel, get_peft_model_state_dict
    from peft.tuners.lora import LoraLayer
    from peft.utils.save_and_load import load_peft_weights

    cfg = PeftConfig.from_pretrained(str(path))
    rank = int(setting.removeprefix("lora_r"))
    if str(cfg.peft_type) not in ("LORA", "PeftType.LORA") or cfg.r != rank:
        raise ValueError(f"Expected LoRA rank {rank}, got {cfg.peft_type}, r={getattr(cfg, 'r', None)}")
    if (cfg.bias != "none" or cfg.modules_to_save or getattr(cfg, "use_dora", False)
            or getattr(cfg, "rank_pattern", {}) or getattr(cfg, "alpha_pattern", {})
            or getattr(cfg, "use_rslora", False)):
        raise ValueError("Expected plain fixed-rank LoRA with bias='none' and no extra trainable modules")
    expected_targets = {name for name, module in model.named_modules()
                        if isinstance(module, nn.Linear) and re.fullmatch(LORA_TARGET_PATTERN, name)}
    adapted = PeftModel.from_pretrained(model, str(path), is_trainable=False)
    actual_targets = {name.removeprefix("base_model.model.")
                      for name, module in adapted.named_modules() if isinstance(module, LoraLayer)}
    if actual_targets != expected_targets or not actual_targets:
        raise ValueError("LoRA targets must cover q/k/v/out projections in both encoder towers only")
    disk_state = load_peft_weights(str(path), device="cpu")
    expected_state = get_peft_model_state_dict(adapted)
    if set(disk_state) != set(expected_state):
        raise ValueError("LoRA adapter keys do not match the loaded model")
    for name, value in disk_state.items():
        if value.shape != expected_state[name].shape or not torch.isfinite(value).all():
            raise ValueError(f"Invalid LoRA parameter: {name}")
    return adapted.merge_and_unload(safe_merge=True)


def load_model(model, setting, ckpt_root="ckpt", device="cuda", *,
               cache_dir=None, local_files_only=False, revision=None):
    """Load a configured base model and optional LiT/LoRA checkpoint.

    Parameters stay float32, initially in eval mode. Use autocast in inference.
    A missing/invalid checkpoint raises; it never falls back to pretrained.
    cache_dir, local_files_only and revision are optional HF loading controls.
    """
    configs = load_models_cfg()
    if model not in configs:
        raise ValueError(f"Unknown model {model!r}; expected one of {tuple(configs)}")
    if setting not in SETTINGS:
        raise ValueError(f"Unknown setting {setting!r}; expected one of {SETTINGS}")
    target_device = torch.device(device)
    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; install CUDA PyTorch or pass device='cpu'")
    checkpoint = _checkpoint_path(ckpt_root, model, setting)
    cfg = configs[model]
    common = {"local_files_only": local_files_only}
    if cache_dir is not None:
        common["cache_dir"] = str(cache_dir)
    if revision is not None:
        common["revision"] = revision
    model_class = CLIPModel if model == "clip" else SiglipModel
    tokenizer_class = CLIPTokenizer if model == "clip" else AutoTokenizer
    native = model_class.from_pretrained(cfg["hf_name"], torch_dtype=torch.float32, **common)
    _validate_config(native, cfg, model)
    tokenizer_args = dict(common)
    resolved_revision = getattr(native.config, "_commit_hash", None)
    if resolved_revision:
        tokenizer_args["revision"] = resolved_revision
    tokenizer = tokenizer_class.from_pretrained(cfg["hf_name"], **tokenizer_args)
    if setting == "lit":
        _load_lit(native, checkpoint)
    elif setting.startswith("lora_"):
        native = _load_lora(native, checkpoint, setting)
    wrapper = ModelWrapper(native, tokenizer, cfg, model_name=model, setting=setting)
    wrapper.to(target_device)
    wrapper.eval()
    return wrapper
