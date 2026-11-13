"""P4: two-tower LoRA training and checkpoint handoff validation."""
import argparse
import gc
import json
import math
import re
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from peft import LoraConfig, get_peft_model
from peft.tuners.lora import LoraLayer
from torch.utils.data import DataLoader, Subset
from transformers import get_cosine_schedule_with_warmup

from data.config import REPO_ROOT
from data.datasets import collate_fn, get_dataset
from models.loader import LORA_TARGET_PATTERN, load_model
from tools.logger import RunLogger
from train.losses import info_nce, sigmoid_loss
from train.train_lit import sample_captions, set_seed


def configure_lora(base, rank, config):
    if rank not in (8, 16):
        raise ValueError("rank must be 8 or 16")
    if config["target_modules_regex"] != LORA_TARGET_PATTERN:
        raise ValueError("Use the exact encoder-only target pattern from the contract")
    targets = [n for n, m in base.named_modules()
               if re.fullmatch(LORA_TARGET_PATTERN, n) and isinstance(m, torch.nn.Linear)]
    expected = 4 * (base.config.vision_config.num_hidden_layers
                    + base.config.text_config.num_hidden_layers)
    if len(targets) != expected or not all(any(n.startswith(t) for n in targets)
                                         for t in ("vision_model.", "text_model.")):
        raise ValueError(f"Unexpected attention targets: {len(targets)} != {expected}")
    base.requires_grad_(False)
    adapted = get_peft_model(base, LoraConfig(
        r=rank, lora_alpha=2 * rank, lora_dropout=0.05,
        target_modules=LORA_TARGET_PATTERN, bias="none"))
    actual = [n.removeprefix("base_model.model.") for n, m in adapted.named_modules()
              if isinstance(m, LoraLayer)]
    if set(actual) != set(targets):
        raise AssertionError("Adapter placement differs from encoder targets")
    for name, param in adapted.named_parameters():
        if param.requires_grad:
            assert "lora_A." in name or "lora_B." in name, name
            assert param.dtype == torch.float32, name
    adapted.print_trainable_parameters()
    return adapted, targets


def encode(wrapper, dataset, device, batch_size=16):
    images, texts = [], []
    wrapper.eval()
    with torch.no_grad():
        for batch in DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn):
            tokens = {k: v.to(device) for k, v in
                      wrapper.tokenize([c[0] for c in batch["captions"]]).items()}
            # Acceptance comparisons use full fp32, with identical batching.
            images.append(wrapper.encode_image(batch["image"].to(device)).float().cpu())
            texts.append(wrapper.encode_text(**tokens).float().cpu())
    return torch.cat(images), torch.cat(texts)


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def run_training(wrapper, dataset, rank, config, output, *, device,
                 gradient_checkpointing=False, workers=4, log_dir=None):
    if (config["lr_schedule"] != "cosine" or config["precision"] != "fp16"
            or not config["freeze_logit_scale_bias"]
            or config["alpha_factor"] != 2 or config["dropout"] != 0.05
            or config["bias"] != "none" or config["train_split"] != ["flickr", "train"]):
        raise ValueError("Configuration violates P4 training contract")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing run: {output}")
    output.mkdir(parents=True, exist_ok=True)
    set_seed(config["seed"])
    device = torch.device(device)
    if device.type == "cuda" and device.index is not None:
        torch.cuda.set_device(device)
    wrapper.to(device=device, dtype=torch.float32)
    wrapper.model, targets = configure_lora(wrapper.model, rank, config)
    if gradient_checkpointing:
        wrapper.model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
    parameters = {n: p for n, p in wrapper.model.named_parameters() if p.requires_grad}
    (output / "trainable_names.txt").write_text("\n".join(parameters) + "\n")
    (output / "target_modules.txt").write_text("\n".join(targets) + "\n")
    trainable = sum(p.numel() for p in parameters.values())
    total = sum(p.numel() for p in wrapper.model.parameters())
    amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    optimizer = torch.optim.AdamW(parameters.values(), lr=config["lr"],
                                  weight_decay=config["weight_decay"])
    loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True,
                        collate_fn=collate_fn, num_workers=workers, pin_memory=amp,
                        generator=torch.Generator().manual_seed(config["seed"]))
    steps = len(loader) * config["epochs"]
    warmup = math.ceil(steps * config["warmup_ratio"])
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup, steps)
    captions_rng = torch.Generator().manual_seed(config["seed"])
    history, skipped, updates = [], 0, 0
    with RunLogger("lora", model=wrapper.model_name, setting=f"lora_r{rank}",
                   seed=config["seed"], log_dir=log_dir,
                   num_samples=len(dataset), epochs=config["epochs"], rank=rank) as log:
        start = time.perf_counter()
        for epoch in range(config["epochs"]):
            wrapper.train()
            loss_sum, count = 0., 0
            for step, batch in enumerate(loader):
                tokens = {k: v.to(device) for k, v in wrapper.tokenize(
                    sample_captions(batch["captions"], captions_rng)).items()}
                pixels = batch["image"].to(device, non_blocking=amp)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
                    image = wrapper.encode_image(pixels)
                    text = wrapper.encode_text(**tokens)
                image, text = F.normalize(image.float(), dim=-1), F.normalize(text.float(), dim=-1)
                base = wrapper.model.get_base_model()
                loss = (info_nce(image, text, base.logit_scale) if wrapper.model_name == "clip"
                        else sigmoid_loss(image, text, base.logit_scale, base.logit_bias))
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite loss: epoch={epoch}, step={step}")
                scaler.scale(loss).backward()
                previous = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if scaler.get_scale() >= previous:
                    scheduler.step()
                    updates += 1
                else:
                    skipped += 1
                loss_sum += loss.item() * len(pixels)
                count += len(pixels)
                if step % 10 == 0:
                    log.info(f"epoch={epoch+1} step={step+1}/{len(loader)} loss={loss.item():.6f}")
            if amp:
                torch.cuda.synchronize(device)
            history.append(loss_sum / count)
            log.info(f"epoch={epoch+1} mean_loss={history[-1]:.6f} samples={count}")
        wrapper.model.save_pretrained(output / "adapter", safe_serialization=True)
        meta = {**config, "model": wrapper.model_name, "rank": rank,
                "setting": f"lora_r{rank}", "trainable_params": trainable,
                "total_params": total, "ratio": trainable / total,
                "target_count": len(targets), "num_samples": len(dataset),
                "gradient_checkpointing": gradient_checkpointing,
                "epoch_loss": history, "wall_sec": time.perf_counter()-start,
                "gpu_mem_peak_MB": torch.cuda.max_memory_allocated(device)/2**20 if amp else 0,
                "warmup_steps": warmup, "optimizer_updates": updates,
                "skipped_updates": skipped, "final_lr": scheduler.get_last_lr()[0],
                "base_model": wrapper.model.get_base_model().config._name_or_path,
                "base_revision": getattr(wrapper.model.get_base_model().config, "_commit_hash", None),
                "versions": {"torch": torch.__version__, "transformers": __import__('transformers').__version__,
                             "peft": __import__('peft').__version__}}
        write_json(output / "meta.json", meta)
        log.metric("epoch_loss", history)
        log.metric("optimizer_updates", updates)
        log.metric("skipped_updates", skipped)
    return meta


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("clip", "siglip"), required=True)
    parser.add_argument("--rank", type=int, choices=(8, 16), required=True)
    parser.add_argument("--config", default="configs/lora.yaml")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--batch-size", type=int, choices=(64, 128))
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--ckpt-root", default="ckpt")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--skip-selfcheck", action="store_true", help="Smoke tests only")
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)
    for key in ("batch_size", "epochs"):
        if getattr(args, key) is not None:
            config[key] = getattr(args, key)
    if args.num_workers < 0 or config["epochs"] < 1:
        parser.error("Invalid workers or epoch count")
    if args.max_samples is not None and args.max_samples < 2:
        parser.error("max-samples must be >= 2")
    set_seed(config["seed"])
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = args.device
    output = Path(args.ckpt_root) / args.model / f"lora_r{args.rank}"
    dataset = get_dataset("flickr", "train", "clean", args.model)
    if args.max_samples:
        dataset = Subset(dataset, range(min(args.max_samples, len(dataset))))
    wrapper = load_model(args.model, "pretrained", device=device, local_files_only=True)
    if not args.skip_selfcheck:
        val = Subset(get_dataset("flickr", "val", "clean", args.model), range(500))
        before = encode(wrapper, val, device)
    meta = run_training(wrapper, dataset, args.rank, config, output, device=device,
                        gradient_checkpointing=args.gradient_checkpointing, workers=args.num_workers)
    if args.skip_selfcheck:
        return
    trained = encode(wrapper, val, device)
    # Drop the in-memory adapters before loading via P2, preventing GPU duplication.
    del wrapper
    gc.collect()
    torch.cuda.empty_cache()
    merged = load_model(args.model, f"lora_r{args.rank}", ckpt_root=args.ckpt_root,
                        device=device, local_files_only=True)
    reload_features = encode(merged, Subset(val, range(32)), device)
    differences = {kind: float((reference[:32] - restored).abs().max())
                   for kind, reference, restored in zip(("image", "text"), trained, reload_features)}
    similarities = {kind: F.cosine_similarity(a, b).numpy()
                    for kind, a, b in zip(("image", "text"), before, trained)}
    np.savez(output / "selfcheck_cosines.npz", **similarities)
    check = {"val_images": 500, "captions": "first caption per image",
             "merge_images": 32, "merge_max_abs_diff": differences,
             "merge_pass": all(v < 1e-4 for v in differences.values()),
             "p2_loader_pass": True,
             "cosine": {k: {"mean": float(v.mean()), "std": float(v.std()),
                            "min": float(v.min()), "max": float(v.max())}
                        for k, v in similarities.items()}}
    write_json(output / "selfcheck.json", check)
    if not check["merge_pass"]:
        raise AssertionError(f"Merge mismatch: {differences}")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(range(1, len(meta["epoch_loss"])+1), meta["epoch_loss"], marker="o")
    ax.set(xlabel="Epoch", ylabel="Training loss", title=f"{args.model} LoRA r={args.rank}")
    fig.tight_layout()
    fig.savefig(output / "loss.png", dpi=180)
    plt.close(fig)
    shutil.copyfile(Path(REPO_ROOT) / "logs/runs.jsonl", Path(REPO_ROOT) / "logs/P4_runs.jsonl")
    print(json.dumps(check, indent=2), flush=True)


if __name__ == "__main__":
    main()
