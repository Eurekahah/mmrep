"""LiT 微调：锁定图像分支，仅训练文本分支。

在仓库根目录运行：
    python -m train.train_lit --model clip --config configs/lit.yaml
    python -m train.train_lit --model siglip --config configs/lit.yaml
"""
import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Subset
from transformers import (
    AutoTokenizer,
    CLIPModel,
    CLIPTokenizer,
    SiglipModel,
    get_cosine_schedule_with_warmup,
)

from data.config import REPO_ROOT, load_models_cfg
from data.datasets import collate_fn, get_dataset
from tools.logger import RunLogger
from train.losses import info_nce, sigmoid_loss


def set_seed(seed):
    """固定模型、数据顺序和其他随机操作的种子。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def configure_lit(model):
    """先冻结全部参数，再只开放文本塔及 CLIP 的文本投影。"""
    model.requires_grad_(False)
    model.text_model.requires_grad_(True)
    # SigLIP 的文本投影 head 已经包含在 text_model 内，无独立 text_projection。
    if hasattr(model, "text_projection"):
        model.text_projection.requires_grad_(True)
    # 全模型保持评估模式，仅将文本塔切为训练模式，图像塔始终保持锁定。
    model.eval()
    model.text_model.train()
    if hasattr(model, "text_projection"):
        model.text_projection.train()
    return {name: param for name, param in model.named_parameters() if param.requires_grad}


def sample_captions(captions, generator):
    """每张图随机选一条文本；生成器跨 batch、跨 epoch 持续推进。"""
    return [items[torch.randint(len(items), (1,), generator=generator).item()]
            for items in captions]


def tokenize_captions(tokenizer, captions, config):
    """从 models.yaml 读取大小写、填充和截断规则。"""
    if config["lowercase"]:
        captions = [caption.lower() for caption in captions]
    return tokenizer(captions, padding=config["padding"],
                     max_length=config["max_length"], truncation=True,
                     return_tensors="pt")


def run_training(model_name, model, tokenizer, dataset, config, model_config, *,
                 device, output_dir, log_dir=None, num_workers=0,
                 gradient_checkpointing=False):
    """执行训练，并保存最后一轮的文本参数及可复现所需的元数据。"""
    # 当前脚本只实现 CONTRACT 3.5 的 LiT 设置，避免配置改动被静默忽略。
    if config["lr_schedule"] != "cosine" or config["precision"] != "fp16":
        raise ValueError("LiT 配置要求 lr_schedule=cosine、precision=fp16。")
    if not config["freeze_logit_scale_bias"]:
        raise ValueError("LiT 必须冻结预训练的 logit_scale 和 logit_bias。")
    if (set(config["trainable"]) != {"text_model", "text_projection"}
            or set(config["frozen"]) != {"vision_model", "visual_projection"}):
        raise ValueError("LiT 的可训练与冻结模块必须符合 CONTRACT 3.5。")
    if list(config["train_split"]) != ["flickr", "train"]:
        raise ValueError("LiT 只能使用 Flickr 的 train 划分。")
    epochs, batch_size = int(config["epochs"]), int(config["batch_size"])
    if epochs <= 0 or batch_size < 2 or len(dataset) < 2:
        raise ValueError("epochs 必须为正，batch_size 和训练样本数必须至少为 2。")
    if not 0 <= float(config["warmup_ratio"]) < 1:
        raise ValueError("warmup_ratio 必须在 [0, 1) 范围内。")

    set_seed(int(config["seed"]))
    device = torch.device(device)
    if device.type == "cuda" and device.index is not None:
        torch.cuda.set_device(device)
    # 参数保持 fp32，只在前向时使用 autocast，不对整个模型调用 half()。
    model = model.to(device=device, dtype=torch.float32)
    trainable = configure_lit(model)
    if gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # shuffle 与 caption 抽样使用独立生成器；num_workers 不影响 caption 的抽样。
    shuffle_generator = torch.Generator().manual_seed(int(config["seed"]))
    caption_generator = torch.Generator().manual_seed(int(config["seed"]))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        collate_fn=collate_fn, num_workers=num_workers,
                        pin_memory=use_amp, generator=shuffle_generator,
                        drop_last=False)
    # 保留最后不足一个 batch 的样本，保证每个 epoch 每张图都被使用一次。
    total_steps = epochs * len(loader)
    warmup_steps = math.ceil(total_steps * float(config["warmup_ratio"]))
    optimizer = torch.optim.AdamW(trainable.values(), lr=float(config["lr"]),
                                  weight_decay=float(config["weight_decay"]))
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    trainable_count = sum(param.numel() for param in trainable.values())
    total_count = sum(param.numel() for param in model.parameters())
    epoch_loss, epoch_seconds, epoch_samples = [], [], []
    with RunLogger("lit", model=model_name, seed=config["seed"], log_dir=log_dir) as log:
        log.info("可训练参数名：\n" + "\n".join(trainable))
        log.info(f"可训练参数 {trainable_count:,} / 总参数 {total_count:,} "
                 f"（{trainable_count / total_count:.2%}）")
        log.info(f"batch_size={batch_size}, gradient_checkpointing={gradient_checkpointing}, "
                 f"precision={'fp16' if use_amp else 'fp32'}")
        if use_amp:
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        for epoch in range(epochs):
            epoch_started = time.perf_counter()
            loss_sum, sample_count = 0.0, 0
            for batch in loader:
                captions = sample_captions(batch["captions"], caption_generator)
                tokens = tokenize_captions(tokenizer, captions, model_config["tokenizer"])
                tokens = {key: value.to(device, non_blocking=use_amp)
                          for key, value in tokens.items()}
                pixels = batch["image"].to(device, non_blocking=use_amp)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, dtype=torch.float16,
                                    enabled=use_amp):
                    # 图像塔不建计算图，既保证冻结，也减少显存占用。
                    with torch.no_grad():
                        image_features = model.get_image_features(pixel_values=pixels)
                    text_features = model.get_text_features(**tokens)
                # 归一化及损失使用 fp32；float() 不会切断文本分支的梯度。
                image_features = F.normalize(image_features.float(), dim=-1)
                text_features = F.normalize(text_features.float(), dim=-1)
                if model_name == "clip":
                    loss = info_nce(image_features, text_features, model.logit_scale)
                else:
                    loss = sigmoid_loss(image_features, text_features,
                                        model.logit_scale, model.logit_bias)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"第 {epoch + 1} 轮出现非有限 loss。")
                scaler.scale(loss).backward()
                previous_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                # 梯度溢出时 GradScaler 跳过更新，此时学习率也不提前推进。
                if scaler.get_scale() >= previous_scale:
                    scheduler.step()
                count = pixels.shape[0]
                loss_sum += loss.item() * count
                sample_count += count
            if use_amp:
                torch.cuda.synchronize(device)
            epoch_loss.append(loss_sum / sample_count)
            epoch_seconds.append(time.perf_counter() - epoch_started)
            epoch_samples.append(sample_count)
            log.info(f"epoch {epoch + 1}/{epochs}: loss={epoch_loss[-1]:.6f}, "
                     f"耗时={epoch_seconds[-1]:.2f}s, lr={scheduler.get_last_lr()[0]:.8g}")

        # 原样保存 Hugging Face 参数名，P2/P6 可用 strict=False 加载到预训练模型。
        # 只保存最后一轮可训练参数，不包含图像塔、温度和偏置，也不挑选最优轮次。
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        state = {name: param.detach().cpu() for name, param in trainable.items()}
        torch.save(state, output_dir / "trainable_state.pt")
        meta = {
            **config,
            "model": model_name,
            "hf_name": model_config["hf_name"],
            "setting": "lit",
            "epochs": epochs,
            "batch_size": batch_size,
            "device": str(device),
            "precision": "fp16" if use_amp else "fp32",
            "gradient_checkpointing": gradient_checkpointing,
            "num_workers": num_workers,
            "num_samples": len(dataset),
            "trainable_names": list(trainable),
            "trainable_params": trainable_count,
            "total_params": total_count,
            "trainable_ratio": trainable_count / total_count,
            "warmup_steps": warmup_steps,
            "total_steps": total_steps,
            "final_lr": scheduler.get_last_lr()[0],
            "epoch_loss": epoch_loss,
            "epoch_seconds": epoch_seconds,
            "epoch_samples": epoch_samples,
            "wall_sec": time.perf_counter() - started,
            "gpu_mem_peak_MB": (torch.cuda.max_memory_allocated(device) / 2**20
                                if use_amp else 0.0),
        }
        with open(output_dir / "meta.json", "w", encoding="utf-8") as file:
            json.dump(meta, file, ensure_ascii=False, indent=2)
        log.metric("epoch_loss", epoch_loss)
        log.info(f"已保存最后一轮文本参数与 meta.json：{output_dir}")
    return meta


def main():
    parser = argparse.ArgumentParser(description="LiT：冻结图像分支，微调文本分支")
    parser.add_argument("--model", choices=["clip", "siglip"], required=True)
    parser.add_argument("--config", default=str(Path(REPO_ROOT) / "configs/lit.yaml"))
    parser.add_argument("--device", default="cuda", help="正式训练使用 cuda，调试可选 cpu")
    parser.add_argument("--batch-size", type=int, choices=[64, 128],
                        help="覆盖配置中的 batch；显存不足时用 64")
    parser.add_argument("--gradient-checkpointing", action="store_true",
                        help="通过重算文本前向来降低激活显存")
    parser.add_argument("--max-samples", type=int, help="仅取训练集前 N 张图试跑，例如 256")
    parser.add_argument("--epochs", type=int, help="覆盖轮数，试跑时可设为 1")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--ckpt-root", default=str(Path(REPO_ROOT) / "ckpt"))
    parser.add_argument("--log-dir", default=str(Path(REPO_ROOT) / "logs"))
    args = parser.parse_args()
    if args.max_samples is not None and args.max_samples < 2:
        parser.error("--max-samples 必须至少为 2")
    if args.num_workers < 0:
        parser.error("--num-workers 不能为负数")

    with open(args.config, encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size
    if args.epochs is not None:
        config["epochs"] = args.epochs
    model_config = load_models_cfg()[args.model]
    set_seed(int(config["seed"]))
    # P2 的 models/loader.py 尚未提供；训练直接使用原生模型以访问参数和梯度。
    # 模型标识和分词规则仍来自全组统一的 models.yaml。
    model_class = CLIPModel if args.model == "clip" else SiglipModel
    tokenizer_class = CLIPTokenizer if args.model == "clip" else AutoTokenizer
    model = model_class.from_pretrained(model_config["hf_name"])
    tokenizer = tokenizer_class.from_pretrained(model_config["hf_name"])
    # 只取 clean 输入，图像尺寸、resize 和归一化均由 P1 数据接口处理。
    dataset = get_dataset(*config["train_split"], cond="clean", model=args.model)
    if args.max_samples is not None:
        dataset = Subset(dataset, range(min(args.max_samples, len(dataset))))
    run_training(args.model, model, tokenizer, dataset, config, model_config,
                 device=args.device, output_dir=Path(args.ckpt_root) / args.model / "lit",
                 log_dir=args.log_dir, num_workers=args.num_workers,
                 gradient_checkpointing=args.gradient_checkpointing)


if __name__ == "__main__":
    main()
