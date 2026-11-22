"""P3 训练后自检：val 图文嵌入对照、loss 曲线和训练资源汇总。

示例：python -m train.selfcheck_lit --model all --ckpt-root ckpt
只读取已有 checkpoint，不训练、不覆盖训练日志或训练元数据。
"""
import argparse
from contextlib import contextmanager, nullcontext
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from data.config import REPO_ROOT
from data.datasets import collate_fn, get_dataset
from models.checkpoint_receipt import read_training_receipt
from models.loader import load_model
from tools.logger import RunLogger


@contextmanager
def strict_fp32(device):
    """自检统一使用 fp32；CUDA 暂时关闭 TF32，退出后恢复原设置。"""
    if device.type != "cuda":
        with torch.autocast(device_type=device.type, enabled=False):
            yield
        return
    previous = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        with torch.autocast(device_type="cuda", enabled=False):
            yield
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = previous


def row_cosine(before, after):
    """逐样本计算余弦；在 CPU 用 float64 汇总，拒绝零向量和非有限值。"""
    before, after = np.asarray(before, dtype=np.float64), np.asarray(after, dtype=np.float64)
    if before.ndim != 2 or before.shape != after.shape or not before.size:
        raise ValueError("两组嵌入必须具有相同的非空 [N, D] 形状。")
    if not np.isfinite(before).all() or not np.isfinite(after).all():
        raise ValueError("嵌入包含非有限数值，不能计算有效的余弦相似度。")
    norm_before = np.linalg.norm(before, axis=1, keepdims=True)
    norm_after = np.linalg.norm(after, axis=1, keepdims=True)
    if (norm_before == 0).any() or (norm_after == 0).any():
        raise ValueError("嵌入包含零向量，余弦相似度无定义。")
    values = np.sum((before / norm_before) * (after / norm_after), axis=1)
    return np.clip(values, -1.0, 1.0)


def summarize(values):
    return {"mean": float(values.mean()), "std": float(values.std()),
            "min": float(values.min()), "max": float(values.max())}


def extract_embeddings(wrapper, dataset, num_samples, batch_size):
    """保持划分顺序，每张图只取第 1 条 caption；两次提取使用相同 batch。"""
    loader = DataLoader(Subset(dataset, range(num_samples)), batch_size=batch_size,
                        shuffle=False, drop_last=False, num_workers=0, collate_fn=collate_fn)
    ids, captions, images, texts = [], [], [], []
    wrapper.eval()
    with torch.no_grad():
        for batch in loader:
            selected = [group[0] for group in batch["captions"]]
            pixels = batch["image"].to(wrapper.device, dtype=torch.float32)
            tokens = {key: value.to(wrapper.device)
                      for key, value in wrapper.tokenize(selected).items()}
            images.append(wrapper.encode_image(pixels).float().cpu().numpy())
            texts.append(wrapper.encode_text(**tokens).float().cpu().numpy())
            ids.extend(batch["id"])
            captions.extend(selected)
    if ids != list(dataset.ids[:num_samples]):
        raise ValueError("提取顺序与 val 划分顺序不一致。")
    return {"ids": ids, "captions": captions,
            "image": np.concatenate(images), "text": np.concatenate(texts)}


def write_csv(path, fields, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def save_loss_curve(out_dir, model_name, meta):
    """绘制已完成训练的 epoch loss；不把自检耗时当作训练耗时。"""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    losses = np.asarray(meta["epoch_loss"], dtype=float)
    seconds = np.asarray(meta["epoch_seconds"], dtype=float)
    if (losses.shape != (meta["epochs"],) or seconds.shape != losses.shape
            or not np.isfinite(losses).all() or not np.isfinite(seconds).all()
            or (seconds < 0).any()):
        raise ValueError("meta.json 的每轮 loss 或耗时缺失、数量不符或含非法数值。")
    epochs = np.arange(1, len(losses) + 1)
    write_csv(out_dir / "epochs.csv", ["epoch", "loss", "seconds"],
              [{"epoch": int(epoch), "loss": float(loss), "seconds": float(duration)}
               for epoch, loss, duration in zip(epochs, losses, seconds)])
    # 使用无窗口的 Agg 后端，服务器不需要桌面环境或中文字体。
    figure = Figure(figsize=(6.4, 4.2), dpi=160)
    FigureCanvasAgg(figure)
    axis = figure.subplots()
    axis.plot(epochs, losses, marker="o", color="#2563eb", linewidth=2)
    axis.set(title=f"{'CLIP' if model_name == 'clip' else 'SigLIP2'} LiT training loss",
             xlabel="Epoch", ylabel="Training loss", xticks=epochs)
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(out_dir / "loss.png")


def run_selfcheck(model_name, *, ckpt_root=Path(REPO_ROOT) / "ckpt", data_root=None,
                  splits_dir=None, device="cuda", num_samples=500, batch_size=32,
                  image_tolerance=1e-5, out_dir=Path(REPO_ROOT) / "results/lit_selfcheck",
                  log_dir=Path(REPO_ROOT) / "logs", cache_dir=None,
                  local_files_only=False, revision=None):
    """比较预训练模型与重载 LiT 模型，写报告并追加独立标记的自检日志。"""
    if num_samples <= 0 or batch_size <= 0 or not 0 < image_tolerance < 1:
        raise ValueError("样本数、batch 必须为正，余弦容差必须在 (0, 1) 内。")
    dataset = get_dataset("flickr", "val", "clean", model_name,
                          data_root=data_root, splits_dir=splits_dir)
    if len(dataset) < num_samples:
        raise ValueError(f"val 只有 {len(dataset)} 张图，无法按要求检查前 {num_samples} 张。")
    ckpt_root = Path(ckpt_root)
    if not ckpt_root.is_absolute():
        ckpt_root = Path(REPO_ROOT) / ckpt_root
    receipt = read_training_receipt(ckpt_root, model_name, "lit")
    meta_path = ckpt_root / model_name / "lit/meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8-sig"))
    destination = Path(out_dir) / model_name
    destination.mkdir(parents=True, exist_ok=True)
    device = torch.device(device)
    results, resolved_revision = {}, revision

    # 独立运行名避免 runs.jsonl 中的自检耗时、显存被误认为原始训练统计。
    with (torch.cuda.device(device) if device.type == "cuda" else nullcontext()), strict_fp32(device):
        with RunLogger("lit_selfcheck", model=model_name, setting="lit", seed=meta["seed"],
                       log_dir=log_dir, track_cuda=device.type == "cuda") as log:
            log.info(f"检查 Flickr val 前 {num_samples} 张 clean 图片，每图第 1 条 caption；fp32。")
            for setting in ("pretrained", "lit"):
                # 依次加载，避免两个完整模型同时占用显存；固定为同一基础模型版本。
                wrapper = load_model(model_name, setting, ckpt_root=ckpt_root, device=str(device),
                                     cache_dir=cache_dir, local_files_only=local_files_only,
                                     revision=resolved_revision)
                try:
                    if setting == "pretrained":
                        resolved_revision = getattr(wrapper.model.config, "_commit_hash", None) or revision
                    results[setting] = extract_embeddings(wrapper, dataset, num_samples, batch_size)
                finally:
                    del wrapper
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                log.info(f"{setting} 的图像、文本嵌入提取完成。")
            before, after = results["pretrained"], results["lit"]
            if before["ids"] != after["ids"] or before["captions"] != after["captions"]:
                raise ValueError("两次提取的图片或文本没有一一对应。")
            image_cosine = row_cosine(before["image"], after["image"])
            text_cosine = row_cosine(before["text"], after["text"])
            bad_count = int(np.count_nonzero(1.0 - image_cosine > image_tolerance))
            report = {
                "model": model_name, "dataset": "flickr", "split": "val", "cond": "clean",
                "num_samples": num_samples, "caption_index": 0, "batch_size": batch_size,
                "device": str(device), "precision": "fp32", "resolved_revision": resolved_revision,
                "checkpoint": str(meta_path.parent / "trainable_state.pt"),
                "training_meta": str(meta_path), "training": receipt,
                "image_cosine": summarize(image_cosine), "text_cosine": summarize(text_cosine),
                "image_tolerance": image_tolerance, "image_failed_samples": bad_count,
                "image_passed": bad_count == 0,
                "image_raw_max_abs": float(np.max(np.abs(
                    after["image"].astype(np.float64) - before["image"].astype(np.float64)))),
                "scope": "checkpoint_reload_comparison",
                "note": "LiT checkpoint 只保存文本参数。本报告检查重载后图像侧一致性，"
                        "不单独证明训练期间图像参数从未变化；需结合冻结测试。"
                        "文本余弦仅作参照，不设置通过阈值。",
            }
            write_csv(destination / "cosine.csv",
                      ["index", "id", "caption", "image_cosine", "text_cosine"],
                      [{"index": i, "id": id_, "caption": before["captions"][i],
                        "image_cosine": float(image_cosine[i]), "text_cosine": float(text_cosine[i])}
                       for i, id_ in enumerate(before["ids"])])
            save_loss_curve(destination, model_name, meta)
            (destination / "report.json").write_text(
                json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
            verdict = "通过" if report["image_passed"] else "需要复核"
            lines = [f"# {model_name} LiT 自检报告", "",
                     f"检查范围：Flickr val 前 {num_samples} 张 clean 图片，每图第 1 条 caption。",
                     f"设备：{device}；精度：fp32；提取 batch：{batch_size}。", "",
                     "| 模态 | 余弦均值 | 标准差 | 最小值 | 最大值 |",
                     "|---|---:|---:|---:|---:|"]
            for label, stats in (("图像", report["image_cosine"]), ("文本", report["text_cosine"])):
                lines.append(f"| {label} | {stats['mean']:.10f} | {stats['std']:.10f} | "
                             f"{stats['min']:.10f} | {stats['max']:.10f} |")
            lines += ["", f"图像检查：**{verdict}**。要求每个样本的 1 − cosine ≤ {image_tolerance:g}；"
                      f"超出容差 {bad_count}/{num_samples} 张。",
                      f"图像原始嵌入最大绝对差：{report['image_raw_max_abs']:.10g}。", "",
                      f"训练记录来源：`{meta_path}`。",
                      f"训练 {receipt['epochs']} 轮，batch {receipt['batch_size']}；"
                      f"总耗时 {receipt['wall_sec']:.2f} 秒，峰值显存 {receipt['gpu_mem_peak_MB']:.2f} MiB。",
                      "显存字段名为 gpu_mem_peak_MB，原代码实际按 2**20 换算；上述数值来自训练元数据，"
                      "不是本次自检的资源开销。", "",
                      "![训练 loss 曲线](loss.png)", "",
                      "[逐样本余弦](cosine.csv) · [每轮 loss 与耗时](epochs.csv) · [机器可读报告](report.json)",
                      "", report["note"], ""]
            (destination / "report.md").write_text("\n".join(lines), encoding="utf-8")
            log.metric("image_passed", report["image_passed"])
            log.metric("image_cosine_mean", report["image_cosine"]["mean"])
            log.metric("text_cosine_mean", report["text_cosine"]["mean"])
            log.info(f"自检{verdict}，输出目录：{destination}")
            # 只追加有明确标记的结果，不重写服务器上已有的训练过程日志。
            training_log = RunLogger("lit", model=model_name, log_dir=log_dir, track_cuda=False)
            training_log.info(f"[SELF_CHECK] checkpoint 重载自检{verdict}；N={num_samples}；"
                              f"image_mean={report['image_cosine']['mean']:.10f}；"
                              f"text_mean={report['text_cosine']['mean']:.10f}；"
                              f"超出容差={bad_count}；报告={destination / 'report.json'}")
            training_log.info(f"[SELF_CHECK] 原训练 meta 汇总：wall_sec={receipt['wall_sec']:.2f}；"
                              f"gpu_mem_peak_MB={receipt['gpu_mem_peak_MB']:.2f}；"
                              f"loss曲线={destination / 'loss.png'}")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["clip", "siglip", "all"], default="all")
    parser.add_argument("--ckpt-root", default=str(Path(REPO_ROOT) / "ckpt"))
    parser.add_argument("--data-root", help="包含 flickr30k/ 的原始数据根目录")
    parser.add_argument("--splits-dir")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-samples", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-tolerance", type=float, default=1e-5)
    parser.add_argument("--out-dir", default=str(Path(REPO_ROOT) / "results/lit_selfcheck"))
    parser.add_argument("--log-dir", default=str(Path(REPO_ROOT) / "logs"))
    parser.add_argument("--cache-dir")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--revision", help="指定基础预训练模型版本；应与训练时一致")
    args = vars(parser.parse_args(argv))
    selection = args.pop("model")
    passed = True
    for name in (["clip", "siglip"] if selection == "all" else [selection]):
        report = run_selfcheck(name, **args)
        passed = passed and report["image_passed"]
    # 数值不满足要求时仍保存报告，但返回非零退出码，避免被当作自检通过。
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
