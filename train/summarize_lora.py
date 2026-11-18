"""Collect only completed full-data P4 runs; never synthesize missing results."""
import csv
import json
from pathlib import Path

from data.config import REPO_ROOT


def main():
    root = Path(REPO_ROOT)
    rows, checks = [], []
    for model in ("clip", "siglip"):
        for rank in (8, 16):
            folder = root / "ckpt" / model / f"lora_r{rank}"
            meta = json.loads((folder / "meta.json").read_text())
            check = json.loads((folder / "selfcheck.json").read_text())
            assert meta["num_samples"] == 8000 and meta["epochs"] == 5
            assert meta["target_count"] == 96 and check["merge_pass"] and check["p2_loader_pass"]
            rows.append({k: meta[k] for k in ("model", "rank", "trainable_params", "total_params", "ratio")})
            checks.append((meta, check))
    with (root / "results/lora_params.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# P4 LoRA 自检", "", "使用 val 前 500 张图与每图第一条 caption；原始嵌入余弦相似度。",
             "merge：同一批 32 张，fp32 eval、关闭 TF32；图像及文本最大绝对差均要求 <1e-4。", "",
             "|模型|rank|图像余弦均值|文本余弦均值|图像 merge 最大差|文本 merge 最大差|P2 加载|",
             "|---|---|---|---|---|---|---|"]
    for meta, check in checks:
        c, d = check["cosine"], check["merge_max_abs_diff"]
        lines.append(f"|{meta['model']}|{meta['rank']}|{c['image']['mean']:.8f}|{c['text']['mean']:.8f}|{d['image']:.9g}|{d['text']:.9g}|通过|")
    lines += ["", "各组详细分布、损失与元数据见对应 checkpoint 目录。",
              "以上只验证表征变化和保存恢复的一致性，不代表检索性能或鲁棒性结论。"]
    lines += ["", "## AMP 运行记录", "",
              "GradScaler 遇到溢出时跳过优化器更新，scheduler 同时不推进，与现有 P3 脚本一致。",
              "因此有跳步时最后学习率可能略大于 0；实际值与跳步次数如下：", ""]
    for meta, _ in checks:
        lines.append(f"- {meta['model']} r={meta['rank']}: optimizer_updates={meta['optimizer_updates']}, "
                     f"skipped_updates={meta['skipped_updates']}, final_lr={meta['final_lr']:.9g}。")
    (root / "docs/lora_selfcheck.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    runs = [json.loads(s) for s in (root / "logs/runs.jsonl").read_text().splitlines() if s.strip()]
    chosen = []
    for meta, _ in checks:
        matches = [r for r in runs if r.get("name") == "lora" and r.get("model") == meta["model"]
                   and r.get("setting") == meta["setting"] and r.get("status") == "ok"
                   and r.get("num_samples") == 8000 and r.get("epochs") == 5]
        assert matches, (meta["model"], meta["rank"])
        chosen.append(matches[-1])
    (root / "logs/P4_runs.jsonl").write_text("".join(json.dumps(r)+"\n" for r in chosen))
    print("P4 summary complete: four verified full-data runs")


if __name__ == "__main__":
    main()
