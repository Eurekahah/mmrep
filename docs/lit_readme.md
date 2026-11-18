# P3：LiT 复现与运行记录

本文说明 CLIP / SigLIP2 的 LiT 训练、冻结自检、结果存档及已有运行耗时。所有命令均在仓库根目录执行，命令示例使用 Linux Bash；`--model siglip` 对应本项目配置的 SigLIP2。

已有两个模型各 5 epoch 的正式训练记录，以及 2026-11-18 完成的 val 前 500 张图自检记录。训练统计取自对应 checkpoint 的 `meta.json`，自检统计取自 `results/lit_selfcheck/{model}/report.json`，自检耗时取自 `logs/runs.jsonl` 中当日 21:53 的成功记录。

## 1. 环境、数据与预训练权重

已有 `mmrep` 环境时只需激活。以下是本次服务器使用的安装命令，供新机器复现；依赖已安装可跳过安装步骤。

```bash
conda create -n mmrep python=3.10 -y
conda activate mmrep
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```
下面的耗时仅代表本次运行，不作为跨硬件性能基准。

LiT 只需要 Flickr30K 图片和仓库已有的 `splits/flickr_split.json`，不需要 COCO / CIFAR-100，也不需要重新划分。默认图片目录为：

```text
data_raw/
└── flickr30k/
    └── flickr30k-images/
        └── *.jpg
```

尚未下载图片时，可在可联网的机器执行：

```bash
sh tools/download_data.sh flickr
```

数据不在默认目录时，设置 `MMREP_DATA_ROOT`，指向包含 `flickr30k/` 的目录；也可修改 `configs/paths.yaml` 的 `data_root`。训练读取已有 train 8000 张，自检读取 val 1000 张中的前 500 张。

模型、图像预处理和分词规则统一读取 `configs/models.yaml`：

| 命令参数 | Hugging Face 模型 | 输入尺寸 | 嵌入维度 |
|---|---|---|---|
| `clip` | `openai/clip-vit-base-patch32` | 224 × 224 | 512 |
| `siglip` | `google/siglip2-base-patch32-256` | 256 × 256 | 768 |

训练需能加载完整的基础模型和 tokenizer：首次加载需要联网下载，离线机器需提前准备完整缓存。`trainable_state.pt` 只含微调后的文本参数，不能替代基础模型或 tokenizer。`tools/make_dummy_features.py` 生成的是下游分析用假特征，不能替代训练所需的图片和文本。

## 2. 实现与训练设置

`train/losses.py` 提供两个共享损失，输入为已 L2 归一化的图像和文本嵌入：

- `info_nce(img, txt, logit_scale)`：用 `exp(logit_scale)` 缩放图文相似度，对图到文、文到图两个方向的交叉熵取平均。
- `sigmoid_loss(img, txt, logit_scale, logit_bias)`：相似度缩放后加偏置，对角为正样本，其余为负样本；负 log-sigmoid 求和后除以 batch 大小 `B`。

`train/train_lit.py` 加载预训练模型，冻结图像侧，仅训练文本侧。CLIP 开放 `text_model` 和 `text_projection`；SigLIP2 开放 `text_model`，其文本投影 `head` 已包含在其中。两者均冻结 `vision_model`、图像投影或池化头以及 `logit_scale`，SigLIP2 同时冻结 `logit_bias`。

图像塔保持 `eval()`，图像前向使用 `torch.no_grad()`；文本塔参与反向传播。CUDA 前向采用 fp16 autocast + GradScaler，参数、L2 归一化和损失计算保持 fp32。脚本打印可训练参数名、数量与占比，并写入 `meta.json`。

正式配置来自 `configs/lit.yaml`，遵循 `docs/CONTRACT.md` 3.5：

| 配置 | 值 |
|---|---|
| 数据 | Flickr train 8000 张，clean 输入 |
| caption | 每轮每图随机取 1 条，独立生成器种子 42 |
| epoch / batch | 5 / 128，保留末尾不足 128 的 batch |
| 优化器 | AdamW，lr=1e-4，weight_decay=0.01，默认 betas |
| 学习率 | 5% warmup，随后 cosine 衰减；本次总步数 315，warmup 16 步 |
| 随机种子 / num_workers | 42 / 0 |
| 保存策略 | 保存最后一轮，不用 val 选择最优 checkpoint |
| 梯度检查点 | 本次 CLIP 关闭，SigLIP2 开启 |

## 3. 测试与训练命令

可先执行已有 CPU 测试；不下载预训练权重，使用合成图片、小型真实模型结构和简化 tokenizer：

```bash
python -m pytest -q tests/test_losses.py tests/test_train_lit.py
python -m pytest -q tests/test_train_lit.py::test_lit_freeze_groups
```

服务器已有结果分别为 `10 passed in 6.22s` 和 `2 passed in 5.49s`。第二条的两个用例已包含在第一条的 10 个中，不是额外的两个测试。

`test_losses.py` 覆盖随机输入下损失及梯度有限、正交完美匹配且缩放足够大时 InfoNCE 接近零、错配损失增大，以及 sigmoid 标签、偏置和除以 `B` 的归约。`test_train_lit.py` 对两个模型分别检查冻结分组、损失与模型内置实现一致，以及训练、文本更新、冻结参数和图像输出不变、checkpoint 重载、元数据与日志输出。这些 CPU 测试不能替代真实 tokenizer、CUDA 混合精度和真实图片的运行验证。

新环境可先取 train 前 256 张真实图片跑 1 epoch，试跑 checkpoint 和日志单独存放：

```bash
python -m train.train_lit --model clip --config configs/lit.yaml \
  --max-samples 256 --epochs 1 --ckpt-root ckpt_debug --log-dir logs/lit_debug
python -m train.train_lit --model siglip --config configs/lit.yaml \
  --max-samples 256 --epochs 1 --gradient-checkpointing \
  --ckpt-root ckpt_debug --log-dir logs/lit_debug
```

与现有正式训练元数据一致的命令为：

```bash
python -m train.train_lit --model clip --config configs/lit.yaml
python -m train.train_lit --model siglip --config configs/lit.yaml --gradient-checkpointing
```

默认输出到 `ckpt/{model}/lit/` 和 `logs/`。重新训练会覆盖同目录的 `trainable_state.pt`、`meta.json`，日志则追加；保留已有结果时，为新一轮训练同时指定独立的 `--ckpt-root` 和 `--log-dir`，后续自检使用相同目录。

显存不足时先开启 `--gradient-checkpointing`，仍不足再加 `--batch-size 64`。实际 batch 与梯度检查点设置会写入元数据；降 batch 后不再是本文表格对应的训练配置。

## 4. 训练后自检与曲线生成

已有正式 checkpoint 时，直接运行以下命令，无需重新训练。本次服务器使用此命令成功完成两个模型的自检：

```bash
python -m train.selfcheck_lit --model all --ckpt-root ckpt --local-files-only
```

`--ckpt-root` 应指向直接包含 `clip/`、`siglip/` 的目录。若下载或解压后的文件实际位于 `ckpt/ckpt/{model}/lit/`，则改用：

```bash
python -m train.selfcheck_lit --model all --ckpt-root ckpt/ckpt --local-files-only
```

常用调整项：

- 仅检查一个模型：`--model clip` 或 `--model siglip`。
- 图片路径：`--data-root /path/to/data_raw`，或沿用 `MMREP_DATA_ROOT` / `configs/paths.yaml`。
- 预训练缓存路径：`--cache-dir /path/to/hf/cache`，此处不是 LiT checkpoint 目录。
- 输出位置：`--out-dir results/lit_selfcheck`、`--log-dir logs`；后者应指向对应的训练日志目录。
- 自检显存不足：`--batch-size 16`，只改变提取批量，不改变训练 batch。

`--local-files-only` 禁止自检联网，前提是基础模型与 tokenizer 缓存完整。未加此参数时，即使已有缓存也可能发生联网版本检查；服务器报 `Network is unreachable` 时应使用离线缓存。缓存缺失需先补齐。此参数属于 `train.selfcheck_lit`，`train.train_lit` 没有同名命令行参数。

自检严格按划分文件顺序取 val 前 500 张 clean 图片，每图固定第 1 条 caption；不足 500 张会报错。基础模型与加载 LiT 文本参数后的模型依次提取图像、文本嵌入，统一使用 `eval()`、`no_grad()`、fp32，并在 CUDA 上关闭 TF32。逐样本余弦再汇总为均值、标准差、最小值和最大值。

图像侧要求每张图都满足 `1 - cosine <= 1e-5`，文本侧只作参照，不设通过阈值。通过时退出码为 0；图像余弦检查不通过时仍保存报告，退出码为 1。浮点误差造成的极小偏差不应直接判作冻结失败。

同时从原训练 `meta.json` 读取每轮 loss 与耗时，生成 `epochs.csv`、`loss.png` 和报告，并向 `logs/lit_{model}.log` 追加 `[SELF_CHECK]` 摘要。

自检范围：这里的“训练前”是重新加载基础模型，“训练后”是加载已保存的文本参数。由于 checkpoint 只保存文本侧，本自检验证的是 checkpoint 重载后的图像侧一致性，需结合冻结及训练单测判断训练实现。原训练元数据未保存基础模型 revision；报告中的 `resolved_revision` 固定了本次自检两次加载的版本，但不能追溯证明历史训练使用了同一版本。

## 5. 已有运行结果与耗时

两组正式训练均为 8000 张图片、5 epoch、batch 128、CUDA fp16。统计如下：

| 模型 | 可训练 / 总参数 | 可训练占比 | 训练耗时（秒） | 峰值显存（MiB） | 首轮 → 末轮 loss |
|---|---|---|---|---|---|
| CLIP | 63,428,096 / 151,277,313 | 41.93% | 279.06 | 4180.89 | 0.547483 → 0.195682 |
| SigLIP2 | 282,303,744 / 376,856,066 | 74.91% | 252.45 | 5878.65 | 1.115975 → 0.563724 |

每轮 loss 为按样本数加权的平均训练损失：

| Epoch | CLIP loss | CLIP 耗时（秒） | SigLIP2 loss | SigLIP2 耗时（秒） |
|---|---|---|---|---|
| 1 | 0.547483 | 56.34 | 1.115975 | 51.19 |
| 2 | 0.432772 | 55.54 | 0.904236 | 50.11 |
| 3 | 0.308193 | 55.59 | 0.740359 | 49.87 |
| 4 | 0.228427 | 55.76 | 0.628967 | 50.15 |
| 5 | 0.195682 | 55.50 | 0.563724 | 49.64 |

训练 `wall_sec` 从训练循环开始前计时，包含各轮数据读取与训练以及末尾文本权重保存，不包含基础模型和 tokenizer 加载。每轮耗时不含末尾 checkpoint 保存，因此每轮耗时之和与总耗时有少量差异。显存使用 `torch.cuda.max_memory_allocated() / 2**20`，字段虽名为 `gpu_mem_peak_MB`，实际单位是 MiB，表示 PyTorch 分配的峰值显存，不等同于整卡显存占用。

两个模型的训练 loss 均逐轮下降；损失定义和尺度不同，不直接用绝对 loss 比较模型优劣。每个模型的曲线分别保存到 `results/lit_selfcheck/{model}/loss.png`。

500 张验证图的自检结果：

| 模型 | 图像余弦均值 | 图像超出容差数 | 文本余弦均值 | 图像原始嵌入最大绝对差 | 图像判定 |
|---|---|---|---|---|---|
| CLIP | 1.0000000000 | 0 / 500 | 0.7341315155 | 0.0 | 通过 |
| SigLIP2 | 1.0000000000 | 0 / 500 | 0.8722189674 | 0.0 | 通过 |

文本余弦低于 1 表明文本表征发生变化，不能据此断言检索性能提高。图像侧的结论适用于上述 checkpoint 重载自检范围。

成功的离线自检运行开销如下，与原训练开销分别记录：

| 模型 | 自检耗时（秒） | 自检峰值显存（MiB） |
|---|---|---|
| CLIP | 9.21 | 687.50 |
| SigLIP2 | 10.78 | 1542.40 |

`logs/runs.jsonl` 同时保留试跑、失败和成功记录。汇总正式训练时应核对对应 checkpoint 的 `meta.json`；本文自检开销取 2026-11-18 21:53 的成功离线运行，不包含此前网络重试。

## 6. 产物位置与交接

以下路径均相对仓库根目录；`{model}` 为 `clip` 或 `siglip`：

| 路径 | 内容 |
|---|---|
| `ckpt/{model}/lit/trainable_state.pt` | 最后一轮文本参数，保留原模型参数名 |
| `ckpt/{model}/lit/meta.json` | 实际训练配置、可训练参数名及占比、每轮 loss / 耗时、总耗时与峰值显存 |
| `logs/lit_{model}.log` | 训练参数清单、每轮记录与追加的 `[SELF_CHECK]` 摘要 |
| `logs/lit_selfcheck_{model}_lit.log` | 自检详细日志与自检耗时、显存 |
| `logs/runs.jsonl` | 结构化运行记录；`name=lit` 为训练，`name=lit_selfcheck` 为自检 |
| `results/lit_selfcheck/{model}/cosine.csv` | 每个样本的序号、图片 ID、caption、图像 / 文本余弦 |
| `results/lit_selfcheck/{model}/epochs.csv` | 原训练每轮 loss 和耗时 |
| `results/lit_selfcheck/{model}/loss.png` | 原训练 loss 曲线 |
| `results/lit_selfcheck/{model}/report.json` | 自检统计、通过判定、基础模型版本及训练元数据来源 |
| `results/lit_selfcheck/{model}/report.md` | 中文自检报告和训练统计 |

重复自检会更新对应报告、CSV 和曲线，日志追加，checkpoint 与训练元数据保持原值。只有自检前已存在的训练日志才包含原训练过程；自检不会补造缺失的训练记录。

交给 P2 / P6 的每个 checkpoint 目录需同时包含 `trainable_state.pt` 和 `meta.json`。恢复时先加载对应基础预训练模型，再用 `model.load_state_dict(state, strict=False)` 加载文本参数；项目 `models/loader.py` 已提供 LiT 加载接口。该 checkpoint 不包含优化器和 GradScaler 状态，不用于断点续训。

交付时一并保留两个模型的 checkpoint、`results/lit_selfcheck/` 和对应日志，保持目录结构。权重及运行产物按项目约定通过网盘等方式交接，不加入 Git；本文件与训练代码、配置一起保存。
