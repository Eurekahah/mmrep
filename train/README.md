# train/

负责人：P3 / P4

待实现：losses.py（P3）、train_lit.py（P3）、train_lora.py（P4），超参见 configs/lit.yaml 与 configs/lora.yaml

约定见 [docs/CONTRACT.md](../docs/CONTRACT.md)。开发期用 `python -m tools.make_dummy_features --out features_dummy` 生成的假特征自测。完成后在这里补一份用法说明（README）。


## P3：LiT 微调

采用 **LU 策略**：加载 CLIP/SigLIP2 预训练权重，冻结图像分支，仅微调文本分支。训练超参数见 `configs/lit.yaml`。

### 1. 文件说明

- `losses.py`：实现 CLIP 的双向 InfoNCE Loss 和 SigLIP2 的 Sigmoid Loss，供 P3/P4 共用。
- `train_lit.py`：实现数据加载、参数冻结、图文对比训练、学习率调度、混合精度、日志记录及 checkpoint 保存。
- `selfcheck_lit.py`：用 val 前 500 张图片及每图第 1 条 caption 比较 pretrained/LiT 嵌入，生成 loss 曲线、训练统计和自检报告。
- `../tests/test_losses.py`：验证随机输入下损失与梯度有限、InfoNCE 正确匹配行为，以及 Sigmoid Loss 的标签、偏置和归约计算。
- `../tests/test_train_lit.py`：使用 CPU 上的轻量 CLIP/SigLIP 模型与合成数据，验证冻结范围、Loss 与模型内置实现一致性、训练流程、参数更新、checkpoint 重载及日志输出。

### 2. 测试命令

在项目根目录运行：

```bash
python -m pytest tests/test_losses.py tests/test_train_lit.py -v
```

上述测试不需要下载预训练权重，不能替代正式模型和真实数据的 GPU 测试。

### 3. 训练命令

小规模试跑（256 张图片、1 epoch）：

```bash
python -m train.train_lit --model clip --max-samples 256 --epochs 1 --ckpt-root ckpt_debug
```

正式训练：

```bash
python -m train.train_lit --model clip --config configs/lit.yaml
python -m train.train_lit --model siglip --config configs/lit.yaml
```

训练完成后，输出至 `ckpt/{model}/lit/`，包含 `trainable_state.pt`（仅可训练参数）和 `meta.json`（训练配置、Loss、耗时、显存等）；训练日志保存在 `logs/`。

### 4. P3 训练后自检

在已有训练环境中，从仓库根目录运行。不需要重新训练；需要 Flickr 原图、划分文件、预训练模型和两个 LiT checkpoint。

```bash
# 服务器权重为 ckpt/{clip,siglip}/lit/ 时：
python -m train.selfcheck_lit --model all --ckpt-root ckpt --local-files-only
```

仅检查一个模型可用 `--model clip` 或 `--model siglip`。图片位于其他位置时，传入
`--data-root /path/to/data_raw`，该目录应包含 `flickr30k/flickr30k-images/`；也可以沿用
`MMREP_DATA_ROOT` 或 `configs/paths.yaml`。已有模型缓存可用 `--cache-dir /path/to/hf/cache`，
加 `--local-files-only` 可禁止联网下载。`--revision` 应指定与原训练一致的基础模型版本；
如需指定两个模型各自的 commit，请分别运行。

默认 `--num-samples 500 --batch-size 32 --image-tolerance 1e-5`。严格取 val 划分顺序的前 500 张，
不足 500 张会报错，不会静默减少样本。每图固定使用第 1 条 caption。两次提取使用相同 batch、
相同模型预处理、`eval()` 和 `no_grad()`，统一用 fp32，并在 CUDA 上临时关闭 TF32；
这不会修改原训练配置中的 fp16 设置。模型依次加载，避免同时占用两个完整模型的显存。
如显存不足，可将自检的 `--batch-size` 降为 16；这不改变原训练的 batch。

报告包含图像/文本逐样本余弦和均值、标准差、最小值、最大值。除检查均值外，还要求每张图满足
`1 - cosine <= image_tolerance`，避免少数异常被平均值掩盖；文本只作为参照，不设通过阈值。
图像未通过时仍输出结果，命令以退出码 1 结束；通过时为 0。不要把浮点误差导致的
`0.9999999` 直接解释为冻结失败。

默认每个模型生成以下文件，可用 `--out-dir` 修改结果根目录：

| 文件 | 内容 |
|---|---|
| `results/lit_selfcheck/{model}/cosine.csv` | val 序号、图片 ID、原始 caption、图像/文本余弦 |
| `results/lit_selfcheck/{model}/report.json` | 判定、统计、基础模型 revision、checkpoint 和训练元数据来源 |
| `results/lit_selfcheck/{model}/report.md` | 中文自检报告、训练耗时和显存汇总 |
| `results/lit_selfcheck/{model}/epochs.csv` | 从 `meta.json` 读取的每轮 loss 和训练耗时 |
| `results/lit_selfcheck/{model}/loss.png` | 每个模型独立的 epoch loss 曲线 |

训练耗时、显存来自原 `meta.json`，不会拿自检开销替代。显存原字段叫 `gpu_mem_peak_MB`，
代码按 `2**20` 换算，因此报告标为 MiB。重复运行会更新这些自检产物，不修改 checkpoint 或元数据。

日志默认写到 `logs/`，可用 `--log-dir` 指定服务器已有训练日志的目录：

- 在 `logs/lit_{model}.log` 追加带 `[SELF_CHECK]` 标记的摘要，原训练内容保持不变。
- 详细自检日志为 `logs/lit_selfcheck_{model}_lit.log`。
- `runs.jsonl` 追加 `name=lit_selfcheck` 的记录，自检与原来的 `name=lit` 训练记录分开。
- 若目标训练日志不存在，脚本只创建带自检标记的记录，不重建或伪造原始训练日志。

这里的“训练前”是重新加载的基础预训练模型，“训练后”是该基础模型加载文本 checkpoint 后的模型。
由于 LiT checkpoint 只保存文本参数，本自检验证的是导出模型重载后的图像侧一致性，不能单独证明
历史训练过程中图像参数从未变化，需结合已有冻结及训练单测。原训练元数据没有保存基础模型 revision，
本脚本只能固定这两次自检加载使用同一版本，无法追溯确认历史版本。

