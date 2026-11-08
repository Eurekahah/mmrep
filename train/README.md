# train/

负责人：P3 / P4

待实现：losses.py（P3）、train_lit.py（P3）、train_lora.py（P4），超参见 configs/lit.yaml 与 configs/lora.yaml

约定见 [docs/CONTRACT.md](../docs/CONTRACT.md)。开发期用 `python -m tools.make_dummy_features --out features_dummy` 生成的假特征自测。完成后在这里补一份用法说明（README）。


## P3：LiT 微调

采用 **LU 策略**：加载 CLIP/SigLIP2 预训练权重，冻结图像分支，仅微调文本分支。训练超参数见 `configs/lit.yaml`。

### 1. 文件说明

- `losses.py`：实现 CLIP 的双向 InfoNCE Loss 和 SigLIP2 的 Sigmoid Loss，供 P3/P4 共用。
- `train_lit.py`：实现数据加载、参数冻结、图文对比训练、学习率调度、混合精度、日志记录及 checkpoint 保存。
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