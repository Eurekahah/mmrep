# eval/

负责人：P5 / P6

待实现：retrieval.py、eval_retrieval.py、drift.py（P5）。P6 的
`probe_cifar100.py` 已实现。

## P6 CIFAR-100 线性探针

先按 `extract/README.md` 提取 8 个模型设置的 CIFAR 特征，再从仓库根目录运行：

```bash
python -m eval.probe_cifar100 --model all --setting all \
  --feature-root features_cifar --device cuda
```

脚本读取 `img_raw` 和 `labels`，先逐样本 L2 归一化，再训练
`nn.Linear(D, 100)`。默认 AdamW 学习率为 `1e-3`、weight decay 为 `1e-4`、
batch size 为 `1024`、训练 100 epoch，并分别运行 seed 0/1/2。最终只使用
第 100 个 epoch 的 test 结果，不从训练集划验证集或挑选最佳 epoch。

结果写入：

- `results/cifar100_top1.csv`：每个模型、设置、种子的 Top-1/Top-5；
- `results/cifar100_summary.csv`：跨种子的均值和标准差；
- `logs/runs.jsonl`：耗时、峰值显存和最终 loss。

约定见 [docs/CONTRACT.md](../docs/CONTRACT.md)。开发期用 `python -m tools.make_dummy_features --out features_dummy` 生成的假特征自测。完成后在这里补一份用法说明（README）。
