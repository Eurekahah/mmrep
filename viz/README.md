# P9：降维可视化、报告图表与日志汇总

按《分工方案_无PPT版》实施。P9 不训练模型、不重提取特征、不制作 PPT。当前上游交付与缺项见 [p9_handoff.md](../docs/p9_handoff.md)，机器可读状态见 `results/p9/delivery_status.json`。

实际上交范围见 [p9_submission_checklist.md](../docs/p9_submission_checklist.md)。代码与 README 放 GitHub；云盘精简包只放正式图表与交付文档。下面提到的坐标、哈希、输入验收和逐单元格 manifest 是脚本运行时生成的本地材料，可按需复现，不要求重复上传。PNG 和 SVG 两种正式图格式都保留。

## 环境和输入

Python 3.10；仅 CPU 即可。可以沿用组内环境，或在独立环境安装：

```bash
python -m pip install -r viz/requirements.txt
```

从组内云盘下载 `features/`、`results/`、`logs/` 及 `ckpt/*/*/meta.json`，保持目录结构，放在仓库根目录。无需下载原始图片或模型权重。`features/` 共 128 个 NPZ；正式绘图使用 val 的文件，test 已支持但不默认重复出图。所需格式与划分以 [CONTRACT.md](../docs/CONTRACT.md) 和仓库 `splits/` 为准。

所有命令在仓库根目录运行。正式一键运行：

```bash
python -m viz.run_p9
```

图已出完、收到新的表格/日志后：

```bash
python -m viz.run_p9 --skip-projections
```

`--skip-projections` 需要已有的 `results/p9/coordinates/` 缓存。精简包不包含缓存；新环境要准备输入后运行完整命令，或从已有本地/历史备份恢复缓存。查看已有图表无需运行脚本。

## 降维和绘图

```bash
python -m viz.tsne_umap --resume
python -m viz.tsne_umap --model clip --method tsne --resume
python -m viz.tsne_umap --model siglip --method umap --resume
```

- 固定种子 42；每个数据集固定抽取 1000 个图像 ID，在所有模型、设置、条件下复用；文本取每张图的第 1 条。
- 使用 P5 的 `load_features` / `assert_aligned` / `unit_rows`，检查真实划分顺序、COCO 标签、5-caption 映射、维度、非有限值、零向量。P9 不修改这些共用函数。
- 逐行 L2 归一化，联合 PCA 到 50 维，再做 t-SNE 或 UMAP。两个模型分别拟合。
- t-SNE：perplexity 30、init=pca、learning_rate=auto、1000 次迭代。UMAP：n_neighbors 15、min_dist 0.1、n_jobs=1。两者种子均为 42。
- 四设置对比：同一模型、数据集、条件下，4 ×（1000 图像 + 1000 文本）= 8000 点一次拟合。clean/noise/occl 各有一组。
- 条件对比：同一设置下 3 × 1000 图像与 1000 文本一起拟合，共 4000 点。文本不随图像扰动变化，因此只拟合一次，并在各面板复用同一组坐标。
- 强度对比：默认仅 CLIP pretrained、Flickr val，clean 与三档噪声、三档遮挡的 7000 图像及 1000 文本共 8000 点联合拟合；固定 30 张图画 clean → 扰动的位移箭头。
- 每组固定抽取 30 对图文画灰色连线；同一对比组使用完全相同的坐标轴范围和比例。
- 模态蓝/橙；设置以底部轴线颜色区分；图像按条件使用固定标记，文本用 `x`。COCO 另出类别图，按固定样本中频数最高的 10 类着色、其余灰色；数字为 COCO 类别 ID，可在 `splits/coco_split.json` 的 `class_names` 查询名称。

默认总计 **58 次联合降维、86 张图**（包含 28 张 COCO 类别版本），每张均保存 PNG（300 dpi）和 SVG。文件名遵循 `{model}_{method}_{group}_{dataset}`，类别版追加 `_classes`，test 版追加 `_test`。同一数据在不同对比组会重新联合拟合，因此不同图之间的二维坐标不可直接比较。

输出：

| 路径 | 内容 |
|---|---|
| `results/p9/figures/` | PNG/SVG |
| `results/p9/coordinates/` | 每组坐标 CSV、参数/环境/源文件与产物 SHA-256 JSON |
| `results/p9/manifest.json` | 全套图覆盖清单（由 `viz.run_p9` 汇总） |
| `results/p9/projection_runtime.csv` | 每次降维与绘图的 CPU 耗时 |

`--resume` 仅复用输入、代码、参数、环境及产物哈希全部匹配的结果。不同模型/方法可以分别运行，使用独立的 `manifest_{model}_{method}.json`，最后用 `viz.run_p9 --skip-projections` 合并覆盖清单。

二维投影仅辅助观察，不能用二维距离代替 P7 在原始嵌入空间计算的间隙。四设置拟合时完全相同的高维点仍可能因降维数值优化出现小幅二维偏差；冻结是否成立以 P2 的 `lit_frozen_check.csv` 为准。跨模型只并列展示，不比较二维距离或将差异归因于损失函数。

方法参考：[scikit-learn 1.5 t-SNE](https://scikit-learn.org/1.5/modules/generated/sklearn.manifold.TSNE.html)、[UMAP 随机种子与复现](https://umap-learn.readthedocs.io/en/latest/reproducibility.html)。实际软件版本记录在每组 JSON；不要在未固定版本时要求跨环境逐位相同。

## 报告表格、图号和核对

```bash
python -m viz.make_report_tables
python -m viz.build_figure_docs
python -m tools.summarize_logs
```

表格读取 `results/summary_table.xlsx` 和原始 CSV，输出 `results/report_tables/*.md`、同名 CSV 及包含每个数值来源行号的 `manifest.json`。原始 CSV 优先；总表可作为交叉检查和缺表时的来源，数值冲突会报错。未收到的 SVD/strength 工作表会明确列为待补，不使用合成数据填补。

**单位按上游接口处理**：P5 Recall 输入已是 0–100，直接保留 1 位并加 `%`；P6/P7 的准确率输入是 0–1，乘 100 后保留 1 位。余弦、间隙、均匀性、能量等保留 4 位，维数/样本数保留整数。来源 CSV 原始精度保持不变。CIFAR 优先采用上游 `cifar100_summary.csv`，原始种子结果仍在 `cifar100_top1.csv`。

`docs/figure_index.csv` 为已收到的组内 PNG 分配稳定图号；同名 SVG/PDF 共用图号。`docs/figure_captions.md` 提供图注和基于来源记录的描述。未提供生成脚本或数值来源的上游图会明确标记；不冒称重新运行过上游脚本。

一致性检查以种子 42 随机抽查 10 个表格数字与 10 张 P9 图，核对原始值、显示值、输入/输出哈希、坐标和图片规格，结果写入 `docs/consistency_check.md` / `.json`。是否满足最终全组交付，还需查看缺项清单。

## 日志汇总

读取 `logs/*_runs.jsonl` 与 `ckpt/*/*/meta.json`；没有分人日志时才回退到 `logs/runs.jsonl`。输出 `results/compute_log.csv`、`.md`、`compute_log_manifest.json`。

只去除完全一致的日志副本，保留失败、重试、跳过及未知状态。可唯一匹配的训练日志由 meta 补充 epoch/batch；没有对应训练日志的 meta 标为 `metadata_only`，不当成新运行累计耗时。上游父任务和子任务的耗时可能重叠，原始记录不要直接求和。显存列按组内命名为 MB，实际单位沿用 RunLogger 的 MiB；峰值显存不能求和。缺少 P3/P6 日志不等于其 checkpoint/准确率结果未完成。

## 测试和演示

```bash
python -m pytest tests/test_p9.py -q
python -m tools.make_dummy_features --out features_dummy --n 40
python -m viz.tsne_umap --features-root features_dummy --out-dir results/p9_demo --model clip --dataset flickr --group settings --n 40 --demo --max-iter 250
```

演示图带 `SYNTHETIC DEMO` 标识，禁止写入正式目录或当作实验结果。正式运行不要使用 `--demo` 或缩小采样数。
