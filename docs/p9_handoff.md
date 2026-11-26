# P9 交付与上游核对（2026-11-26）

P9 依据《分工方案_无PPT版》负责降维可视化、报告图表整理、一致性核对与日志汇总，不包含模型训练或 PPT。代码入口和复现命令见 [viz/README.md](../viz/README.md)。

## 已完成

| 项目 | 交付 |
|---|---|
| 输入验收 | 128 份真实 NPZ，全部通过官方 split/ID/标签/首条 caption 映射检查；全部 SHA-256 与 P5 收件记录一致 |
| 降维 | 58 次联合 PCA + t-SNE/UMAP；每个数据集固定 1000 张图、每图第 1 条文本 |
| 正式图 | 86 张 PNG（300 dpi）及 86 张 SVG，含 28 张 COCO 类别版本；CLIP pretrained/Flickr 包含一组七条件强度比较 |
| 可追溯数据 | 58 份坐标 CSV 和 58 份参数/版本/来源哈希 JSON |
| 报告表格 | 7 份 Markdown 与同名 CSV：val/test 检索、CIFAR、对齐与均匀性、间隙、类别原型、漂移 |
| 全组图号 | `docs/figure_index.csv` 登记 161 张已收到的图，包含历史阶段图；同图的 PNG/SVG/PDF 共用图号 |
| 图注 | `docs/figure_captions.md`，描述投影或来源记录，不作因果判定 |
| 一致性检查 | 固定种子随机抽查 10 个数字、10 张图，通过；详情 `docs/consistency_check.md` / `.json` |
| 日志汇总 | `results/compute_log.csv` / `.md`，138 条记录；清楚区分运行记录与仅有元数据的记录 |
| 自动测试 | P9 专项 12 项通过；未声称运行全部需要 PyTorch/GPU 的仓库测试 |

所有正式投影来自云盘真实特征；开发期演示位于独立的 `results/p9_demo/`，不进入正式交付包。两种模型分别降维，所有对比组先联合拟合，再拆面板，组内坐标轴范围一致。

## 上游实际交付情况

核查范围：GitHub `Eurekahah/mmrep` 的 `main`（`79e32d14b480613168d585edd57def956b5f1fa8`）和组内云盘共享目录。云盘递归复查共 287 个文件。这里只陈述共享位置已交付的内容，不推测个人本地进度。

| 负责人 | 共享位置已看到的交付 | 仍缺的 P9 依赖 |
|---|---|---|
| P1 | 划分、配置、数据契约、假特征生成器、RunLogger | 无 |
| P2 | 128 个 NPZ、160 行范数统计、图、验证收据、`P2_runs.jsonl` | 无 |
| P3 | 两组 LiT 权重与 meta，均记录 5 epoch / batch 128；P2 已交付 LiT 特征 | `logs/P3_runs.jsonl` |
| P4 | 四组 LoRA 权重与 meta、参数量表、自检、`P4_runs.jsonl` | 无 |
| P5 | 检索 val 80 行 / test 48 行，漂移 448 行，检索降幅、图及观察记录 | 无 |
| P6 | `cifar100_top1.csv`：2 模型 × 4 设置 × 3 种子，共 24 行；仓库有 8 行汇总表 | `logs/P6_runs.jsonl` |
| P7 | 几何指标 80 行、原型指标 24 行、几何与强度图 | 无 |
| P8 | `analysis/README.md` 仍为待实现占位；未见交付的 SVD 表/图或总表 | `results/svd_energy.csv`、SVD 图及生成脚本、`results/summary_table.xlsx`（含 `strength` sheet） |

“P3 checkpoint 好了，通知 P2/P6”和“P6 CIFAR 结果已上传，通知 P8”两条交接与实际文件相符。完整运行日志是额外的交付项，缺少日志不表示这些实验结果未完成。

云盘 `logs/` 中实际找到的 JSONL 只有：

- `logs/P2_runs.jsonl`，96,898 字节。
- `logs/P4_runs.jsonl`，1,496 字节。

所有其他子目录也未找到 `P3_runs.jsonl` 或 `P6_runs.jsonl`。GitHub 的 `logs/` 仅有 `.gitkeep`。

P3 两个 `meta.json` 包含训练耗时/显存，可先作为元数据引用：CLIP-LiT 279.06 秒、4180.89 MiB；SigLIP2-LiT 252.45 秒、5878.65 MiB。已以 `metadata_only` 纳入汇总，不把它们伪装成尚未收到的 RunLogger 日志。

## 待上游补齐后的操作

将新增表、图和分人 JSONL 按原目录放入仓库，再执行：

```bash
python -m viz.run_p9 --skip-projections
```

这会重新生成报告表格、日志汇总、图号索引、图注和一致性抽查，保留原有图号。待 P8 表/图与 P3/P6 日志实际到齐后，才可将机器可读 `delivery_status.json` 的整体状态视为全部完成。当前 `projections_complete=true`、`consistency_passed=true`，整体 `complete=false`。

## 数值与解释边界

- P5 Recall 输入为 0–100；P6/P7 准确率输入为 0–1。输出统一百分数一位小数，余弦、间隙等四位小数。CIFAR 的很小标准差因此可能显示为 `0.0%`，原始非零值保存在来源 CSV 和表格 manifest。
- 二维质心距离只描述本张投影，不是原空间模态间隙，不跨图或跨模型比较；高维冻结检查以 P2 表为准。
- 所有上游图登记了可找到的生成脚本/来源；未重跑上游绘图。`gantt.png` 没有单独生成脚本，索引保留“待补”状态。
- 本次没有改动 P1–P8 的共享实现。P9 复用 `eval/p5_common.py` 的输入校验函数，并以来源清单和 README 说明依赖，方便组内协作。

## 交付位置

完整交付包已上传到组内云盘 `results/P9_交付_2026-11-26.zip`，旁边的同名 `.json` 保存大小与 SHA-256。压缩包共 480 个文件、101,387,527 字节；上传后重新下载，内容哈希与本地一致。回执见 [p9_upload_receipt.json](p9_upload_receipt.json)。包内包含正式 PNG/SVG、坐标、报告表格、来源小文件及 P9 代码；128 份特征和模型权重仍从原云盘目录获取。

GitHub 的 P9 分支包含代码、说明与小型报告文件，完整图和坐标从上述云盘交付包解压到仓库根目录即可查看。包内 `P9_MANIFEST.json` 逐文件记录 SHA-256；当前 README 与上传回执是在上传完成后补充的说明，不影响压缩包已验证的内容。
