# P9 交付与上游核对（2026-11-27）

P9 依据《分工方案_无PPT版》负责降维可视化、报告图表整理、一致性核对与日志汇总，不包含模型训练或 PPT。代码入口和复现命令见 GitHub 的 [viz/README.md](https://github.com/Eurekahah/mmrep/blob/p9-visualization-report-20261126/viz/README.md)。本次按 [上交清单](p9_submission_checklist.md) 精简云盘包，只保留交接成果。

## 已完成

| 项目 | 交付 |
|---|---|
| 输入验收 | 128 份真实 NPZ，全部通过官方 split/ID/标签/首条 caption 映射检查；全部 SHA-256 与 P5 收件记录一致 |
| 降维 | 58 次联合 PCA + t-SNE/UMAP；每个数据集固定 1000 张图、每图第 1 条文本 |
| 正式图 | 86 张 PNG（300 dpi）及 86 张 SVG，含 28 张 COCO 类别版本；CLIP pretrained/Flickr 包含一组七条件强度比较 |
| 本地复现材料 | 58 份坐标 CSV 和 58 份参数/版本/来源哈希 JSON，保留本地和历史备份，不再重复上交 |
| 报告表格 | 7 份 Markdown 与同名 CSV：val/test 检索、CIFAR、对齐与均匀性、间隙、类别原型、漂移 |
| 全组图号 | `docs/figure_index.csv` 登记 161 张已收到的图，包含历史阶段图；同图的 PNG/SVG/PDF 共用图号 |
| 图注 | `docs/figure_captions.md`，描述投影或来源记录，不作因果判定 |
| 一致性检查 | 固定种子随机抽查 10 个数字、10 张图，通过；交付 `docs/consistency_check.md`，机器检查 JSON 留本地 |
| 日志汇总 | `results/compute_log.csv` / `.md`，194 条记录；P2/P3/P4/P6 日志均已收到，元数据参考记录单独标记 |
| 自动测试 | P9 专项 12 项通过；未声称运行全部需要 PyTorch/GPU 的仓库测试 |

所有正式投影来自云盘真实特征；开发期演示位于独立的 `results/p9_demo/`，不进入正式交付包。两种模型分别降维，所有对比组先联合拟合，再拆面板，组内坐标轴范围一致。

## 上游实际交付情况

核查范围：GitHub `Eurekahah/mmrep` 的 `main`（`79e32d14b480613168d585edd57def956b5f1fa8`）、P9 PR #2 和组内云盘共享目录。11 月 27 日上传精简包前，云盘递归复查共 291 个文件。这里只陈述共享位置已交付的内容，不推测个人本地进度。

| 负责人 | 共享位置已看到的交付 | 仍缺的 P9 依赖 |
|---|---|---|
| P1 | 划分、配置、数据契约、假特征生成器、RunLogger | 无 |
| P2 | 128 个 NPZ、160 行范数统计、图、验证收据、`P2_runs.jsonl` | 无 |
| P3 | 两组 LiT 权重与 meta，均记录 5 epoch / batch 128；LiT 特征、10 条运行日志已收到 | 无 |
| P4 | 四组 LoRA 权重与 meta、参数量表、自检、`P4_runs.jsonl` | 无 |
| P5 | 检索 val 80 行 / test 48 行，漂移 448 行，检索降幅、图及观察记录 | 无 |
| P6 | `cifar100_top1.csv` 24 行、8 行汇总表与 46 条日志；其中 24 条探针 Top-1 与 CSV 逐行完全一致 | 无 |
| P7 | 几何指标 80 行、原型指标 24 行、几何与强度图 | 无 |
| P8 | `analysis/README.md` 仍为待实现占位；未见交付的 SVD 表/图或总表 | `results/svd_energy.csv`、SVD 图及生成脚本、`results/summary_table.xlsx`（含 `strength` sheet） |

“P3 checkpoint 好了，通知 P2/P6”和“P6 CIFAR 结果已上传，通知 P8”两条交接与实际文件相符。完整运行日志与 checkpoint/准确率结果分开验收；本轮已补齐人员日志。

云盘 `logs/` 中实际找到的人员 JSONL：

- `logs/P2_runs.jsonl`，96,898 字节。
- `logs/P3_runs.jsonl`，2,277 字节。
- `logs/P4_runs.jsonl`，1,496 字节。
- `logs/P6_runs.jsonl`，13,085 字节。

P3 有 10 条记录，包含试跑、正式训练和自检，其中 2 条原始失败记录如实保留。P6 有 46 条记录，其中 24 条探针结果与 `cifar100_top1.csv` 按模型/设置/种子逐行对齐，Top-1 最大差异为 0。

P3 训练日志的 `setting` 为空、未直接记录 epoch/batch，汇总保留原始字段；两个 `meta.json` 另以 `metadata_only` 提供训练配置和耗时参考，不能再累计一次训练时间。CLIP-LiT 279.06 秒、SigLIP2-LiT 252.45 秒的正式日志与元数据耗时一致。

## 待上游补齐后的操作

将新增表、图和分人 JSONL 按原目录放入仓库，再执行：

```bash
python -m viz.run_p9 --skip-projections
```

这会重新生成报告表格、日志汇总、图号索引、图注和一致性抽查，保留原有图号；此命令需要已有的本地坐标缓存。若只下载了精简包且没有缓存，先按 GitHub 的 README 准备输入，再运行 `python -m viz.run_p9`。当前日志已齐，只待 P8 表/图与总表：`projections_complete=true`、`missing_logs=[]`、`consistency_passed=true`，整体 `complete=false`。P8 总表的计划交接日期是 11 月 28 日。

## 数值与解释边界

- P5 Recall 输入为 0–100；P6/P7 准确率输入为 0–1。输出统一百分数一位小数，余弦、间隙等四位小数。CIFAR 的很小标准差因此可能显示为 `0.0%`，原始非零值保存在来源 CSV 和表格 manifest。
- 二维质心距离只描述本张投影，不是原空间模态间隙，不跨图或跨模型比较；高维冻结检查以 P2 表为准。
- 所有上游图登记了可找到的生成脚本/来源；未重跑上游绘图。`gantt.png` 没有单独生成脚本，索引保留“待补”状态。
- 本次没有改动 P1–P8 的共享实现。P9 复用 `eval/p5_common.py` 的输入校验函数，并以来源清单和 README 说明依赖，方便组内协作。

## 交付位置

代码和 README：<https://github.com/Eurekahah/mmrep/pull/2>。代码的辅助模块属于实际运行依赖，随代码保留；生成的检查 JSON、逐单元格 manifest 和运行缓存不再随 PR 提交。

本次云盘交接包：`results/P9_交付_精简版_2026-11-27.zip`。包含 P9 正式 PNG/SVG、报告表格、图注、全组图号索引、一致性记录和日志汇总。其他同学的图和原始结果仍按索引从原目录获取，代码从 GitHub 获取。查看图表无需下载坐标缓存或模型权重。

2026-11-26 的完整包保留为历史备份，不作为本次推荐交接包；其“缺 P3/P6 日志”说明已由本次核查更新。精简包由 `python -m tools.package_p9` 生成，打包时逐文件校验，上传后再次下载比较 SHA-256，校验回执留本地。
