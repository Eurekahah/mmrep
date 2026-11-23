# P5 实验笔记

## 实现和复现

P5 模块位于 `eval/`；运行步骤、指标单位与 CSV 字段说明见 [eval/README.md](../eval/README.md)。

```powershell
conda activate mmrep
python -m pytest tests/test_retrieval.py tests/test_p5_pipeline.py -q --basetemp .pytest_cache/p5
python -m eval.run_p5 --model all --require-complete
```

终端从项目根目录运行。输入为 P2 的 `.npz`，所有原始特征只读，不重新训练或提取。
每次运行自动更新 [实验数字记录](../results/p5_observations.md)，
[覆盖检查](../results/p5_feature_check.json)、[运行记录](../results/p5_run.json)
和 [图注索引](../results/p5_figures.csv)。这些文件是本笔记的数字证据。

## 最终全量实验

P5 的 CLIP 与 SigLIP2 实验现已完成。最终交付采用项目 `results/` 下的全量表格与图，
`results/p5_clip/` 仅保留此前的 CLIP 单模型阶段记录。

- 输入覆盖 2 模型 × 4 设置，每组 16 个条件文件，共 128/128 份真实特征。
- 两个数据集的 val/test、clean/noise/occl 与 Flickr val 的弱、强扰动均已纳入。
- 官方划分、ID 顺序、文本五对一映射、COCO 标签、维度、dtype 和有限数值检查通过。
- 同模型同数据集同 split 的各设置，原图摘要、输入尺寸、模型配置和扰动配置一致。
- 34 项 P5 测试通过。全量运行状态为 `completed`，没有缺失的漂移参照。
- 全量 CLIP 的四个 CSV 与此前单模型阶段逐项核对一致，64 份 CLIP 输入 SHA-256 未变。

| 最终产物 | 行数或数量 |
|---|---:|
| `retrieval_val.csv` | 80 行 |
| `retrieval_test.csv` | 48 行 |
| `retrieval_drop.csv` | 576 行 |
| `drift_cosine.csv` | 448 行 |
| 报告图 | 18 张，各有 PNG/PDF，共 36 个图文件 |

### 检索观察

Flickr clean 的 R@1（百分数，展示保留 1 位）：

| 模型 | 设置 | val i2t | val t2i | test i2t | test t2i |
|---|---|---:|---:|---:|---:|
| CLIP | pretrained | 80.9 | 59.4 | 80.2 | 60.2 |
| CLIP | LiT | 81.6 | 65.8 | 84.1 | 66.8 |
| CLIP | LoRA-r8 | 88.0 | 73.8 | 88.8 | 74.9 |
| CLIP | LoRA-r16 | 88.8 | 73.8 | 88.1 | 74.6 |
| SigLIP2 | pretrained | 87.6 | 73.6 | 86.6 | 73.1 |
| SigLIP2 | LiT | 89.6 | 74.7 | 87.1 | 73.4 |
| SigLIP2 | LoRA-r8 | 91.0 | 78.2 | 91.8 | 78.5 |
| SigLIP2 | LoRA-r16 | 90.4 | 78.2 | 90.7 | 77.8 |

上表用于各模型内部比较不同设置，不能据两个模型的绝对值作损失函数的因果判断。
例如 SigLIP2 在 Flickr val clean 上，LoRA-r8 的 i2t R@1 较其 pretrained 增加 3.4 个百分点，
而 LoRA-r16 增加 2.8 个百分点；其 test 上对应增幅为 5.2 和 4.1 个百分点。
COCO val clean 上 SigLIP2 的 i2t R@1 为 pretrained 72.8%、LiT 72.4%、
LoRA-r8 75.6%、LoRA-r16 75.4%，并非每种微调都在所有数据集与方向上提高检索。

### 表征漂移观察

CLIP 与 SigLIP2 各自的 16 组 LiT/pretrained 图像 raw 数组均逐元素一致，
两个模型的最大绝对差都为 0，对应 same_cond 图像余弦均值为 1。
Flickr val clean 相对各自 pretrained clean 的平均余弦如下（4 位小数）：

| 模型 | 设置 | 图像余弦 | 文本余弦 |
|---|---|---:|---:|
| CLIP | LiT | 1.0000 | 0.7677 |
| CLIP | LoRA-r8 | 0.9276 | 0.9428 |
| CLIP | LoRA-r16 | 0.9268 | 0.9373 |
| SigLIP2 | LiT | 1.0000 | 0.8905 |
| SigLIP2 | LoRA-r8 | 0.9468 | 0.9545 |
| SigLIP2 | LoRA-r16 | 0.9378 | 0.9541 |

表中 LoRA-r16 相对 r8 的余弦差异只描述这个数据集与条件下的观测，
不外推为所有条件下必然成立的秩与漂移关系。完整的 same_cond、pretrained_clean、
self_clean 统计及标准差、分位数见漂移 CSV。

### 扰动与强度观察

Flickr val 的 32 条 R@1 强度曲线（2 模型 × 4 设置 × 2 检索方向 × 2 扰动），
相对 clean 的掉点均按弱、中、强逐档增加。这里报告实际数值的顺序，没有对曲线强制单调拟合。

跨模型使用相对降幅作观察。例如 pretrained 在 Flickr val、中档扰动下：

| 扰动 | 方向 | CLIP 相对下降 | SigLIP2 相对下降 |
|---|---|---:|---:|
| noise | i2t R@1 | 7.3% | 8.6% |
| noise | t2i R@1 | 7.0% | 8.2% |
| occl | i2t R@1 | 25.1% | 24.5% |
| occl | t2i R@1 | 22.3% | 24.9% |

不同扰动与检索方向上的降幅关系并不一致。这些数值交给 P8 与几何指标等证据一同汇总，
不单独判定 SigLIP2 更鲁棒，也不将其解释为 sigmoid 损失的单独效果。

### 最终交接范围

P1 接收 P5 的六个评测脚本、README、两个测试文件和本笔记（共 10 个代码/说明文件）。
P8 接收全量四个 CSV，P9/P10 接收图、图注索引、自动数字记录及本笔记。
特征文件和其他成员的 `lora_params.csv` 不属于 P5 新增交付。
交接包放在作业根目录 `P5_交付/`：代码包用于合并已有仓库，实验结果包保留标准的
`results/`、`docs/` 和 `eval/README.md` 路径，并附包内文件的 SHA-256 清单。

## CLIP 单模型阶段（历史记录）

CLIP 四种设置的 64 份特征到齐后，执行：

```powershell
python -m eval.run_p5 --model clip --require-complete
```

结果与数字记录保存在 [CLIP 结果目录](../results/p5_clip/)，包括检索 val/test 表、
掉点表、漂移表、图表、图注和输入验收记录。这一阶段只评测 CLIP，不作 CLIP 与 SigLIP2
之间的比较。64/64 表示 CLIP 的输入齐全，不表示全组两个模型的实验都已完成。
SigLIP2 特征到齐后可先用 `--model siglip --require-complete` 单独补跑，再用
`--model all --require-complete` 生成统一总表。

### CLIP 阶段实测记录（2026-11-23）

本次使用 64 份真实特征，覆盖 pretrained、LiT、LoRA-r8、LoRA-r16，两个数据集的
val/test、三个主条件和 Flickr val 的四个额外强度条件。官方划分、ID、caption 映射和
输入类型检查通过。输出检索 val 40 行、test 24 行、掉点 288 行、漂移 224 行；
18 张图各有 PNG/PDF，共 36 个图文件。34 项 P5 测试通过。

Flickr clean 的 R@1 如下，单位为百分数，展示值保留 1 位：

| 设置 | val i2t | val t2i | test i2t | test t2i |
|---|---:|---:|---:|---:|
| pretrained | 80.9 | 59.4 | 80.2 | 60.2 |
| LiT | 81.6 | 65.8 | 84.1 | 66.8 |
| LoRA-r8 | 88.0 | 73.8 | 88.8 | 74.9 |
| LoRA-r16 | 88.8 | 73.8 | 88.1 | 74.6 |

Flickr val 上，中档噪声的 i2t R@1 下降依次为 5.9、6.3、6.4、7.2 个百分点；
中档遮挡的下降依次为 20.3、23.8、20.7、23.3 个百分点，顺序与上表一致。
四个设置、两个检索方向的噪声与遮挡 R@1 掉点，在弱、中、强三档上均逐档增加。
这些记录是测量结果，不将其直接等同于模态间隙变化。

相同条件的 LiT 图像特征与 pretrained 的 16 组 raw 数组逐元素一致，最大绝对差为 0；
对应图像余弦均值为 1。Flickr val clean 的逐样本平均余弦如下：

| 设置 | 图像 vs pretrained clean | 文本 vs pretrained clean |
|---|---:|---:|
| LiT | 1.0000 | 0.7677 |
| LoRA-r8 | 0.9276 | 0.9428 |
| LoRA-r16 | 0.9268 | 0.9373 |

数据集之间的检索变化并非完全相同：例如 COCO val clean 的 i2t R@1 从 pretrained
的 59.9% 变为 LiT 的 55.6%，LoRA-r8 与 LoRA-r16 分别为 65.3% 和 65.8%。
因此描述微调效果时必须同时标明数据集、split、检索方向和输入条件。

全部未舍入指标及不同参照下的漂移统计，见 [CLIP 数字记录](../results/p5_clip/p5_observations.md)
和同目录的四个 CSV。该次单模型运行未包含 SigLIP2，最终两个模型的结果见上文全量阶段。

## 指标口径

- 检索：L2 归一化后精确分块检索，两方向均使用全部五条 caption，Recall 单位为百分数。
- 掉点：clean 减扰动的绝对差用百分点；除以 clean 后的相对下降用百分比。
- 漂移：按 ID 严格对应逐样本比较，分别记录 `same_cond`、`pretrained_clean`、`self_clean`。
  均值、总体标准差与 5/50/95 分位数来自余弦相似度，不是向量模长变化。
- val/test 分开；弱、强扰动只在 Flickr val。文本在图像扰动下复用 clean 文本。
- pretrained 同条件自比较只是恒等自检。单独扰动的影响应看 `self_clean`。

## 交接

- P8：两个检索表、掉点表、漂移表，连同 `eval/README.md` 中的单位和字段说明。
- P9/P10：本次 `p5_figures.csv` 索引中的 PNG/PDF 及图注、`p5_observations.md`。
- P1：`eval/` 的 P5 脚本、两个 P5 测试文件与 README。大文件按组内要求走云盘，不进入 Git。

## 后续复现与职责边界

当前全量数字与主要观察已记录在上述最终实验段落和自动数字记录中。
若上游替换了特征，应重新执行全量命令并更新交接包。P5 陈述观测数字，假设的综合判定
由 P8 完成；模态间隙、对齐/均匀性、原型、SVD 等由对应成员计算，不能仅由检索和漂移推断。

CLIP/SigLIP2 的分辨率、预处理、训练数据和嵌入维度不同，跨模型围绕变化量讨论，
不能将模型间观察直接解释为 sigmoid 或 InfoNCE 损失的单独因果作用。
