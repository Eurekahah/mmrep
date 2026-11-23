# eval/：P5 检索、漂移与 P6 线性探针

P5 的代码已实现；P6 的 `probe_cifar100.py` 已实现，由 P6 维护，见文末对应章节。统一接口见
[CONTRACT](../docs/CONTRACT.md)。P5 只读取特征，不加载模型、不训练、不需要 CUDA。

## 从哪里运行

在项目根目录（含 `eval/`、`features/`、`requirements.txt` 的那层 `mmrep`）打开 PowerShell：

```powershell
conda activate mmrep
python -m pytest tests/test_retrieval.py tests/test_p5_pipeline.py -q --basetemp .pytest_cache/p5
python -m eval.run_p5
```

所有命令使用 `python -m ...`。不要进入 `eval/` 后直接执行 `.py`，否则相对导入会失败。
默认路径由代码文件定位到项目根目录，不依赖写死的个人磁盘路径。
依赖来自项目的 `requirements.txt`；这里主要使用 numpy、pandas、matplotlib、pytest、
以及 scikit-learn 的依赖 threadpoolctl。
测试临时目录放在项目的 `.pytest_cache/p5`，避免 Windows 默认临时路径过长。

## 一键运行包含哪些工作

1. 扫描 `features/{clip,siglip}/{setting}/*.npz`，列出相对完整 128 组合的缺失文件。
2. 校验字段、float32 维度、有限值、非零范数、ID 顺序、五条 caption 映射、COCO 标签；
   默认还核对仓库的 `splits/`。P2 的额外 `_meta` 字段允许存在，并核对其中的路径身份和 count。
3. 用每块 512 行的相似度矩阵评测 i2t/t2i Recall@1/5/10。
4. 计算扰动掉点、三种参照口径的逐样本漂移统计。
5. 输出 PNG/PDF 图、图注索引、实验数字记录以及运行与输入 SHA-256 记录。

允许输入尚未齐全：只有 pretrained 时先跑真实基线，不生成缺失微调设置的虚假结果。
新特征放到相应目录后重新运行同一条命令，重新计算所有已存在的组合并更新 P5 输出。
程序只替换自己生成的结果文件，不修改原始特征或其他同学的结果。绘图索引以本次
`p5_figures.csv` 为准；不要把目录里不在该索引中的旧图当作本次产物。

## 常用命令

只完成 CLIP 的全部实验（四种设置共 64 个文件），结果保存到 `results/p5_clip/`：

```powershell
python -m eval.run_p5 --model clip --require-complete
```

这条命令只读取 CLIP，不评测或检查 SigLIP。`--require-complete` 按所选模型计算完整性：
单模型为 64 份，`--model all`（默认）为 128 份。单模型未指定输出目录时自动使用
`results/p5_clip/` 或 `results/p5_siglip/`，以保留之前的总表。
单模型目录里的 CSV 字段与全量结果一致，可以直接交接并注明模型范围。

后续只补 SigLIP，或合并计算两个模型：

```powershell
python -m eval.run_p5 --model siglip --require-complete
python -m eval.run_p5 --model all --require-complete
```

只做输入检查：

```powershell
python -m eval.run_p5 --validate-only
```

正式运行（默认 CPU BLAS 线程为 2，避免占满电脑）：

```powershell
python -m eval.run_p5
```

特征全部到齐后，要求完整覆盖才允许评测：

```powershell
python -m eval.run_p5 --require-complete
```

只更新数值，不画图：

```powershell
python -m eval.run_p5 --no-plots
```

单独运行各模块：

```powershell
python -m eval.eval_retrieval
python -m eval.drift
python -m eval.plot_p5
```

自定义输入/输出位置，或降低查询分块大小：

```powershell
python -m eval.run_p5 --features-root features --out-dir results/p5_trial --block-size 256 --threads 2
```

假特征调试，与正式结果隔离：

```powershell
python -m tools.make_dummy_features --out features_dummy --n 20
python -m eval.run_p5 --features-root features_dummy --out-dir results/p5_dummy --no-split-check
```

假特征不对应真实 `splits/`，因此显式关闭划分核对。程序禁止在关闭划分核对时直接写
默认的正式 `results/`。假特征输出不能进入报告，不能把假特征拷进正式 `features/`。

## 代码职责

| 文件 | 作用 |
|---|---|
| `retrieval.py` | `recall_at_k(img, txt, txt2img, ks=(1,5,10))`；精确分块检索 |
| `eval_retrieval.py` | 批量检索和掉点长表 |
| `drift.py` | `cosine_summary` 与三种参照下的批量漂移 |
| `plot_p5.py` | 从 CSV 生成报告图，不重新计算特征指标 |
| `run_p5.py` | 一键入口、运行记录、数字笔记 |
| `p5_common.py` | 统一输入校验、路径、输出列名与文件写入 |
| `../tests/test_retrieval.py` | 完美匹配、随机水平、独立全排序参照、分块与异常检查 |
| `../tests/test_p5_pipeline.py` | 完整 128 组合假特征集成、参照口径、部分交付及输入错配检查 |

## 输出、单位与交接

| 路径（相对 results/） | 内容 | 给谁 |
|---|---|---|
| `retrieval_val.csv` / `retrieval_test.csv` | 双向 Recall@1/5/10 | P8 |
| `retrieval_drop.csv` | 各扰动相对同组 clean 的绝对和相对下降 | P8 |
| `drift_cosine.csv` | 两模态的余弦相似度统计 | P8 |
| `figs/*.png` / `figs/*.pdf` | R@1 柱状图、掉点热图、强度折线图、漂移柱状图 | P9、P10 |
| `p5_figures.csv` | 本次图清单与英文图注 | P9、P10 |
| `p5_observations.md` | 自动生成的数字记录，无假设判定 | P8、P10，供补充实验笔记 |
| `p5_feature_check.json` | 输入 SHA-256、形状与缺失列表 | 自检及复现 |
| `p5_drift_missing_references.json` | 缺失的 pretrained 对照，正常可为 `[]` | 自检 |
| `p5_run.json` | 状态、环境、耗时、覆盖率及输出行数 | 自检及复现 |

CSV 使用 UTF-8 BOM，方便 Excel 读取中文路径附近的文件。数值保留足够精度；报告呈现时
Recall 使用百分数保留 1 位，余弦保留 4 位。正式数值不会为了排版提前舍入。

### 检索与掉点

`retrieval_{split}.csv` 的列严格为：

```text
dataset,model,setting,cond,i2t_R1,i2t_R5,i2t_R10,t2i_R1,t2i_R5,t2i_R10
```

- Recall 数值范围为 **0–100**，例如 `78.5` 表示 78.5%。两端特征各自 L2 归一化后做点积，
  不乘模型 logit_scale，也不使用 sigmoid/softmax 改变排名。
- i2t 在 5N 条文本里找 top-k，任意一条正确 caption 命中即成功；t2i 在 N 张图里检索。
  使用全部五条 caption，不只使用第一条。扰动图像配同模型、同设置、同划分的 clean 文本。
- 相似度相同时按候选索引升序打破平局，使分块大小不改变平局规则。
- top-k 超过候选数时以候选数为上限。正常实验样本数远大于 10。
- 掉点长表的每行对应一个 `metric`（例如 `i2t_R1`）：
  `drop_pp = clean_recall - perturbed_recall`，单位为**百分点**；
  `drop_relative_pct = 100 * drop_pp / clean_recall`，单位为**百分比**。
  正数表示下降，负数表示上升，clean 为零时相对降幅留空，不填无穷大。
- 掉点表含 `split`，包括弱、强条件。弱、强条件仅允许 Flickr val。

### 漂移

保留分工要求的列，并追加 `split`、`n`，防止 val/test 混淆并标明样本数：

```text
dataset,model,setting,cond,modality,ref,mean,std,p5,p50,p95,split,n
```

- `modality` 为 `image` 或 `text`。图像 n=N，文本使用全部五条 caption，n=5N。
- `same_cond`：微调特征 vs pretrained 的同条件特征。
- `pretrained_clean`：微调特征 vs pretrained clean。
- `self_clean`：pretrained 扰动图像 vs pretrained clean，只衡量扰动。
- pretrained 自身的 `same_cond` 行是恒等自检：图像在每个条件都保留，文本只在 clean 保留；
  余弦均值精确为 1、标准差为 0。这些自检行不能解释为扰动没有影响，应查看 `self_clean`。
- 文本不受图像扰动影响；微调设置的各条件复用 clean 文本，两个 `ref` 的文本数值相同是正常的。
  绘图仅显示一次 clean 文本，避免造成文本也受图像扰动的误解。
- std 为总体标准差（ddof=0）；p5/p50/p95 为逐样本余弦相似度的分位数。
  相似度越接近 1，方向保留越多；这不是图文对齐或模态间隙本身。

只有两个模型完整的 pretrained 输入时，预期检索 val 20 行、test 12 行、掉点 144 行、
漂移 64 行。全部 128 文件齐全时，分别为 80、48、576、448 行。
只完成一个模型的四种设置（64 份特征）时，分别为 40、24、288、224 行。

## 图表与解释

图内用英文标签以避免缺中文字体，中文实验说明在 `p5_observations.md`。
每个模型单独一个面板；R@1 图以主条件为主，强度折线只用 Flickr val。
缺失组不画成零值。漂移误差条表示总体标准差（ddof=0），不是置信区间。
漂移的不同 ref 分面显示，不能把同条件的恒等自检与扰动参照混为一谈。
跨模型主要比较相对 clean 或 pretrained 的变化；不要把 CLIP/SigLIP2 的差异单独归因于损失。

## 常见问题

- `No module named eval`：终端进入项目根目录，再执行 `python -m eval.run_p5`。
- `No NPZ features`：检查 `.gitkeep` 之外是否有真实 NPZ，以及是否多套了一层 `features/`。
- `official split` / `IDs/order`：特征与划分或不同条件不一致，保留报错路径交 P2 核对，
  不要为了通过检查而关闭真实数据的划分核对。
- `Missing clean`：补齐对应 clean 文件；扰动 NPZ 没有自己的文本。
- `Coverage: 32/128`：目前只收到两个模型的 pretrained 特征，是正常的部分交付。
- 命令行显示 `P5 completed` 后，查看 `results/p5_run.json` 的状态及覆盖率。

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
