# P7 几何指标模块

本目录实现图文嵌入的对齐、均匀性、模态间隙和 COCO 类别原型指标，并提供跑批和绘图工具。指标公式和解释见 [`docs/metric_definition.md`](../docs/metric_definition.md)；特征文件格式见 [`docs/CONTRACT.md`](../docs/CONTRACT.md)。

## 文件说明

- `geometry.py`：单项指标函数，包括 `alignment_cos`、`alignment_w`、`uniformity`、`modality_gap`、`gap_linsep`、`prototypes`、`proto_distance`、`proto_nn_acc` 和通用配对重采样函数 `bootstrap_ci`。
- `run_geometry.py`：读取特征文件并生成正式 CSV。间隙置信区间由此脚本按图像成组 bootstrap 计算。
- `plot_geometry.py`：根据几何指标 CSV 生成四张汇总图。

## 输入数据与统计口径

默认从 `features/<model>/<setting>/` 读取特征。每个条件文件为 `<dataset>_val_<cond>.npz`，至少包含 `img_raw` 和 `txt_raw`：

- `img_raw` 形状为 `[N, D]`；`txt_raw` 形状为 `[5N, D]`。每张图的五条 caption 连续排列，第一条可用 `txt_raw[0::5]` 取得。
- 所有指标在函数内部按行进行 L2 归一化。
- 若文件包含 `txt2img`，跑批脚本会核验其与五条 caption 连续排列的对应关系。COCO 文件还须包含每张图一个标签的 `labels` 数组。
- 正式输出只计算 `val`。clean、noise、occl 条件适用于 Flickr 与 COCO；noise_lo、noise_hi、occl_lo、occl_hi 仅用于 Flickr val。扰动条件改变图像特征，文本特征复用同一组合的 clean 文件。

各指标使用的文本范围不同：对齐和文本均匀性使用每张图第一条 caption；模态间隙和线性探针使用全部五条 caption。COCO 原型先将每张图的五条归一化 caption 向量平均并重新归一化，再按图像类别计算原型；只纳入当前 split 中图像数不少于 20 的类别。

模态间隙的 95% 置信区间默认重采样 200 次、随机种子 42。每轮抽取 N 个图像索引，并同时取这些图像及其各自五条 caption，保持图像与 caption 组对应。`geometry.py` 中的通用 `bootstrap_ci` 用于等长且逐行对齐的数组；正式间隙 CI 使用 `run_geometry.py` 中的专用成组实现。

## 运行

从项目根目录执行：

```bash
python -m metrics.run_geometry --features features --out results --ci-samples 200 --seed 42
python -m metrics.plot_geometry --csv results/geometry_metrics.csv --out results/figs
```

第一条命令生成：

- `results/geometry_metrics.csv`：列为 `dataset, model, setting, cond, align_cos, align_w, unif_img, unif_txt, gap, gap_ci_low, gap_ci_high, gap_linsep`。
- `results/prototype_metrics.csv`：COCO 原型结果，列为 `dataset, model, setting, cond, proto_dist_mean, proto_dist_median, proto_dist_weighted_mean, proto_nn_acc, proto_nn_acc_weighted`。距离和准确率分别报告类别宏平均及按类别图像数加权的结果；中位数为类别中位数。

特征齐全时，几何 CSV 预期 80 行：2 个模型 × 4 种设置 ×（Flickr val 的 7 个条件 + COCO val 的 3 个条件）；原型 CSV 预期 24 行：2 个模型 × 4 种设置 × COCO val 的 3 个条件。缺少 clean 特征的组合会跳过；缺少其它条件文件时只跳过该条件。

第二条命令在 `results/figs/` 写出 `modality_gap_bars.png`、`alignment_uniformity_scatter.png`、`strength_modality_gap.png` 和 `strength_alignment.png`。强度图展示 Flickr val 中噪声与遮挡的 low、medium、high 三档。

## 假特征调试

开发期可生成假特征并检查 CSV 和绘图流程：

```bash
python -m tools.make_dummy_features --out features_dummy
python -m metrics.run_geometry --features features_dummy --out results_dummy
python -m metrics.plot_geometry --csv results_dummy/geometry_metrics.csv --out results_dummy/figs
```

假特征仅用于调试脚本和输出格式，不能作为实验结果引用。
