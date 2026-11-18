# metrics/

负责人：P7

模块：

- `geometry.py`：对齐、均匀性、模态间隙、类别原型和 bootstrap 置信区间函数。
- `run_geometry.py`：遍历特征文件，输出几何指标和 COCO 原型指标 CSV。

特征格式与组合约定见 [docs/CONTRACT.md](../docs/CONTRACT.md)。运行：

```bash
python -m metrics.run_geometry --features features --out results
```

脚本生成 `results/geometry_metrics.csv` 和 `results/prototype_metrics.csv`。弱、强扰动条件只读取 Flickr val；COCO 原型只统计样本数不少于 20 的类别。模态间隙置信区间通过分别对图像和 caption 向量独立 bootstrap 估计。缺少特征的组合会跳过。

开发期可用 `python -m tools.make_dummy_features --out features_dummy` 生成假特征，再运行：

```bash
python -m metrics.run_geometry --features features_dummy --out results_dummy
```

假特征仅用于检查脚本和输出格式，不能作为实验结果引用。
