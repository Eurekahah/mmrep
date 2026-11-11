"""几何指标：图文对齐、均匀性、模态间隙与类别原型。"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def _unit(x):
    # 所有几何指标统一在单位球面上计算，避免向量长度影响结果
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] == 0 or x.shape[1] == 0:
        raise ValueError("features must be a non-empty [N, D] matrix")
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, 1e-12)


def alignment_cos(img, txt1):
    # txt1是每张图对应的第一条caption特征
    a, b = _unit(img), _unit(txt1)
    if a.shape != b.shape:
        raise ValueError("img and txt1 must have the same shape")
    return float(1.0 - np.sum(a * b, axis=1).mean())


def alignment_w(img, txt1, alpha=2):
    # Wang–Isola对齐距离：配对图文的欧氏距离alpha次方均值
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    a, b = _unit(img), _unit(txt1)
    if a.shape != b.shape:
        raise ValueError("img and txt1 must have the same shape")
    return float((np.linalg.norm(a - b, axis=1) ** alpha).mean())


def uniformity(x, t=2, max_pairs=5000, seed=42):
    # 样本对过多时用固定种子去重子采样（对成对距离的无偏估计，报告里应注明）
    # 未超过总对数时用triu_indices全量，均避免构造完整 N×N 距离矩阵
    x = _unit(x)
    if t <= 0 or max_pairs <= 0:
        raise ValueError("t and max_pairs must be positive")
    n = len(x)
    if n < 2:
        raise ValueError("uniformity requires at least two samples")
    total = n * (n - 1) // 2
    rng = np.random.default_rng(seed)
    if total <= max_pairs:
        i, j = np.triu_indices(n, 1)
    else:
        # 在不分配 O(N²) 索引数组的前提下，对无序样本对做去重抽样
        keys = set()
        while len(keys) < max_pairs:
            batch = rng.integers(0, n, size=(max_pairs * 2, 2))
            for a, b in batch:
                if a != b:
                    keys.add((min(int(a), int(b)), max(int(a), int(b))))
                    if len(keys) == max_pairs:
                        break
        pairs = np.fromiter((v for pair in keys for v in pair), dtype=np.int64).reshape(-1, 2)
        i, j = pairs.T
    d2 = np.sum((x[i] - x[j]) ** 2, axis=1)
    return float(np.log(np.exp(-t * d2).mean()))


def modality_gap(img, txt_all):
    # 主间隙指标：图像与全部文本嵌入的质心距离
    a, b = _unit(img), _unit(txt_all)
    if a.shape[1] != b.shape[1]:
        raise ValueError("image and text embedding dimensions must match")
    return float(np.linalg.norm(a.mean(axis=0) - b.mean(axis=0)))


def gap_linsep(img, txt_all):
    # 辅助指标：交叉验证评估线性分类器区分图像/文本模态的能力
    a, b = _unit(img), _unit(txt_all)
    if a.shape[1] != b.shape[1]:
        raise ValueError("image and text embedding dimensions must match")
    X = np.concatenate((a, b))
    y = np.r_[np.zeros(len(a), dtype=int), np.ones(len(b), dtype=int)]
    counts = np.bincount(y)
    folds = min(5, int(counts.min()))
    if folds < 2:
        raise ValueError("at least two samples per modality are required")
    cv = StratifiedKFold(folds, shuffle=True, random_state=42)
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=42))
    return float(cross_val_score(clf, X, y, cv=cv, scoring="accuracy").mean())

def prototypes(img, txt, labels, min_count=20):
    """按类别计算图像和文本原型。

    ``txt`` 可为每图一条向量 ``[N, D]``，也可为按图连续排列的五条 caption ``[5N, D]``。
    五条 caption 会先按图求均值并归一化，再用于计算类别文本原型。
    返回类别到（图像原型、文本原型）的映射；低于 ``min_count`` 的类别会跳过。
    本函数不返回类别频数；加权统计时需由调用方统计频数并传给汇总函数。
    """
    img = _unit(img)
    txt = _unit(txt)
    labels = np.asarray(labels)
    if len(labels) != len(img):
        raise ValueError("labels must have one entry per image")
    if len(txt) == 5 * len(img):
        # 每张图的五条 caption 连续存放；先在图像级聚合，再按类别求均值。
        txt = txt.reshape(len(img), 5, -1).mean(axis=1)
        txt = _unit(txt)
    elif len(txt) != len(img):
        raise ValueError("txt must contain one embedding per image or five consecutive captions per image")
    # 忽略样本过少的类别，降低稀疏类别原型的不稳定性。
    result = {}
    for label in np.unique(labels):
        idx = labels == label
        if idx.sum() < min_count:
            continue
        def mean_unit(v):
            m = v[idx].mean(axis=0)
            return m / max(np.linalg.norm(m), 1e-12)
        result[label.item() if hasattr(label, "item") else label] = (mean_unit(img), mean_unit(txt))
    return result


def proto_distance(img_prototypes, txt_prototypes, sample_counts=None):
    """汇总同类别图像与文本原型的余弦距离。

    参数是类别到原型向量的映射；只统计两个映射中共有的类别。返回 ``mean``（类别宏平均）
    和 ``median``。若提供类别样本数映射 ``sample_counts``，还返回按样本数加权的 ``weighted_mean``。
    距离定义为 ``1 - cosine``，越小表示图文原型越接近。
    """
    common = [k for k in img_prototypes if k in txt_prototypes]
    if not common:
        raise ValueError("no matching class prototypes")
    # 距离为 1-cos；可选按每类图像样本数计算加权均值。
    d = np.array([1 - np.dot(_unit(np.asarray(img_prototypes[k])[None])[0],
                             _unit(np.asarray(txt_prototypes[k])[None])[0]) for k in common])
    out = {"mean": float(d.mean()), "median": float(np.median(d))}
    if sample_counts is not None:
        w = np.array([sample_counts[k] for k in common], dtype=float)
        out["weighted_mean"] = float(np.average(d, weights=w))
    return out


def proto_nn_acc(img_prototypes, txt_prototypes, sample_counts=None):
    """计算图像类别原型对文本类别原型的最近邻分类准确率。

    每个共有类别的图像原型在文本原型中按余弦相似度选最近类别。返回 ``mean``（类别宏平均准确率）；
    若提供类别样本数映射 ``sample_counts``，还返回按样本数加权的 ``weighted_mean``。值越高越好。
    """
    # 用每个图像原型检索最相似的文本原型，统计类别匹配率。
    common = [k for k in img_prototypes if k in txt_prototypes]
    if not common:
        raise ValueError("no matching class prototypes")
    T = np.stack([_unit(np.asarray(txt_prototypes[k])[None])[0] for k in common])
    I = np.stack([_unit(np.asarray(img_prototypes[k])[None])[0] for k in common])
    pred = np.argmax(I @ T.T, axis=1)
    correct = (pred == np.arange(len(common))).astype(float)
    out = {"mean": float(correct.mean())}
    if sample_counts is not None:
        out["weighted_mean"] = float(np.average(correct, weights=[sample_counts[k] for k in common]))
    return out

def bootstrap_ci(fn, *args, n=200, seed=42, confidence=0.95, **kwargs):
    """沿样本轴有放回重采样，返回百分位置信区间。

    多个输入数组须等长且逐行对应；每轮共用同一组索引，以保留配对关系。
    原始五条 caption 特征为 ``[5N, D]``，不能直接与图像 ``[N, D]`` 一起传入；
    应先转为每图一条文本表示。区间端点由 ``confidence`` 指定的尾部分位数确定。
    """
    if n < 1 or not 0 < confidence < 1:
        raise ValueError("n must be positive and confidence must be in (0, 1)")
    arrays = [np.asarray(a) for a in args]
    if not arrays or arrays[0].ndim < 1:
        raise ValueError("at least one sample array is required")
    size = len(arrays[0])
    if any(len(a) != size for a in arrays):
        raise ValueError("bootstrap sample arrays must have equal lengths")
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n):
        # 同一组索引用于全部数组，保留配对样本的对应关系。
        idx = rng.integers(0, size, size=size)
        values.append(fn(*(a[idx] for a in arrays), **kwargs))
    tail = (1 - confidence) / 2
    return float(np.quantile(values, tail)), float(np.quantile(values, 1 - tail))
