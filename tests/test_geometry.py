import numpy as np
import pytest

from metrics.geometry import (
    alignment_cos, alignment_w, bootstrap_ci, gap_linsep, modality_gap,
    proto_distance, proto_nn_acc, prototypes, uniformity,
)


def test_alignment_metrics():
    # 完全重合时距离为零；循环错配后配对余弦下降，距离增大。
    x = np.eye(4)
    assert alignment_cos(x, x * 3) == pytest.approx(0)
    assert alignment_w(x, x) == pytest.approx(0)
    assert alignment_cos(x, np.roll(x, 1, axis=0)) > 0


def test_uniformity_is_lower_for_spread_spherical_points():
    # 分散球面点的均匀性值应更低；固定种子保证抽样结果可复现。
    rng = np.random.default_rng(42)
    spread = rng.normal(size=(200, 8))
    concentrated = np.tile([1.0] + [0.0] * 7, (200, 1)) + 0.02 * rng.normal(size=(200, 8))
    assert uniformity(spread, max_pairs=1000) < uniformity(concentrated, max_pairs=1000)
    assert uniformity(spread, seed=3) == uniformity(spread, seed=3)


def test_modality_gap_and_linear_probe():
    # 同分布质心间隙为零；质心分离时主指标为正，线性探针可区分模态。
    x = np.eye(8)
    assert modality_gap(x, x) == pytest.approx(0)
    rng = np.random.default_rng(17)
    y = np.tile([1.0] + [0.0] * 7, (8, 1)) + 0.05 * rng.normal(size=(8, 8))
    assert modality_gap(x, y) > 0
    assert gap_linsep(x, y) >= 0.5


def test_gap_ignores_radius_after_unit_norm():
    # 同向向量只差长度；逐行 L2 归一化后两组质心应重合。
    img = np.tile([2.0, 0.0], (6, 1))
    txt = np.tile([0.25, 0.0], (6, 1))
    # 原始质心明显分离，确认间隙归零依赖函数内部的单位化处理。
    assert np.linalg.norm(img.mean(axis=0) - txt.mean(axis=0)) > 1.0
    assert modality_gap(img, txt) == pytest.approx(0)


def test_class_prototypes_and_summaries():
    # 验证低频类别被过滤，且一致的图文原型距离为零、近邻分类正确。
    img = np.array([[1., 0], [.9, .1], [0, 1], [.1, .9], [1, 1]])
    txt = img.copy()
    labels = np.array([0, 0, 1, 1, 2])
    proto = prototypes(img, txt, labels, min_count=2)
    ip = {label: pair[0] for label, pair in proto.items()}
    tp = {label: pair[1] for label, pair in proto.items()}
    assert set(ip) == {0, 1}  # 标签 2 样本数不足，应被过滤。
    distance = proto_distance(ip, tp, {0: 2, 1: 2})
    accuracy = proto_nn_acc(ip, tp, {0: 2, 1: 2})
    assert distance["mean"] == pytest.approx(0)
    assert distance["weighted_mean"] == pytest.approx(0)
    assert accuracy["mean"] == pytest.approx(1)
    assert accuracy["weighted_mean"] == pytest.approx(1)


def test_prototypes_accept_five_captions_per_image():
    img = np.array([[1., 0], [0., 1.]])
    # 五条 caption 按图连续排列；第二张图的文本均值应指向第二类。
    txt = np.array([[1., 0]] * 5 + [[0., 1]] * 5)
    result = prototypes(img, txt, np.array([0, 1]), min_count=1)
    np.testing.assert_allclose(result[0][1], [1, 0])
    np.testing.assert_allclose(result[1][1], [0, 1])


def test_bootstrap_ci_contains_statistic():
    # 均值的 bootstrap 区间应覆盖样本均值。
    x = np.arange(10.0)
    lo, hi = bootstrap_ci(lambda a: a.mean(), x, n=100)
    assert lo <= x.mean() <= hi


def test_bootstrap_uses_shared_indices_for_aligned_arrays():
    # 配对数组的逐行差恒定；共同重采样后统计量仍应保持该差值。
    x = np.arange(20.0)[:, None]
    y = x + 1
    lo, hi = bootstrap_ci(lambda a, b: np.mean(a - b), x, y, n=50)
    assert lo == pytest.approx(-1)
    assert hi == pytest.approx(-1)
