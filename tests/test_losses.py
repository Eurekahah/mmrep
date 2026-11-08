"""P3/P4 共用损失的最小自测：不需要图片、模型权重或网络。"""
import math

import pytest
import torch
import torch.nn.functional as F

from train.losses import info_nce, sigmoid_loss


@pytest.mark.parametrize("model", ["clip", "siglip"])
def test_random_loss_and_gradients_are_finite(model):
    generator = torch.Generator().manual_seed(42)
    img = F.normalize(torch.randn(8, 16, generator=generator), dim=-1).requires_grad_()
    txt = F.normalize(torch.randn(8, 16, generator=generator), dim=-1).requires_grad_()
    scale = torch.tensor(math.log(10.0))
    loss = (info_nce(img, txt, scale) if model == "clip" else
            sigmoid_loss(img, txt, scale, torch.tensor(-5.0)))
    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()
    for features in (img, txt):
        assert features.grad is not None
        assert torch.isfinite(features.grad).all()
        assert features.grad.abs().sum() > 0


def test_info_nce_perfect_match():
    # 正交单位向量：正样本相似度为 1、负样本为 0；足够大的缩放才会使 loss 接近 0。
    features = torch.eye(8)
    scale = torch.tensor(math.log(100.0))
    assert info_nce(features, features, scale).item() < 1e-6
    # 错位配对必须产生更大的损失，避免恒返回 0 也通过测试。
    assert info_nce(features, features.roll(1, dims=0), scale).item() > 1.0


def test_sigmoid_labels_bias_and_reduction():
    features = torch.eye(4)
    # 正样本 logits=5，负样本 logits=-5，所有配对的负对数似然均为 softplus(-5)。
    actual = sigmoid_loss(features, features, torch.tensor(math.log(10.0)),
                          torch.tensor(-5.0))
    expected = features.shape[0] * F.softplus(torch.tensor(-5.0))
    torch.testing.assert_close(actual, expected)
    # 全零 logits 的损失应为 B*log(2)，可识别误除 B²。
    zeros = torch.zeros_like(features)
    actual = sigmoid_loss(zeros, zeros, torch.tensor(0.0), torch.tensor(0.0))
    torch.testing.assert_close(actual, torch.tensor(4 * math.log(2.0)))
