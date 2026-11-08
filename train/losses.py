import torch
import torch.nn.functional as F


def _check_embeddings(
    img: torch.Tensor,
    txt: torch.Tensor,
) -> None:
    """检查图文特征是否满足同批次、同维度、对角线一一配对的约定。"""
    if img.ndim != 2 or txt.ndim != 2:
        raise ValueError(
            "img and txt must both have shape [B, D]. "
            f"Got {tuple(img.shape)} and {tuple(txt.shape)}."
        )

    if img.shape != txt.shape:
        raise ValueError(
            "img and txt must have identical shapes. "
            f"Got {tuple(img.shape)} and {tuple(txt.shape)}."
        )

    if img.shape[0] == 0:
        raise ValueError("Batch size must be greater than zero.")

    if img.device != txt.device:
        raise ValueError(
            "img and txt must be on the same device."
        )


def info_nce(
    img: torch.Tensor,
    txt: torch.Tensor,
    logit_scale: torch.Tensor,
) -> torch.Tensor:
    """
    CLIP 使用的双向对称 InfoNCE 损失。

    参数：
        img：已做 L2 归一化的图像特征，形状为 [B, D]。
        txt：已做 L2 归一化的文本特征，形状为 [B, D]。
        logit_scale：模型保存的对数缩放系数，即 log(1 / 温度)。

    返回：
        标量损失张量。

    计算公式：
        logits = exp(logit_scale) * img @ txt.T

        loss = (
            CE(logits, labels)
            + CE(logits.T, labels)
        ) / 2

        labels = [0, 1, ..., B-1]
    """
    _check_embeddings(img, txt)

    batch_size = img.shape[0]

    # 图像与全部文本两两计算相似度，得到 [B, B] 的分数矩阵。
    logits = logit_scale.exp() * (img @ txt.T)

    # 第 i 张图与第 i 条文本是正样本，其余位置都是负样本。
    labels = torch.arange(
        batch_size,
        device=img.device,
        dtype=torch.long,
    )

    # 交叉熵使用 fp32 计算，改善数值稳定性。
    logits = logits.float()

    # 分别计算图像检索文本、文本检索图像的交叉熵，再取平均。
    loss_i2t = F.cross_entropy(logits, labels)
    loss_t2i = F.cross_entropy(logits.T, labels)

    return 0.5 * (loss_i2t + loss_t2i)


def sigmoid_loss(
    img: torch.Tensor,
    txt: torch.Tensor,
    logit_scale: torch.Tensor,
    logit_bias: torch.Tensor,
) -> torch.Tensor:
    """
    SigLIP / SigLIP2 使用的图文配对 sigmoid 损失。
    这里只实现图文匹配目标，不包含 SigLIP2 预训练时的辅助目标。

    参数：
        img：已做 L2 归一化的图像特征，形状为 [B, D]。
        txt：已做 L2 归一化的文本特征，形状为 [B, D]。
        logit_scale：模型保存的对数缩放系数，即 log(1 / 温度)。
        logit_bias：模型保存的偏置。

    返回：
        标量损失张量。

    计算公式：
        logits = exp(logit_scale) * img @ txt.T + logit_bias

        labels[i, j] 在 i == j 时为 +1，否则为 -1

        loss = -sum(logsigmoid(labels * logits)) / B
    """
    _check_embeddings(img, txt)

    batch_size = img.shape[0]

    # 缩放余弦相似度并加上偏置；每个图文对独立进行二分类。
    logits = (
        logit_scale.exp() * (img @ txt.T)
        + logit_bias
    )

    # 对角线正样本标签为 +1，非对角线负样本标签为 -1。
    labels = (
        2.0 * torch.eye(
            batch_size,
            device=img.device,
            dtype=logits.dtype,
        )
        - 1.0
    )

    # 使用 logsigmoid 稳定计算 log(sigmoid(x))，避免先求 sigmoid 再取对数。
    signed_logits = (labels * logits).float()

    # 按 SigLIP 定义除以 B：先对每张图的所有配对求和，再对 batch 求平均。
    # 若直接对整个矩阵求 mean，会多除一次 B，改变损失及梯度的尺度。
    loss = -F.logsigmoid(signed_logits).sum() / batch_size

    return loss
