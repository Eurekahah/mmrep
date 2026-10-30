import os

import numpy as np

from tools.make_dummy_features import MODELS, SETTINGS, make


def test_dummy_format(tmp_path):
    out = str(tmp_path / "f")
    n = make(out, n=20)
    assert n == 2 * 4 * (2 * 2 * 3 + 4)   # 每模型每设置：2 数据集 x 2 split x 3 主条件 + flickr val 的 4 个强度条件
    for model, dim in MODELS.items():
        ref = None
        for setting in SETTINGS:
            z = np.load(os.path.join(out, model, setting, "flickr_val_clean.npz"))
            assert z["img_raw"].shape == (20, dim) and z["img_raw"].dtype == np.float32
            assert z["txt_raw"].shape == (100, dim) and z["txt2img"].shape == (100,)
            assert (z["txt2img"] == np.repeat(np.arange(20), 5)).all()
            ids = z["img_ids"]
            ref = ids if ref is None else ref
            assert (ids == ref).all()             # 所有设置的样本顺序一致
            for c in ("noise", "occl", "noise_lo", "occl_hi"):
                zc = np.load(os.path.join(out, model, setting, f"flickr_val_{c}.npz"))
                assert "txt_raw" not in zc.files and (zc["img_ids"] == ids).all()
            zc = np.load(os.path.join(out, model, setting, "coco_val_clean.npz"))
            assert zc["labels"].shape == (20,) and zc["labels"].max() < 80
        assert not os.path.exists(os.path.join(out, model, "lit", "coco_val_noise_lo.npz"))   # 强度档只在 flickr val
    # lit 的图像嵌入应与 pretrained 的图像嵌入差别很小（假数据里 lit 图像漂移为 0）
    a = np.load(os.path.join(out, "clip", "pretrained", "flickr_val_clean.npz"))["img_raw"]
    b = np.load(os.path.join(out, "clip", "lit", "flickr_val_clean.npz"))["img_raw"]
    assert np.allclose(a, b)
