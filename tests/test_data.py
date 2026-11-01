import json
import os

import numpy as np
import pytest
import torch
from PIL import Image

from data import perturb as P
from data.config import REPO_ROOT, load_models_cfg
from data.datasets import collate_fn, get_dataset
from data.preprocess import load_resized_01, normalize

MODELS = ["clip", "siglip"]
SIZE = {"clip": 224, "siglip": 256}


@pytest.fixture(scope="module")
def mini(tmp_path_factory):
    """造一个迷你 flickr 数据集：6 张不同长宽比的合成图 + 对应划分文件。"""
    root = tmp_path_factory.mktemp("raw")
    splits = tmp_path_factory.mktemp("splits")
    img_dir = root / "flickr30k" / "flickr30k-images"
    img_dir.mkdir(parents=True)
    rs = np.random.RandomState(0)
    shapes = [(400, 200), (200, 400), (300, 300), (640, 480), (480, 640), (256, 256)]
    ids = []
    for k, (w, h) in enumerate(shapes):
        arr = (rs.rand(h, w, 3) * 255).astype(np.uint8)
        Image.fromarray(arr).save(img_dir / f"img{k}.jpg", quality=95)
        ids.append(f"img{k}.jpg")
    meta = {"dataset": "flickr", "seed": 42, "train": ids[:2], "val": ids[2:4], "test": ids[4:],
            "captions": {i: [f"caption {i} {j}" for j in range(5)] for i in ids},
            "labels": {i: -1 for i in ids}, "class_names": []}
    (splits / "flickr_split.json").write_text(json.dumps(meta))
    return str(root), str(splits), ids


def test_input_size_from_config():
    for m in MODELS:
        assert load_models_cfg()[m]["input_size"] == SIZE[m]


@pytest.mark.parametrize("model", MODELS)
def test_shape_and_dtype(mini, model):
    root, splits, ids = mini
    ds = get_dataset("flickr", "val", "clean", model, data_root=root, splits_dir=splits)
    x = ds[0]["image"]
    assert x.shape == (3, SIZE[model], SIZE[model]) and x.dtype == torch.float32
    b = collate_fn([ds[0], ds[1]])
    assert b["image"].shape == (2, 3, SIZE[model], SIZE[model]) and len(b["captions"][0]) == 5


def test_clip_crops_siglip_squashes(tmp_path):
    # 400x200 的横图：左侧 1/4 涂红。CLIP 中心裁剪会把它裁掉，SigLIP 直接压缩会保留。
    arr = np.full((200, 400, 3), 128, np.uint8)
    arr[:, :100] = [255, 0, 0]
    img = Image.fromarray(arr)
    c = load_resized_01(img, "clip")
    s = load_resized_01(img, "siglip")
    assert c.shape[-1] == 224 and s.shape[-1] == 256
    assert c[0, :, 0].mean() < 0.7 and c[1, :, 0].mean() > 0.4          # 左边缘已不是红色
    assert s[0, :, 0].mean() > 0.95 and s[1, :, 0].mean() < 0.05       # 左边缘仍是红色


def test_siglip_interpolation_is_bilinear():
    assert load_models_cfg()["siglip"]["resize"]["interpolation"] == "bilinear"
    assert load_models_cfg()["clip"]["resize"]["interpolation"] == "bicubic"


def test_normalize_stats():
    x = torch.full((3, 4, 4), 0.5)
    assert torch.allclose(normalize(x, "siglip"), torch.zeros(3, 4, 4))
    c = load_models_cfg()["clip"]
    y = normalize(x, "clip")
    assert abs(float(y[0, 0, 0]) - (0.5 - c["mean"][0]) / c["std"][0]) < 1e-6


# ---------------- 扰动 ----------------
def test_clean_unchanged():
    x = torch.rand(3, 224, 224)
    assert torch.equal(P.perturb(x, "clean", 3), x)


@pytest.mark.parametrize("mode", P.ALL_MODES)
def test_perturb_deterministic_and_nonmutating(mode):
    x = torch.rand(3, 224, 224)
    x0 = x.clone()
    a, b = P.perturb(x, mode, 7), P.perturb(x, mode, 7)
    assert torch.equal(a, b) and torch.equal(x, x0) and a.shape == x.shape
    if mode != "clean":
        assert not torch.equal(P.perturb(x, mode, 8), a)


def test_noise_range_and_scaling():
    x = torch.full((3, 224, 224), 0.5)
    outs = {m: P.perturb(x, m, 11) for m in ("noise_lo", "noise", "noise_hi")}
    for o in outs.values():
        assert float(o.min()) >= 0 and float(o.max()) <= 1
    f = P.noise_field((3, 224, 224), 11)
    for m, sig in (("noise_lo", 0.05), ("noise", 0.1), ("noise_hi", 0.2)):
        ok = (f * sig).abs() < 0.49          # 没被 clip 的像素
        assert torch.allclose(outs[m][ok], (0.5 + f * sig)[ok], atol=1e-6)
    assert abs(float((f * 0.1).std()) - 0.1) < 0.005


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("mode,area", [("occl_lo", 0.10), ("occl", 0.25), ("occl_hi", 0.50)])
def test_occl_area_aspect_fill(model, mode, area):
    s = SIZE[model]
    mean = load_models_cfg()[model]["mean"]
    ratios = []
    for idx in range(60):
        x = torch.rand(3, s, s) * 0.5            # 取值 <0.5，保证和填充色可区分
        out = P.perturb(x, mode, idx, fill=mean)
        changed = (out != x).any(0)
        ys, xs = torch.where(changed)
        bh, bw = int(ys.max() - ys.min()) + 1, int(xs.max() - xs.min()) + 1
        top, left, h2, w2 = P.occl_box(s, s, area, idx)
        assert (bh, bw) == (h2, w2) and (int(ys.min()), int(xs.min())) == (top, left)   # 变化区域就是返回的矩形
        assert abs(h2 * w2 / (s * s) - area) < 0.02                                       # 面积比准确（取整误差内）
        assert 0.25 <= (w2 / h2) <= 4.0 or area >= 0.5                                  # 长宽比在范围内（大面积会被画面截断）
        box = out[:, top:top + h2, left:left + w2]
        assert torch.allclose(box, torch.tensor(mean).view(3, 1, 1).expand_as(box))      # 填的是均值像素
        assert torch.equal(out[:, ~changed], x[:, ~changed])                              # 其余像素不变
        ratios.append(w2 / h2)
    assert max(ratios) / min(ratios) > 3          # 长宽比确实有变化，不是固定正方形


def test_occl_fill_gives_zero_after_normalize():
    for m in MODELS:
        s = SIZE[m]
        x = torch.rand(3, s, s)
        out = normalize(P.perturb(x, "occl", 3, fill=load_models_cfg()[m]["mean"]), m)
        top, left, bh, bw = P.occl_box(s, s, 0.25, 3)
        assert float(out[:, top:top + bh, left:left + bw].abs().max()) < 1e-5


def test_occl_same_relative_region_across_models():
    for idx in range(50):
        for area in (0.10, 0.25, 0.50):
            reg = {}
            for m in MODELS:
                s = SIZE[m]
                t, l, bh, bw = P.occl_box(s, s, area, idx)
                reg[m] = (t / s, l / s, (t + bh) / s, (l + bw) / s)
            assert max(abs(a - b) for a, b in zip(reg["clip"], reg["siglip"])) < 0.02


def test_occl_levels_share_shape_and_position():
    for idx in range(30):
        r1, u1, v1 = P.occl_params(idx)
        assert (r1, u1, v1) == P.occl_params(idx)
        lo = P.occl_box(224, 224, 0.10, idx)
        hi = P.occl_box(224, 224, 0.50, idx)
        assert lo[2] * lo[3] < hi[2] * hi[3]


# ---------------- 数据集 ----------------
def test_three_conds_one_to_one(mini):
    root, splits, ids = mini
    items = {c: get_dataset("flickr", "val", c, "siglip", data_root=root, splits_dir=splits)[1]
             for c in ("clean", "noise", "occl")}
    assert len({(v["id"], v["idx"], tuple(v["captions"])) for v in items.values()}) == 1
    assert not torch.equal(items["clean"]["image"], items["noise"]["image"])
    assert not torch.equal(items["clean"]["image"], items["occl"]["image"])
    assert items["noise"]["image"].shape == items["clean"]["image"].shape


def test_strength_scope_restricted(mini):
    root, splits, _ = mini
    get_dataset("flickr", "val", "noise_lo", "clip", data_root=root, splits_dir=splits)
    with pytest.raises(ValueError):
        get_dataset("flickr", "test", "noise_lo", "clip", data_root=root, splits_dir=splits)
    with pytest.raises(ValueError):
        get_dataset("coco", "val", "occl_hi", "clip", data_root=root, splits_dir=splits)


def test_no_image_written_to_disk(mini):
    root, splits, _ = mini
    before = sorted(os.listdir(os.path.join(root, "flickr30k", "flickr30k-images")))
    ds = get_dataset("flickr", "val", "noise", "clip", data_root=root, splits_dir=splits)
    _ = [ds[i] for i in range(len(ds))]
    assert before == sorted(os.listdir(os.path.join(root, "flickr30k", "flickr30k-images")))


# ---------------- 真实划分文件（存在才测） ----------------
SPL = os.path.join(REPO_ROOT, "splits")


@pytest.mark.skipif(not os.path.exists(os.path.join(SPL, "flickr_split.json")), reason="no flickr split")
def test_flickr_split_real():
    m = json.load(open(os.path.join(SPL, "flickr_split.json"), encoding="utf-8"))
    tr, va, te = map(set, (m["train"], m["val"], m["test"]))
    assert (len(tr), len(va), len(te)) == (8000, 1000, 1000)
    assert not (tr & va) and not (tr & te) and not (va & te)
    assert len(va) == len(te) and m["seed"] == 42
    assert all(len(m["captions"][i]) == 5 for i in tr | va | te)
    assert set(m["captions"]) == tr | va | te and set(m["labels"].values()) == {-1}


@pytest.mark.skipif(not os.path.exists(os.path.join(SPL, "coco_split.json")), reason="no coco split")
def test_coco_split_real():
    m = json.load(open(os.path.join(SPL, "coco_split.json"), encoding="utf-8"))
    va, te = set(m["val"]), set(m["test"])
    assert not (va & te) and abs(len(va) - len(te)) <= 1 and m["train"] == []
    assert len(m["class_names"]) == 80
    assert all(len(m["captions"][i]) == 5 and 0 <= m["labels"][i] < 80 for i in va | te)


def test_split_reproducible(tmp_path):
    csv = os.path.join(os.environ.get("MMREP_DATA_ROOT", ""), "flickr30k", "flickr_annotations_30k.csv")
    if not os.path.exists(csv):
        pytest.skip("no flickr csv")
    from data.prepare_data import prepare_flickr
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    prepare_flickr(csv, str(a)); prepare_flickr(csv, str(b))
    assert a.read_bytes() == b.read_bytes()
    if os.path.exists(os.path.join(SPL, "flickr_split.json")):
        assert a.read_bytes() == open(os.path.join(SPL, "flickr_split.json"), "rb").read()
