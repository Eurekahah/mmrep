"""生成划分索引文件 splits/*.json 与 CIFAR-100 类别文件。种子固定为 42。

用法（先把原始数据放进 data_root，目录约定见 configs/paths.yaml）：
  python -m data.prepare_data flickr  --csv <flickr_annotations_30k.csv> [--images <flickr30k-images 目录>]
  python -m data.prepare_data coco    --ann <annotations_trainval2017.zip 或其解压目录> [--images <val2017 目录>]
  python -m data.prepare_data cifar100 --root <data_root>/cifar100 [--download]

Flickr 只需要标注 csv 就能生成划分；--images 只用来额外检查图片是否都在。
"""
import argparse
import json
import os
import zipfile

import numpy as np

from .config import get_paths

SEED = 42


def _dump(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)
    print(f"wrote {path} ({os.path.getsize(path) / 1e6:.2f} MB)")


def _check_images(ids, img_dir):
    missing = [i for i in ids if not os.path.exists(os.path.join(img_dir, i))]
    if missing:
        raise FileNotFoundError(f"{len(missing)} 张图片缺失，例如 {missing[:3]}，目录 {img_dir}")
    print(f"图片检查通过：{len(ids)} 张都存在")


def prepare_flickr(csv_path, out_path, n_total=10000, n_val=1000, n_test=1000, images=None):
    import pandas as pd
    df = pd.read_csv(csv_path)
    df = df.sort_values("filename").reset_index(drop=True)   # 排序，保证和下载顺序无关
    caps_all = [json.loads(r) for r in df["raw"]]
    keep = [i for i, c in enumerate(caps_all) if len(c) >= 5]
    print(f"源数据 {len(df)} 张，其中 caption>=5 条的有 {len(keep)} 张")
    rng = np.random.RandomState(SEED)
    pick = [keep[i] for i in rng.permutation(len(keep))[:n_total]]
    n_train = n_total - n_val - n_test
    parts = {"train": pick[:n_train],
             "val": pick[n_train:n_train + n_val],
             "test": pick[n_train + n_val:n_total]}
    ids = {k: [df["filename"][i] for i in v] for k, v in parts.items()}
    all_ids = sum(ids.values(), [])
    if images:
        _check_images(all_ids, images)
    meta = {"dataset": "flickr", "seed": SEED, "source_total": int(len(df)),
            "train": ids["train"], "val": ids["val"], "test": ids["test"],
            "captions": {df["filename"][i]: [c.strip() for c in caps_all[i][:5]] for i in pick},
            "labels": {df["filename"][i]: -1 for i in pick},
            "class_names": []}
    _dump(meta, out_path)
    print({k: len(v) for k, v in ids.items()})


def _read_coco_json(ann, name):
    if os.path.isdir(ann):
        for sub in ("annotations", ""):
            p = os.path.join(ann, sub, name)
            if os.path.exists(p):
                with open(p, "r", encoding="utf-8") as f:
                    return json.load(f)
        raise FileNotFoundError(name)
    with zipfile.ZipFile(ann) as z:
        with z.open(f"annotations/{name}") as f:
            return json.load(f)


def prepare_coco(ann, out_path, images=None):
    inst = _read_coco_json(ann, "instances_val2017.json")
    caps = _read_coco_json(ann, "captions_val2017.json")
    cats = sorted(inst["categories"], key=lambda c: c["id"])
    cat2idx = {c["id"]: i for i, c in enumerate(cats)}
    fname = {im["id"]: im["file_name"] for im in inst["images"]}
    # 每张图的类别 = 单个实例面积最大的那个实例所属类别
    best = {}
    for a in inst["annotations"]:
        k = a["image_id"]
        if k not in best or a["area"] > best[k][0]:
            best[k] = (a["area"], a["category_id"])
    cap_by = {}
    for c in sorted(caps["annotations"], key=lambda c: c["id"]):
        cap_by.setdefault(c["image_id"], []).append(c["caption"].strip().replace("\n", " "))
    ok = sorted(k for k in fname if k in best and len(cap_by.get(k, [])) >= 5)
    print(f"val2017 共 {len(fname)} 张，有实例标注且 caption>=5 条的有 {len(ok)} 张")
    ok_names = [fname[k] for k in ok]
    order = sorted(range(len(ok)), key=lambda i: ok_names[i])
    rng = np.random.RandomState(SEED)
    perm = [order[i] for i in rng.permutation(len(order))]
    half = len(perm) // 2
    val_i, test_i = perm[:half], perm[half:half * 2]
    ids = {"val": [ok_names[i] for i in val_i], "test": [ok_names[i] for i in test_i]}
    if images:
        _check_images(ids["val"] + ids["test"], images)
    sel = val_i + test_i
    meta = {"dataset": "coco", "seed": SEED, "source_total": len(fname),
            "train": [], "val": ids["val"], "test": ids["test"],
            "captions": {ok_names[i]: cap_by[ok[i]][:5] for i in sel},
            "labels": {ok_names[i]: cat2idx[best[ok[i]][1]] for i in sel},
            "class_names": [c["name"] for c in cats]}
    _dump(meta, out_path)
    print({k: len(v) for k, v in ids.items()})


def prepare_cifar100(root, out_path, download=False):
    from torchvision.datasets import CIFAR100
    ds = CIFAR100(root=root, train=False, download=download)
    names = list(ds.classes)
    obj = {"classes": names,
           "prompts": [f"a photo of a {c.replace('_', ' ')}" for c in names],
           "n_train": 50000, "n_test": 10000}
    _dump(obj, out_path)


def main():
    _, splits_dir = get_paths()
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("flickr"); f.add_argument("--csv", required=True); f.add_argument("--images")
    f.add_argument("--out", default=os.path.join(splits_dir, "flickr_split.json"))
    c = sub.add_parser("coco"); c.add_argument("--ann", required=True); c.add_argument("--images")
    c.add_argument("--out", default=os.path.join(splits_dir, "coco_split.json"))
    r = sub.add_parser("cifar100"); r.add_argument("--root", required=True); r.add_argument("--download", action="store_true")
    r.add_argument("--out", default=os.path.join(splits_dir, "cifar100_classes.json"))
    a = ap.parse_args()
    if a.cmd == "flickr":
        prepare_flickr(a.csv, a.out, images=a.images)
    elif a.cmd == "coco":
        prepare_coco(a.ann, a.out, images=a.images)
    else:
        prepare_cifar100(a.root, a.out, download=a.download)


if __name__ == "__main__":
    main()
