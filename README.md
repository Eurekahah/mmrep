# mmrep：多模态图文表征学习分析实验（项目一第二组）

CLIP-ViT-B/32 与 SigLIP2-ViT-B/32 的下游微调（LiT、LoRA）与扰动下的共享嵌入空间分析。
不做预训练，只加载公开权重。全组约定见 [docs/CONTRACT.md](docs/CONTRACT.md)，改动前先读它。

## 目录

| 目录 | 内容 | 负责 |
|---|---|---|
| `configs/` | `models.yaml`（两个模型的输入尺寸、预处理、分词）、`perturb.yaml`、`paths.yaml`、`lit.yaml`、`lora.yaml` | P1（lit/lora 由 P3/P4 补充） |
| `splits/` | 划分索引文件 `flickr_split.json`、`coco_split.json`、`cifar100_classes.json`（已提交，种子 42，不要重新划分） | P1 |
| `data/` | 预处理、在线扰动、`get_dataset`、划分脚本 | P1 |
| `tools/` | 日志 `RunLogger`、假特征生成、扰动示例图 | P1 |
| `models/` `train/` `extract/` `eval/` `metrics/` `analysis/` `viz/` | 各自模块，目前只有说明占位 | P2 – P9 |
| `tests/` | 单元测试 | 各人补充 |
| `features/` `ckpt/` `results/` `logs/` | 产出物，不进 git（只放 `.gitkeep`），用网盘交接 | |

## 环境

Python 3.10，PyTorch 2.4.1（cu121），其余版本见 `requirements.txt`。以下二选一。

uv：

```
uv venv --python 3.10 .venv
source .venv/bin/activate                  # Windows: .venv\Scripts\activate
uv pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
uv pip install -r requirements.txt
```

conda：

```
conda create -n mmrep python=3.10 -y && conda activate mmrep
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

没有显卡的同学把 `cu121` 换成 `cpu`；驱动较老换成 `cu118`；RTX 50 系列见 CONTRACT 3.1。

## 数据：各人自行下载

划分文件已经在 `splits/` 里，别人不用重新跑 `prepare_data`。需要原始图片的人（P2 P3 P4 P6）按下面放到 `data_raw/`（已 gitignore），
也可以用环境变量 `MMREP_DATA_ROOT` 指向别的位置。

```
data_raw/
  flickr30k/flickr30k-images/*.jpg            # 4.4GB，HF 数据集 nlphuji/flickr30k 里的 flickr30k-images.zip
  flickr30k/flickr_annotations_30k.csv
  coco/val2017/*.jpg                          # 0.8GB
  coco/annotations/{instances,captions}_val2017.json
  cifar100/cifar-100-python/                  # torchvision 自动生成
```

下载命令（一条命令版：`sh tools/download_data.sh all`，下面是分步写法）：

```
# Flickr30K（国内慢可 export HF_ENDPOINT=https://hf-mirror.com）
huggingface-cli download nlphuji/flickr30k --repo-type dataset --local-dir data_raw/flickr30k
unzip data_raw/flickr30k/flickr30k-images.zip -d data_raw/flickr30k   # 解压后应出现 flickr30k-images/ 目录

# COCO val2017 + 标注
mkdir -p data_raw/coco && cd data_raw/coco
curl -LO http://images.cocodataset.org/zips/val2017.zip
curl -LO http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip val2017.zip && unzip annotations_trainval2017.zip && cd ../..

# CIFAR-100
python -m data.prepare_data cifar100 --root data_raw/cifar100 --download
```

下完后检查图片是否齐全（会重新生成划分并与 `splits/` 比对，应完全相同）：

```
python -m data.prepare_data flickr --csv data_raw/flickr30k/flickr_annotations_30k.csv --images data_raw/flickr30k/flickr30k-images --out /tmp/flickr_check.json
cmp /tmp/flickr_check.json splits/flickr_split.json && echo same
python -m data.prepare_data coco --ann data_raw/coco --images data_raw/coco/val2017 --out /tmp/coco_check.json
cmp /tmp/coco_check.json splits/coco_split.json && echo same
```

划分概况（种子 42）：Flickr30K 取 10000 张，train 8000 / val 1000 / test 1000，每张 5 条 caption；
COCO val2017 中有实例标注且 caption ≥5 条的 4952 张，val 2476 / test 2476，类别取面积最大实例所属的类（80 类）；CIFAR-100 用官方划分。

## 使用

```python
from data.datasets import get_dataset, collate_fn
from torch.utils.data import DataLoader

ds = get_dataset("flickr", "val", cond="noise", model="siglip")    # cond: clean / noise / occl，flickr val 另有 *_lo / *_hi
dl = DataLoader(ds, batch_size=64, collate_fn=collate_fn)
b = next(iter(dl))   # b["image"]: [B,3,256,256]（CLIP 为 224）, b["captions"]: 每张图 5 条, b["idx"], b["id"], b["label"]
```

- 预处理顺序固定：按各模型官方方式 resize（CLIP 短边缩放 + 中心裁剪，SigLIP2 直接缩放）→ 扰动 → 归一化；扰动在线生成，不存盘。
- 同一样本的 clean / noise / occl 一一对应，扰动只由 `(种子 42, 样本序号)` 决定，可复现。
- 尺寸、均值方差、分词参数都从 `configs/models.yaml` 读，不要在代码里写死 224 或 256。
- 没有真实特征时：`python -m tools.make_dummy_features --out features_dummy` 生成同格式的假特征，仅用于调通代码，数值没有意义。
- 记录耗时与显存：`from tools.logger import RunLogger`，见文件头注释。
- 扰动示例图：`python -m tools.make_perturb_examples --n 8 --out docs`。

## 测试

```
pytest -q
```

没有原始图片也能跑（用合成图片）；有 `flickr_annotations_30k.csv` 时（在 `MMREP_DATA_ROOT/flickr30k/` 下）会额外检查划分文件可复现。

## 提交约定

- 每个人只改自己负责的目录，改别人的接口先在群里说。
- 大文件（特征、checkpoint、图片）不进 git，用网盘，按 `features/`、`ckpt/`、`results/` 的目录结构放。
