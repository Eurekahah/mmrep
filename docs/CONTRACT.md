# CONTRACT：全组统一约定

本文件整理自分工文档第三节，是全组的唯一接口依据。11.4 之后冻结，要改的话先在群里说，改完更新本文件并通知受影响的人。

## 3.1 环境

Python 3.10，CUDA 12.1（驱动 ≥ 525）。

```
conda create -n mmrep python=3.10 -y
conda activate mmrep
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
pip install transformers==4.49.0 peft==0.14.0 accelerate==1.2.1 sentencepiece==0.2.0 protobuf safetensors numpy==1.26.4 scikit-learn==1.5.2 umap-learn==0.5.7 pandas==2.2.3 matplotlib==3.9.2 seaborn==0.13.2 openpyxl==3.1.5 pillow tqdm pyyaml
```

用 uv 的话（和上面的 conda 二选一，装的版本完全一样）：

```
# 安装 uv：macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
# 安装 uv：Windows PowerShell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
# 或者已有 pip：pip install uv

uv venv --python 3.10 .venv
source .venv/bin/activate          # Windows：.venv\Scripts\activate
uv pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
uv pip install transformers==4.49.0 peft==0.14.0 accelerate==1.2.1 sentencepiece==0.2.0 protobuf safetensors numpy==1.26.4 scikit-learn==1.5.2 umap-learn==0.5.7 pandas==2.2.3 matplotlib==3.9.2 seaborn==0.13.2 openpyxl==3.1.5 pillow tqdm pyyaml
```

- 两条 `uv pip install` 要分开执行，顺序不能反：torch 那条指定了 PyTorch 的源，其余库走默认的 PyPI；合成一条的话 uv 会因为两个源的包版本冲突报错或装错版本
- 国内网络慢的话，第二条加 `--index-url https://pypi.tuna.tsinghua.edu.cn/simple`（torch 那条不要加，PyTorch 的包只在官方源上）
- 装完验证：`python -c "import torch;print(torch.__version__, torch.cuda.is_available())"`，应显示 `2.4.1+cu121 True`
- 我在沙箱里只做了解析检查（`--dry-run`），确认上面几个版本能解析出一致的依赖集合，没有在带显卡的机器上实际安装并跑通，有报错请反馈
- uv 的虚拟环境放在项目目录的 `.venv` 里，`.venv` 要加进 `.gitignore`，不要提交

- 没有显卡的人（P1、P5、P7、P8、P9、P10）把 torch 那行的链接换成 `https://download.pytorch.org/whl/cpu`，版本号不变
- 驱动较老的人把链接里的 cu121 换成 cu118
- 例外：RTX 50 系列（Blackwell）显卡，torch 2.4.1 不支持，这类卡把 torch 那行换成 `pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128`，其余库版本不变；换了的人在群里说一声，特征与结果不受影响
- 为什么是这些版本：SigLIP2 需要 transformers ≥ 4.49，torch 2.4.1 与 numpy 1.26.4 搭配最稳，peft 0.14.0 与这版 transformers 配套。指导书写的 PyTorch ≥1.12 太旧

## 3.2 数据集与权重

| 内容 | 来源 | 大小 | 谁下载 |
|---|---|---|---|
| Flickr30K 图片包 `flickr30k-images.zip` 与标注 `flickr_annotations_30k.csv` | HuggingFace `nlphuji/flickr30k` | 约 4.4GB | P1、P2、P3、P4 |
| COCO val2017 图片 `val2017.zip` | `http://images.cocodataset.org/zips/val2017.zip` | 约 0.8GB | P1、P2 |
| COCO 标注 `annotations_trainval2017.zip` | `http://images.cocodataset.org/annotations/annotations_trainval2017.zip` | 约 0.25GB | P1、P2、P7 |
| CIFAR-100 | torchvision 自动下载 | 约 0.17GB | P1、P6 |
| CLIP B/32 | HF `openai/clip-vit-base-patch32` | 约 0.6GB | P2、P3、P4、P6 |
| SigLIP2 B/32 | HF `google/siglip2-base-patch32-256` | 约 1.5GB | P2、P3、P4、P6 |

- P3、P4 训练只用 Flickr 的 train 部分，但原始文件要整包下载；P3、P4 不需要 COCO
- P5、P7、P8、P9、P10 不需要任何原始数据和权重，只用特征文件和 csv（P7 额外要 COCO 的类别标签文件，P1 会提供）
- HF 下载慢的话：`export HF_ENDPOINT=https://hf-mirror.com`
- 特征文件、checkpoint、csv 体积不大（全部特征约 1GB，LiT 的 SigLIP checkpoint 约 1.1GB，LoRA 几十 MB），用群文件或网盘传

## 3.3 数据划分（种子 42，P1 产出，各人不要自己另划）

- Flickr30K：从标注文件里的 31014 张（caption 都满 5 条）中抽 10000 张，train 8000 / val 1000 / test 1000，每张 5 条 caption
- COCO：只用 val2017，其中有实例标注且 caption ≥5 条的是 4952 张（另外 48 张没有实例，没法确定类别，剔除），val 2476 / test 2476，不用于训练（实测类别很不均衡：val 里实际出现 68 类，person 占 32%，样本数 ≥20 的只有 37 类；算类别原型时只统计样本数 ≥20 的类，见 P7），用来调脚本和算类别原型；每张图的类别取面积最大的实例所属类别
- CIFAR-100：官方 train 5 万 / test 1 万
- 指标以 val 为主，test 作复核
- 划分文件 `splits/flickr_split.json`、`splits/coco_split.json`：`{"seed":42, "train":[id...], "val":[id...], "test":[id...], "captions":{id:[5 条文本]}, "labels":{id:int}}`（Flickr 的 labels 全为 -1）；`id` 是图片文件名；P1 提供 `data/prepare_data.py`，各人下完原始数据后运行它建立本地的图片索引

## 3.4 模型与输入

- 统一用 HuggingFace transformers：CLIP 用 `CLIPModel`，SigLIP2 用 `SiglipModel`，两者都有 `get_image_features`、`get_text_features`，输出为未归一化的嵌入
- 输入尺寸与 mean/std 写在 `configs/models.yaml`，所有人从这里读，不要写死：CLIP 为 224，SigLIP2 为 256；CLIP 使用 OpenAI 官方 mean/std，SigLIP2 为 mean=std=0.5
- 文本：CLIP 用 `CLIPTokenizer`（max_length 77）；SigLIP2 的 tokenizer 先把文本转小写，padding="max_length"，max_length=64
- 嵌入维度：CLIP 512，SigLIP2 768。几何指标只在同一个模型内部比较，不跨模型直接比向量

分辨率说明（两个模型输入尺寸不同，全组都要知道）：
- CLIP：224×224，7×7=49 个 patch；SigLIP2：256×256，8×8=64 个 patch；两者 patch 都是 32×32 像素，所以单个 patch 内的像素数一样
- 同一张原图要分别预处理成两个尺寸，不要共用一份张量，也不要把一个模型的缓存给另一个模型用。`get_dataset(..., model)` 按 `model` 取各自的尺寸，P1 负责保证这点
- 预处理方式以各模型官方处理器为准，写进 `configs/models.yaml`：CLIP 是“短边缩放到 224（双三次）再中心裁剪”，SigLIP2 是“直接缩放到 256×256（双线性，不保持长宽比、不裁剪）”。这两种处理在非正方形图上会让两边看到的画面不完全一样（CLIP 会裁掉左右或上下边缘，SigLIP2 会拉伸），属于已知差异，写进报告的局限里，不要试图在代码里统一
- 所有尺寸相关的数值（遮挡边长、CIFAR 的放大尺寸、可视化的输入）都从 `models.yaml` 读，代码里不允许出现写死的 224 或 256
- 跨模型比较只比“相对变化”（例如 noise 相对 clean 的 Recall 降幅、间隙增幅），不直接比较两个模型的绝对数值，原因见 `docs/resolution_note.md`（P10 汇总）

## 3.5 训练设置（P3 与 P4 必须一致，只有微调方式不同）

| 项 | 值 |
|---|---|
| 训练集 | Flickr train 8000，每个 epoch 每张图随机取 1 条 caption（`torch.Generator` 种子 42） |
| epoch / batch | 5 epoch，batch 128（OOM 时开 gradient checkpointing，仍不行再降到 64 并在 meta.json 注明） |
| 优化器 | AdamW，weight decay 0.01，betas 默认；LiT lr 1e-4，LoRA lr 5e-4 |
| 学习率策略 | 5% warmup，之后 cosine 衰减到 0 |
| 精度 | fp16 autocast + GradScaler；LoRA 参数保持 fp32 |
| 损失 | 沿用原模型的损失：CLIP 用对称 InfoNCE；SigLIP 用 sigmoid loss（见 `train/losses.py`） |
| logit_scale / logit_bias | 冻结在预训练值，不参与训练（保证改变的只是表征） |
| 保存 | 最后一个 epoch，不按指标挑最优 |
| LiT 可训练范围 | 文本塔 `text_model` 全部 + 文本投影（CLIP 的 `text_projection`）；图像塔 `vision_model` 与 `visual_projection` 全部冻结 |
| LoRA 目标模块 | 正则 `.*(vision_model|text_model).*\.(q_proj|k_proj|v_proj|out_proj)`，alpha = 2r，dropout 0.05，bias="none" |
| 种子 | 42 |

`train/losses.py` 里的函数（P3 写）：
- `info_nce(img, txt, logit_scale)`：img、txt 已 L2 归一化，`logits = exp(logit_scale) * img @ txt.T`，对角为正样本，行方向与列方向的交叉熵取平均
- `sigmoid_loss(img, txt, logit_scale, logit_bias)`：`logits = exp(logit_scale) * img @ txt.T + logit_bias`，标签矩阵 `2*eye(B)-1`，loss = `-sum(logsigmoid(标签 * logits)) / B`

## 3.6 扰动（P1 实现）

- 在 [0,1] 像素空间做，顺序固定：resize 到模型输入尺寸，再加扰动，再归一化
- 噪声：加 N(0, σ²)，clip 到 [0,1]，σ 见下表
- 强度分三档（主设置是中档，所有数据集、所有分析都用中档；弱、强两档只在 Flickr val 上跑，用来看“扰动越强，间隙和对齐变化是否越大”）：

| cond 名 | 噪声 σ | 遮挡边长占输入边长 | 遮挡面积占比 | CLIP 边长 | SigLIP2 边长 |
|---|---|---|---|---|---|
| `noise_lo` / `occl_lo` | 0.05 | 20% | 4% | 45 | 51 |
| `noise` / `occl`（主设置） | 0.1 | 30% | 9% | 67 | 77 |
| `noise_hi` / `occl_hi` | 0.2 | 40% | 16% | 90 | 102 |

- 遮挡：1 个方块，边长为该模型输入边长乘上表中的比例（四舍五入），填 0；每个样本的位置随机，所以不同样本被遮挡的位置本来就各不相同
- 遮挡位置用相对坐标：先用种子抽 `(u, v)`，u、v 取 [0,1] 均匀分布，左上角 = `round(u * (H - 边长))`、`round(v * (W - 边长))`，这样同一个样本在两个模型里遮挡的是原图的同一块区域
- 噪声：每个模型在自己的分辨率上生成（形状不同，没法逐像素相同），种子相同；σ 都是 0.1，两边每个 patch 内像素数相同，所以每个 patch 受到的噪声强度一致
- 随机数：`torch.Generator().manual_seed(42 + 样本序号)`，同一样本在同一个模型的所有设置下得到完全相同的扰动；跨模型只保证遮挡区域相同，噪声是同分布的不同实例
- 在线生成，不存盘；同一样本一次给出 clean/noise/occl 三份，一一对应；弱、强档同理，同一样本、同一种子，只改强度，位置与随机数不变（遮挡位置一致、噪声同一个标准正态场乘不同 σ），保证强度之间只有强度这一个变量

## 3.7 特征文件格式（P2 产出）

路径：`features/{model}/{setting}/{dataset}_{split}_{cond}.npz`

| 取值 | 范围 |
|---|---|
| model | `clip`、`siglip` |
| setting | `pretrained`、`lit`、`lora_r8`、`lora_r16` |
| dataset | `flickr`、`coco` |
| split | `val`、`test` |
| cond | `clean`、`noise`、`occl`（主设置，所有数据集 split 都有）；`noise_lo`、`noise_hi`、`occl_lo`、`occl_hi`（只有 `flickr_val`） |

字段：
- `img_raw`：float32 `[N, D]`，未归一化的图像嵌入，需要时自己做 L2 归一化
- `img_ids`：长度 N 的字符串数组，顺序与划分文件里该 split 的 id 顺序一致；所有文件相同
- clean 文件额外含：`txt_raw` float32 `[5N, D]`，排列为 `txt_raw[5*i + j]` 是第 i 张图的第 j 条 caption；`txt2img` int64 `[5N]`，等于 `i`
- noise、occl 及其弱、强档文件只有图像字段（文本不受图像扰动影响，不重复存）
- COCO 文件额外含：`labels` int64 `[N]`

CIFAR-100：`features_cifar/{model}/{setting}/{train|test}.npz`，含 `img_raw`（float16）和 `labels`；另有 `features_cifar/{model}/{setting}/class_txt.npz`，含 100 条 “a photo of a {类名}” 的文本嵌入 `txt_raw [100, D]`

## 3.8 checkpoint 格式（P3、P4 产出，P2、P6 读取）

- LiT：`ckpt/{model}/lit/trainable_state.pt`：`dict[param_name -> tensor]`，只含 requires_grad 的参数；加载用 `model.load_state_dict(sd, strict=False)`
- LoRA：`ckpt/{model}/lora_r8/adapter/`、`ckpt/{model}/lora_r16/adapter/`：peft 的 `save_pretrained` 目录；加载用 `PeftModel.from_pretrained(base, path)` 后 `merge_and_unload()`
- 每个目录里放 `meta.json`：lr、epoch、batch、种子、可训练参数量、总耗时（秒）、峰值显存（MB）、每个 epoch 的 loss

## 3.9 函数接口

| 接口 | 约定 |
|---|---|
| `load_model(model, setting, ckpt_root="ckpt", device="cuda")`（P2，`models/loader.py`） | 返回 `ModelWrapper`，属性：`embed_dim`、`input_size`、`mean`、`std`；方法：`tokenize(list[str]) -> dict`、`encode_image(pixel_values) -> Tensor[B,D]`、`encode_text(**tokens) -> Tensor[B,D]`，输出均为未归一化；`model` 取 `clip/siglip`，`setting` 取 `pretrained/lit/lora_r8/lora_r16` |
| `get_dataset(name, split, cond, model)`（P1，`data/datasets.py`） | `name` 取 `flickr/coco/cifar100`；每项是 dict：`image`（按 `model` 对应的尺寸与官方预处理 resize、扰动、归一化后的 Tensor，CLIP 为 `[3,224,224]`，SigLIP2 为 `[3,256,256]`）、`captions`（list[str]，CIFAR 为空）、`idx`、`id`、`label`；同时提供 `collate_fn` |
| `perturb(img01, mode, idx)`（P1，`data/perturb.py`） | `img01` 为 `[3,H,W]`、取值 [0,1]；`mode` 取 `clean/noise/noise_lo/noise_hi/occl/occl_lo/occl_hi`，返回同形状 |
| `Logger`（P1，`tools/logger.py`） | 记录耗时和 `torch.cuda.max_memory_allocated`，写入 `logs/` |

没有真实数据时，所有人用 P1 的 `tools/make_dummy_features.py`（造满足 3.7 格式的假特征）开发。

## 3.10 其他选项（已采用的与备选）

| 项目 | 已采用 | 备选 |
|---|---|---|
| Flickr 子集大小 | 10000 张 | 6000 张（4000/1000/1000），显卡慢时 |
| 训练轮数 | 5 epoch | 3 epoch |
| 噪声强度 | 三档 0.05 / 0.1 / 0.2（只有 Flickr val 跑三档，其余只用 0.1） | 只用 0.1，去掉强度这一维 |
| 遮挡边长 | 三档 20% / 30% / 40%（同上） | 只用 30% |
| 模态间隙主指标 | 图文质心距离 | MMD |
| 线性探针 | PyTorch 单层 Linear | sklearn LogisticRegression |
| 降维采样 | 1000 张图 | 500 张 |
| 特征格式 | npz | torch.save 的 pt |

