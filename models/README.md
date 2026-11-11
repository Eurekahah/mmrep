# P2 模型加载器

环境沿用仓库 Python 3.10、PyTorch 2.4.1/cu121、transformers 4.49.0、peft 0.14.0。从仓库根目录运行：

```python
import torch
from models.loader import load_model

wrapper = load_model("clip", "pretrained", device="cuda")
tokens = {k: v.to(wrapper.device) for k, v in wrapper.tokenize(["a photo of a dog"]).items()}
with torch.no_grad():
    text_raw = wrapper.encode_text(**tokens)  # [1,512]，未归一化
```

`load_model(model, setting, ckpt_root="ckpt", device="cuda")` 保持 CONTRACT 3.9 的接口。model 为 `clip/siglip`，setting 为 `pretrained/lit/lora_r8/lora_r16`。可选关键字 `cache_dir`、`local_files_only`、`revision` 分别控制缓存、离线运行及模型版本。CUDA 不可用会报错，CPU 需显式传 `device="cpu"`。

返回 `ModelWrapper`（nn.Module）：

- `embed_dim/input_size/mean/std` 来自 models.yaml，`device` 是模型当前设备。
- `tokenize(list[str])` 返回 CPU 张量，调用方把输入移到模型设备。
- `encode_image(pixel_values)`、`encode_text(**tokens)` 返回原始投影后特征 `[B,D]`。
- `.model` 是原生 HF 模型，`.tokenizer` 是分词器，`.model_config` 是项目配置副本。
- 初始为 eval、float32 参数。encode 方法保留自动求导能力，不内置 no_grad；pretrained 保留原生可训练参数，LoRA 合并前由 PEFT 按推理模式加载。训练方必须通过 `.model` 显式配置冻结与训练模式，推理方负责 no_grad/inference_mode 和 autocast。

## 微调权重

LiT 读取 `ckpt/{model}/lit/trainable_state.pt`，要求文本塔和 CLIP 文本投影的全部参数，拒绝缺失/额外视觉或温度参数/错误形状/非有限值。使用 `weights_only=True`，验证后才执行 `strict=False` 部分加载。

LoRA 读取 `ckpt/{model}/lora_r8/adapter/` 或 `lora_r16/adapter/`，校验秩、权重键/形状、两塔全部 encoder attention q/k/v/out 投影，并执行 `merge_and_unload(safe_merge=True)`。拒绝额外全参数模块、bias、DoRA 等不属于本组的设置；SigLIP 池化头不属于目标。

相对 ckpt_root 以仓库根目录为基准。缺少/错误 checkpoint 明确失败，不会退回 pretrained。小模型回归只验证加载逻辑，P3/P4 正式 checkpoint 到位后需补验。

## 验证

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_model_loader.py -q
.\.venv\Scripts\python.exe -m tests.test_extract_one_batch --model all --device cuda --cache-dir '..\.p2-runtime\hf\hub' --local-files-only --out-dir logs/p2_smoke
```

第二条命令使用真实预训练权重和确定性合成图文，验证加载、推理、五条 caption 对应关系及 NPZ 导出，不计算数据集性能。需提前下载两个模型；首次下载时去掉 `--local-files-only`。`--fp32` 可运行全 float32 验证；`--device cpu` 可在不占用显卡的情况下运行。

模型说明和本次验收见 `docs/model_notes.md`、`docs/p2_delivery.md`，批量提取见 `extract/README.md`。
