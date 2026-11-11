# P2 实验与本地交付记录

## 2026 年 11 月 11 日正式交付进度

第一批 pretrained 已完成并交付至仓库旁的 `P2_本地交接/P2_01_pretrained.zip`。ZIP 内有 32 份真实数据的 NPZ、40 行范数统计、16 张直方图、正式运行日志、数据及 GPU 单批验收记录。每个成员已按包内 `MANIFEST.json` 的 SHA-256 读回核对。第一批 ZIP 的 SHA-256 为 `a6fe8b98bb46fe1704a064446bbfa4aa3e5d5fe1c9a42f97688ee40215fe4225`。

| 模型 | 正式 NPZ | 图像向量（含扰动） | 文本向量 | 图像 batch | 各文件写入耗时之和 | PyTorch 峰值显存 |
|---|---:|---:|---:|---:|---:|---:|
| CLIP | 16 | 24,856 × 512 | 34,760 × 512 | 256 | 201.09 秒 | 1,294.2 MiB |
| SigLIP2 | 16 | 24,856 × 768 | 34,760 × 768 | 192 | 205.38 秒 | 2,071.6 MiB |

运行设备为 RTX 3080，使用 FP16 autocast 和 FP32 参数，导出 dtype 为 float32。两个模型均按起始 batch 完成。正式耗时统计取 `logs/P2_runs.jsonl` 的 32 条 `written` 记录，计入每个文件的图文编码和导出。COCO 的第一次验证性提取另存于 `logs/p2_initial_coco/`。

**数据**。Flickr 的 val/test 各 1,000 张，COCO 各 2,476 张。共 6,952 张图片逐一解码成功，每张五条 caption。四份源文件的 SHA-256 与公布值匹配，三个 ZIP 的全部成员通过 CRC 校验。P1 重新生成的两份划分与仓库文件逐字节相同。每个正式 NPZ 的图片内容摘要与其数据划分验收收据相同，详见 `logs/p2_data_validation.json`。

**特征验收**。每个模型均用真实 COCO 图文在 GPU 上完成单批检查。图像与文本两次推理及与原生编码器比较的最大绝对差均为 0。全量第一批已检查 ID、形状、类型、有限数值、COCO 类别及五句文本映射。COCO 的 12 份文件加入图片字节摘要后重新从真实图片推理，与首轮相应数组逐项相同，见 `logs/p2_coco_repeat.json`。

**统计与测试**。结果位于 `results/p2_stages/pretrained/`，其中 `norm_stats.csv` 有 32 行图像和 8 行文本统计，标准差为总体标准差。该目录的 `norm_stats_validation.json` 包含特征与统计产物的校验值。全仓测试 106 通过、2 项显式启用的模型测试默认跳过、1 项原有 P3 零容差测试失败；P2 的 66 项离线测试全部通过，两项真实权重 GPU 单批已经单独执行。全仓记录见 `logs/p2_delivery_pytest.xml`，P3 基础提交复现记录见 `logs/p2_baseline_test.txt`。

**待上游交接**。第二批 CLIP、第三批 SigLIP2 各需要 LiT、LoRA-r8、LoRA-r16 三组正式训练权重和对应的 `meta.json`。截至本次交付，本地 `ckpt/` 仅含占位文件。每种设置收到后即可先提取 16 份。两批合计还需 96 份特征、120 行范数统计和 32 对 LiT 图像冻结核验。

11 月 10 日完成模型加载器初版、模型说明、提取脚本和 CPU 单批检查。

仓库：Eurekahah/mmrep；基础提交 `23eb9be4d6ff5ab697857150207afb44a57e5e91`；本地分支 `p2/model-loader`。

## 运行环境

11 月 10 日在 CPU 上完成加载器首阶段验收。11 月 11 日用户确认可使用本机 RTX 3080，正式批次采用 GPU FP16 autocast，代码、特征与日志均在本地交付。

## 文件范围

- `models/loader.py`：模型封装、tokenizer、pretrained/LiT/LoRA 加载与 checkpoint 校验。
- `models/README.md`、`docs/model_notes.md`：调用、模型输入输出、参数位置与版本说明。
- `extract/extract_features.py`、`extract/README.md`：提取、NPZ 校验、缓存身份、断点续跑和 smoke 输出隔离。
- `tests/test_model_loader.py`：离线小模型加载、梯度、LiT/LoRA 回归。
- `tests/test_extract_features.py`：特征格式、五 caption 对应、缓存与 128 组合矩阵。
- `tests/test_extract_one_batch.py`：真实预训练权重、合成图文输入的集成检查。
- `tools/logger.py`、`tests/test_p2_logging.py`：增加兼容默认行为的 `track_cuda` 开关，P2 的 CPU 日志完全跳过 CUDA 查询和峰值重置。

## 11 月 10 日 CPU 验证结果

11 月 9 日的加载器初版和模型说明已经完成，并通过真实预训练权重 CPU 验证；特征提取脚本、测试与说明也已实现。机器可读验收记录为 `docs/p2_validation.json`。

| 验证项 | CLIP | SigLIP2 |
|---|---|---|
| 真实公开预训练权重 | 已下载并校验 SHA-256 | 已下载并校验 SHA-256 |
| 设备/精度 | CPU / FP32 | CPU / FP32 |
| 输入尺寸 | 224 x 224 | 256 x 256 |
| 单批图像与首条文本输出 | [2,512] | [2,768] |
| clean 导出 | 2 张图、10 条文本 | 2 张图、10 条文本 |
| noise 导出 | 2 张图，不重复保存文本 | 2 张图，不重复保存文本 |
| 重复推理最大绝对差 | 图像 0，文本 0 | 图像 0，文本 0 |
| 与原生模型 raw 输出最大绝对差 | 图像 0，文本 0 | 图像 0，文本 0 |
| NPZ 保存/重读、五 caption 映射、有限数值及 L2 检查 | 通过 | 通过 |

输入为明确标记的确定性合成 RGB 图片和文字，走 P1 的实际预处理；使用真实模型权重，不是伪造特征。这证明加载、编码及导出流程可运行，不代表 Flickr/COCO 检索性能。输出位于 `logs/p2_smoke_cpu/{clip,siglip}/`，与正式 `features/` 隔离；这些 smoke 日志也不应计作正式数据集实验日志。

### 自动测试

- P2：38 项通过，0 失败（loader 15、extractor 22、CPU 日志隔离 1）。
- 全仓：77 项通过、3 项跳过、1 项原有失败；完整日志 `logs/p2_pytest_output.txt`，JUnit `logs/p2_pytest.xml`。
- 原有失败：`tests/test_train_lit.py::test_training_smoke[siglip]` 要求冻结前后输出零误差，本机 CPU FP32 下最大绝对差为 `2.980232238769531e-7`；视觉参数本身逐位未变。
- 已将未修改的基础提交归档到 `logs/p2_baseline/`，隔离运行同一测试，复现相同失败。证据 `logs/p2_baseline_test.txt`。这是基础版本在当前环境中的零容差断言问题，不是 P2 新回归；P3 代码和测试均未修改。
- 实际模型两项单批检查由 CLI 单独运行并通过；pytest 默认跳过两项需要真实权重的 opt-in 测试，另有一项原始 Flickr 标注复现检查因未下载图片数据/CSV 而跳过。
- `uv pip check`：60 个已安装包兼容；`git diff --check` 通过。完整环境快照 `logs/p2_environment.txt`。

### 环境与权重来源

独立环境位于仓库 `.venv/`：Python 3.10.22、torch 2.4.1+cu121、torchvision 0.19.1+cu121、transformers 4.49.0、peft 0.14.0、numpy 1.26.4。没有更改系统 Python 的包。

Python 运行时、下载工具和缓存位于仓库旁 `../.p2-runtime/`；HF 缓存为 `../.p2-runtime/hf/hub/`，权重不进入 Git。

| 模型 | HF revision | 权重 SHA-256 |
|---|---|---|
| CLIP | `3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268` | `a63082132ba4f97a80bea76823f544493bffa8082296d62d71581a4feff1576f` |
| SigLIP2 | `94dffa8cb1179de3e03f091dbc3917e5d5a9ae84` | `7d241bb3becad218f211f480487f491df4f8c0a472ecf7afdec5615815a301f1` |

## CPU 重跑命令

在仓库根目录的 PowerShell 中运行：

```powershell
$env:CUDA_VISIBLE_DEVICES = '-1'
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
$env:OPENBLAS_NUM_THREADS = '2'
$env:PYTHONUTF8 = '1'
$env:HF_HUB_OFFLINE = '1'
.\.venv\Scripts\python.exe -m pytest tests/test_model_loader.py tests/test_extract_features.py tests/test_p2_logging.py -q
.\.venv\Scripts\python.exe -m tests.test_extract_one_batch --model all --device cpu --batch-size 2 --cache-dir '..\.p2-runtime\hf\hub' --local-files-only --out-dir logs/p2_smoke_cpu
```

已有兼容特征会校验后跳过，模型前向检查仍会运行。若要另存新的一次验收，将 `--out-dir` 改为新的日志目录。

## 正式批次入口

在支持 GPU 的新终端中，从本仓库根目录运行以下 PowerShell 命令。`--cond complete` 包含 Flickr val 的四个额外强度条件。

```powershell
.\.venv\Scripts\python.exe -m extract.extract_features --model all --setting pretrained --cond complete --cache-dir '..\.p2-runtime\hf\hub' --local-files-only
.\.venv\Scripts\python.exe -m tools.package_p2 --stage pretrained
.\.venv\Scripts\python.exe -m tools.package_p2 --stage source
```

P3 和 P4 每移交一组权重，将 `meta.json` 及 `trainable_state.pt` 或 `adapter/` 放在相应 `ckpt/{model}/{setting}/` 内。将提取命令的 model 改为 `clip` 或 `siglip`，setting 改为 `lit`、`lora_r8` 或 `lora_r16`。每组各产生 16 份特征；每模型三组齐全后用 `--stage clip-finetuned` 或 `--stage siglip-finetuned` 统计、核验并打包，全部齐全后执行 `--stage all`。

全部改动保存在本地分支，尚未推送 GitHub，也未上传网盘或发送组内消息。
