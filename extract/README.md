# 图文特征提取（P2）

`extract_features.py` 使用 P1 的数据模块和 P2 的 `models.loader`，提取 Flickr30K / COCO 的图文原始嵌入。CIFAR-100 的 `extract_cifar.py` 由 P6 实现。本模块不训练模型，也不下载图片数据。

从仓库根目录运行；Windows PowerShell 与 Linux 均使用 `python -m extract.extract_features`。先按总 README 安装环境，准备仓库规定的图片目录和 `splits/`。模型权重可通过 Hugging Face 自动取得，也可用 `--cache-dir` 指定已有缓存，并加 `--local-files-only` 禁止联网。LiT/LoRA 还需要 P3/P4 的 checkpoint。

## 单批检查与正式提取

```powershell
# Windows，从仓库根目录运行。仅需 COCO val 划分首两张图片和缓存的预训练权重。
.\.venv\Scripts\python.exe -m tests.test_extract_one_batch --model all --device cuda --batch-size 2 --dataset coco --split val --local-files-only --out-dir logs/p2_real_smoke

# 两模型 pretrained 主条件及 Flickr-val 弱强档：合计 32 份
python -m extract.extract_features --model all --setting pretrained --dataset all --split all --cond complete

# checkpoint 齐全后，所有模型/设置/数据集/划分：合计 128 份
python -m extract.extract_features --model all --setting all --dataset all --split all --cond complete

# 只列输出计划，不加载模型/图片、不写特征
python -m extract.extract_features --model all --setting all --cond complete --dry-run
```

Linux 或 Git Bash 可用 `sh run_extract_all.sh --setting pretrained` 批量生成两模型共 32 份预训练特征。六个微调 checkpoint 齐备后用 `sh run_extract_all.sh` 生成完整 128 份。脚本优先使用仓库虚拟环境，支持通过 `MMREP_PYTHON` 指定解释器。

真实单批验收逐个加载模型，通过 P1 数据接口读取选定 split 的首两张图像和十条原始 caption，检查原生编码器与 wrapper 输出、重复推理、有限值和 L2 范数。它导出并重读 clean、noise 文件，检查 `img_ids`、五条 caption 排列、`txt2img`、COCO 标签和特征数值。上例的报告为 `logs/p2_real_smoke/coco_val/{clip,siglip}/report.json`。权重缓存或图片在其他目录时，分别追加 `--cache-dir <HF缓存>` 和 `--data-root <图片根目录>`。选 `--dataset flickr` 可用 Flickr 图片执行相同检查。

`--cond all` 只包含 `clean/noise/occl` 三种主条件（全模型、全设置时为 96 份）。`--cond strength` 只包含 Flickr-val 的 `noise_lo/noise_hi/occl_lo/occl_hi`（32 份），`--cond complete` 合并两者（128 份）。`strength` 搭配 `--dataset all --split all` 会只选择 Flickr-val；若明确只选 COCO 或 test，则报错。

CUDA 默认图像 batch 为 CLIP 256、SigLIP2 192，文本按相同上限分块。CPU 默认 16。可用 `--batch-size` 和 `--text-batch-size` 指定上限。Windows 默认 `--num-workers 0`。CUDA 提取使用 FP32 参数和 FP16 autocast，保存为 float32。CPU 选 `--device cpu`。CUDA 的 FP32 对照选 `--no-amp`。

CUDA 前向发生显存不足时，脚本释放失败批次，将图像、文本 batch 分别减半，从同一文件的第一张图重新提取，直到成功或 batch 为 1 仍失败。该模型在同一次批跑中沿用成功的上限。LiT 优先读取对应 pretrained 文件记录的实际 batch；显式指定的上限优先。这样可以用相同精度和批大小核对 LiT 的冻结图像塔。

其他路径参数：`--data-root`、`--splits-dir`、`--ckpt-root`、`--cache-dir`、`--out`、`--log-dir`。`--revision` 可固定模型仓库版本；同一次多模型运行传入该值时，两个模型仓库都必须存在这个 revision。

## 输出与验收

正式默认路径为仓库根目录下的 `features/{model}/{setting}/{dataset}_{split}_{cond}.npz`。保存字段：

- 所有条件：`img_raw` float32 `[N,D]`、`img_ids` Unicode `[N]`。
- 仅 clean：`txt_raw` float32 `[5N,D]`、`txt2img` int64 `[5N]`。文本顺序为第 i 张图片的第 0–4 条 caption，`txt2img = repeat(arange(N), 5)`。
- COCO 额外保存 `labels` int64 `[N]`。
- `_meta` 是额外的标量 Unicode JSON，记录模型、设置、划分、输入和配置摘要、精度、模型 revision、checkpoint 哈希。微调设置还记录 P3/P4 的 `meta.json` SHA256。CUDA 或 CPU 的 `device`、`device_name`、最大实际图像和文本 microbatch（`batch_size`、`text_batch_size`）与 OOM 重试次数也随文件保存。仅图像的扰动条件中 `text_batch_size` 为 null。

嵌入不做 L2 归一化。`img_ids` 遵循 `dataset.ids`；一致性比较限于**相同数据集、相同划分**内的不同模型、设置、扰动，不跨 Flickr/COCO 或 val/test 比较 ID 列表。COCO 数量读取仓库划分文件，不写死 2500 或 2476。

`--limit N` 始终视为检查运行：默认写到 `features_smoke/limit_N/`，后缀为 `.limitN.npz`；指定 `--out` 时也会加入 `limit_N/` 和该后缀，且禁止输出到正式 `features/` 目录。取消 `--limit` 才生成正式文件。

已有文件先核验字段、形状、dtype、有限数值、图像顺序、COCO 标签、caption 映射与输入/权重身份，全部通过才跳过。损坏、不完整、旧格式、模型版本或精度不匹配会报错并原样保留，请核查后移到其他目录再重跑；没有强制覆盖开关。新文件先写同目录临时文件、读回校验，再原子替换到目标名称，中途失败不会留下半个正式文件。

对于 P1 的真实图片，提取和断点续跑均逐个读取本次所选 ID 的原文件，按划分顺序计算 `image_bytes_sha256`。该值随 NPZ 元数据和写入日志保存，可与原始数据验收收据逐 split 核对。相同 ID 对应的图片字节有变化时，须重新提取。历史真实特征缺失该摘要时，请先归档旧文件，再从已验证的原图重算。

每个条件通过 `RunLogger` 记录设备、精度、实际 batch、耗时、峰值显存、缓存状态和样本数量。统计正式提取时间时选 `status=ok`、`cache_status=written`、`limit=null` 的写入日志。跳过已验证文件时会用 `cache_status=validated_skip` 追加一条核验耗时记录，并保留原始写入记录。旧文件若仅缺设备与 batch 字段，其余身份及数组校验仍须通过，跳过时记录 `runtime_metadata_complete=false`，原文件保持原样。交接时将 `logs/runs.jsonl` 复制为 `logs/P2_runs.jsonl`，和特征一起提交。

Python 调用 `extract_dataset` 时，模型名称和设置默认读取实际 wrapper，显式参数不匹配会报错。自定义或修改过的模型权重应显式传入独一的 `cache_identity`（例如 checkpoint SHA256）；直接 API 不会遍历大模型参数做哈希。CLI 会自动记录原模型 revision，并对微调 checkpoint 计算一次哈希。相对 `--ckpt-root` 与加载器一样相对于仓库根目录解析。

## 本地测试

正式输入的校验与第一批交付记录在 [P2 交付记录](../docs/p2_delivery.md)。以下命令在本仓库根目录运行，Python 使用项目虚拟环境。

```powershell
python -m tools.prepare_p2_inputs fetch-coco
python -m tools.prepare_p2_inputs fetch-flickr
python -m tools.prepare_p2_inputs verify
python -m extract.norm_stats --stage pretrained
python -m tools.package_p2 --stage pretrained
python -m tools.package_p2 --stage source
```

`norm_stats` 可以按 `pretrained`、`clip-finetuned`、`siglip-finetuned` 或 `all` 运行。阶段报告写到 `results/p2_stages/{stage}/`，全量报告写到 `results/`。L2 范数基于逐样本 raw embedding，CSV 的 11 列遵循分工方案；直方图同一图共用分箱边界。预训练、两批微调的统计行数分别为 40、60、60。

LiT 对照保存实际最大绝对差和发生差异的图像 ID。打包前若发现非零差异，会按原 batch 重跑预训练和 LiT，记录原精度的基线重复误差、FP32 对照和视觉参数摘要。无法由已测重复误差解释的结果保留在 `lit_rechecks/` 供定位。也可以用 `python -m extract.recheck_lit --help` 单独复核指定条件。

`tools.package_p2` 会核对最新 NPZ、原图、真实 GPU 小批、训练权重及其收据、统计表与计算日志。默认在仓库旁的 `P2_本地交接/` 生成 ZIP 和校验文件。`--verify <ZIP路径>` 会重新读出每个成员核验校验值。

```powershell
python -m pytest tests/test_extract_features.py -q
```

测试使用 CPU 假模型与内存数据，检查 raw 向量、五 caption 顺序、扰动字段、COCO 标签、缓存跳过与拒绝、失败不产生正式文件、检查输出隔离，以及 96/32/128 条件矩阵。真实权重单批验证见 `tests/test_extract_one_batch.py`；全量特征交付须在图片与各 checkpoint 到位后运行。
