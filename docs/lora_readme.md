# P4：双塔 LoRA 微调

在仓库根目录运行。环境为 Python 3.10、torch 2.4.1+cu121、transformers 4.49.0、peft 0.14.0，依赖见 requirements.txt。

## 本机准备

```bash
source .venv/bin/activate
export HF_HOME=/media/hdd2/xiaoying/model-weights/mmrep
export HF_HUB_OFFLINE=1
```

本机两套独立模型目录通过符号链接接入 HF_HOME/hub 的标准快照缓存，因此 P2 原有 load_model 可以直接按公开仓库 ID 离线加载；更换机器时需自行下载这两个公开模型，或迁移模型目录并修复缓存链接。不要只迁移链接而丢掉实际权重。

图片目录为 data_raw/flickr30k/flickr30k-images；使用仓库 splits/flickr_split.json，不重新划分。

## 测试和训练

```bash
pytest -q tests/test_model_names.py tests/test_train_lora.py
python -m train.train_lora --model clip --rank 8 --config configs/lora.yaml
python -m train.train_lora --model clip --rank 16 --config configs/lora.yaml
python -m train.train_lora --model siglip --rank 8 --config configs/lora.yaml
python -m train.train_lora --model siglip --rank 16 --config configs/lora.yaml
```

每组使用干净 Flickr train 8000 张、5 epoch、batch 128、seed 42。每张图每轮随机选一条 caption，抽样与 shuffle 使用独立的种子 42 生成器，复用 P3 的 caption 抽样函数。AdamW lr=5e-4、weight decay=0.01，5% warmup 后 cosine 衰减。参数保持 fp32，CUDA 前向 fp16 autocast + GradScaler，归一化和损失用 fp32。梯度溢出时记录 skipped_updates 并跳过对应 scheduler 更新。

OOM 时先用 --gradient-checkpointing，仍不够才加 --batch-size 64。失败运行目录先保留到别处再重跑，脚本拒绝覆盖已有产出。--max-samples 和 --epochs 仅供试跑；试跑加 --skip-selfcheck，并用 --ckpt-root 指向独立临时目录，不要把试跑当正式交付。

## 参数范围与损失

两塔 encoder 的 12 层 self_attn.q_proj/k_proj/v_proj/out_proj 共 96 个线性层，图像/文本各 48 个。r=8/16，alpha=2r，dropout=0.05，bias=none。除 lora_A/lora_B 外所有参数冻结，包括 logit_scale、logit_bias、两侧投影与 SigLIP 池化头。配置采用分工方案的精确正则，修正了仓库草稿的宽泛匹配。

CLIP 用 train.losses.info_nce，SigLIP2 用 train.losses.sigmoid_loss。保存最后一轮，不用 val 选择 checkpoint。总参数量包含新增的 LoRA 参数；ratio=trainable_params/total_params。

## 交付与验收

每组 ckpt/{model}/lora_r{rank}/ 含 adapter/、meta.json、trainable_names.txt、target_modules.txt、loss.png、selfcheck.json、selfcheck_cosines.npz。

训练前后对 val 前 500 张图片及每张第一条 caption 计算余弦相似度，样本顺序沿用划分文件。merge 校验取前 32 张，使用相同 batch=16、eval 模式、fp32、关闭 TF32，对图像和文本的原始嵌入都要求最大绝对差 <1e-4。恢复路径实际调用 P2 models.loader.load_model，再由其执行 PeftModel.from_pretrained 和 merge_and_unload。自检失败会非零退出，保留已有 adapter 和失败证据。

日志为 logs/lora_{model}_lora_r{rank}.log 和 logs/runs.jsonl。日志中的 num_samples、epochs 可区分试跑。汇总脚本 train.summarize_lora 生成 results/lora_params.csv、docs/lora_selfcheck.md，并只提取四个正式运行的记录写入 logs/P4_runs.jsonl。

交接时保持 ckpt/、results/、logs/ 的相对目录结构；权重不进 git。云盘上传和群内通知需要单独执行，本脚本不会发送消息。

## 文献与解释边界

LoRA 原论文 https://arxiv.org/abs/2106.09685 提出冻结原权重、学习低秩增量 W'=W+(alpha/r)BA；训练的自由度随秩增加，推理可把增量合并回原权重。本实验将该机制应用于两塔注意力投影，比较 r=8 与 r=16 的表征漂移。

MultiLoReFT https://arxiv.org/abs/2607.16789 使用共享与模态特异的低秩表征子空间，并引入子空间约束及自适应剪枝。它提供分析低秩子空间的参考，但本 P4 实现的是标准权重 LoRA，没有实现 MultiLoReFT；不能把漂移或几何变化直接解释成已实现共享/特异信息解耦。两篇工作的模块和实验场景不同，不能直接套用它们的效果结论。
