# P4 交付说明

完成日期：2026-11-13。
服务器项目：/media/hdd2/xiaoying/projects/mmrep
基于仓库提交：52afe23aa96d6fbfe8c693616fbbdb76bf77882a。工作区保留 P4 改动，未 commit/push。

## 已完成

- CLIP 与 SigLIP2，各 r=8/r=16，四组均使用 Flickr train 8000 张、5 epoch、batch 128、seed 42。
- 每组两塔共 96 个注意力投影挂载 LoRA，主干、logit_scale/logit_bias 冻结；LoRA 参数 fp32，前向 fp16 autocast + GradScaler。
- 保存四个 PEFT adapter、meta.json、参数名、目标层名、loss 曲线、500 张 val 漂移统计和 32 张 merge 校验。
- 四组均通过实际 P2 load_model 保存后加载及 merge 校验。
- 仓库完整测试：124 passed、3 skipped；pip check 通过；四张 loss 曲线已逐张检查。

## 实验结果

|模型|rank|可训练参数|比例|最后一轮 loss|最大 merge 误差|训练秒数|峰值显存 MB|
|---|---|---|---|---|---|---|---|
|clip|8|983040|0.6456%|0.191866|2.98023224e-06|158.64|6118.5|
|clip|16|1966080|1.2830%|0.167067|3.81469727e-06|105.14|6136.0|
|siglip|8|1179648|0.3120%|0.576969|6.91413879e-06|144.90|7238.3|
|siglip|16|2359296|0.6222%|0.524867|3.81469727e-06|82.85|7266.0|

loss 为各自模型的训练目标，不跨模型比较。耗时为训练和保存阶段，不含安装、训练前特征和后续验证。

## 文件位置

- ckpt/{clip,siglip}/lora_r{8,16}/adapter/：交给 P2/P6 的 adapter。
- 同级 meta.json、loss.png、trainable_names.txt、target_modules.txt、selfcheck.json、selfcheck_cosines.npz。
- results/lora_params.csv：交给 P10 的参数量表。
- docs/lora_selfcheck.md、docs/lora_readme.md：自检与复现说明，含文献阅读要点。
- logs/P4_runs.jsonl、logs/P4_environment.txt：正式实验日志和环境；试跑日志已从交接 JSONL 中排除。
- train/train_lora.py、train/run_lora.sh、train/summarize_lora.py、configs/lora.yaml 和对应测试。

## 运行细节与交接边界

SigLIP2 两组各有 3 次 AMP 溢出跳步，实际更新 312/315 次；scheduler 随有效更新推进，与 P3 现有实现一致，最后 lr=1.2418637e-7。两组 CLIP 无跳步，最后 lr=0。均保留最后一轮，不按 val 选择。完整细节见自检记录与 meta.json。

本包覆盖到同版本 mmrep 仓库根目录即可交接 P4 文件；不包含原始图片、预训练模型和 .venv，复现需按 README 准备。服务器模型快照使用链接，搬迁时需要保留实际权重。

尚未上传共享云盘或发送群消息；未开展其他分工的特征提取、检索与几何指标实验。
