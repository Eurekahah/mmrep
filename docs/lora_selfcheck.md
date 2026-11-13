# P4 LoRA 自检

使用 val 前 500 张图与每图第一条 caption；原始嵌入余弦相似度。
merge：同一批 32 张，fp32 eval、关闭 TF32；图像及文本最大绝对差均要求 <1e-4。

|模型|rank|图像余弦均值|文本余弦均值|图像 merge 最大差|文本 merge 最大差|P2 加载|
|---|---|---|---|---|---|---|
|clip|8|0.92677712|0.92939788|2.38418579e-06|2.98023224e-06|通过|
|clip|16|0.92612952|0.92278308|3.81469727e-06|2.74181366e-06|通过|
|siglip|8|0.94614911|0.94975138|6.91413879e-06|3.33786011e-06|通过|
|siglip|16|0.93749106|0.94836354|3.78489494e-06|3.81469727e-06|通过|

各组详细分布、损失与元数据见对应 checkpoint 目录。
以上只验证表征变化和保存恢复的一致性，不代表检索性能或鲁棒性结论。

## AMP 运行记录

GradScaler 遇到溢出时跳过优化器更新，scheduler 同时不推进，与现有 P3 脚本一致。
因此有跳步时最后学习率可能略大于 0；实际值与跳步次数如下：

- clip r=8: optimizer_updates=315, skipped_updates=0, final_lr=0。
- clip r=16: optimizer_updates=315, skipped_updates=0, final_lr=0。
- siglip r=8: optimizer_updates=312, skipped_updates=3, final_lr=1.2418637e-07。
- siglip r=16: optimizer_updates=312, skipped_updates=3, final_lr=1.2418637e-07。
