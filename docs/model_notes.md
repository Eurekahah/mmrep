# P2 模型说明

本模块沿用 `configs/models.yaml` 和 P1 预处理。加载时校验模型配置与项目输入尺寸、嵌入维度一致。

| 项目 | CLIP | SigLIP2 |
|---|---|---|
| HF 标识 | `openai/clip-vit-base-patch32` | `google/siglip2-base-patch32-256` |
| HF 类 | `CLIPModel` | `SiglipModel`（该 checkpoint 的 model_type 为 siglip） |
| 输入尺寸 | 224 x 224 | 256 x 256 |
| 预处理 | 短边缩放、中心裁剪、CLIP mean/std | 拉伸缩放、mean=std=0.5 |
| 图文输出维度 | 512 | 768 |
| tokenizer | `CLIPTokenizer` | `AutoTokenizer` |
| max_length | 77 | 64 |
| padding / truncation | max_length / True | max_length / True |
| 封装额外转小写 | 否 | 是 |
| 温度参数 | `wrapper.model.logit_scale` | `wrapper.model.logit_scale` |
| 偏置参数 | 无 logit_bias | `wrapper.model.logit_bias` |
| 文本投影 | text_projection | text_model.head |
| 图像投影/池化 | visual_projection | vision_model.head |

encode_image/encode_text 保存 get_image_features/get_text_features 的 raw 输出，不做 L2 归一化，不乘温度或加偏置。需要余弦、检索或训练损失的下游自行归一化。保留两模型各自维度，不跨模型直接计算向量距离。

模型参数保留 float32，提取脚本在 CUDA 上使用 fp16 autocast，保存时转换 float32。模型和 tokenizer 采用同一解析后的 HF revision。

新版组内方案已记录 CLIP 224 / SigLIP2 256 的调整，与原指导书统一 224 不同；实现沿用现有仓库配置，不自行改变 P1 接口。COCO 的数量以仓库 splits 索引为准，不重新划分。

参考：[CLIP 配置](https://huggingface.co/openai/clip-vit-base-patch32/blob/main/config.json)、[SigLIP2 配置](https://huggingface.co/google/siglip2-base-patch32-256/blob/main/config.json)、[SigLIP2 tokenizer 配置](https://huggingface.co/google/siglip2-base-patch32-256/blob/main/tokenizer_config.json)，以及 CONTRACT 3.4/3.7/3.8/3.9。

实际下载版本、实测输出和验收记录见 `docs/p2_delivery.md`。
