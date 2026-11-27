# 图注与观察

图号按文件路径稳定分配；PNG/SVG 是同一张图的两种格式，共用图号。P9 二维投影只作定性辅助，原空间指标以 P7 为准。

## F001 — gantt

图注：gantt：组内上游已交付的图。

该图已存在于组内仓库，登记其原有文件路径供报告引用。

未提供可核对的数值 CSV，不能在此确认数值一致性。

文件：`docs/gantt.png`；来源：`docs/分工方案.md`。

## F002 — perturb_examples_clip

图注：perturb_examples_clip：组内上游已交付的图。

该图已存在于组内仓库，登记其原有文件路径供报告引用。

未提供可核对的数值 CSV，不能在此确认数值一致性。

文件：`docs/perturb_examples_clip.png`；来源：`configs/perturb.yaml;configs/models.yaml`。

## F003 — perturb_examples_siglip

图注：perturb_examples_siglip：组内上游已交付的图。

该图已存在于组内仓库，登记其原有文件路径供报告引用。

未提供可核对的数值 CSV，不能在此确认数值一致性。

文件：`docs/perturb_examples_siglip.png`；来源：`configs/perturb.yaml;configs/models.yaml`。

## F004 — alignment_uniformity_scatter

图注：alignment_uniformity_scatter：组内上游已交付的图。

对应来源记录中 gap 的范围为 0.8016 至 1.1021（沿用原始单位）。

对应来源记录中 align_cos 的范围为 0.6734 至 0.8855（沿用原始单位）。

此处描述来源记录的数值范围；因果解释和假设判定由 P8/P10 汇总。

文件：`results/figs/alignment_uniformity_scatter.png`；来源：`results/geometry_metrics.csv`。

## F005 — modality_gap_bars

图注：modality_gap_bars：组内上游已交付的图。

对应来源记录中 gap 的范围为 0.8016 至 1.1021（沿用原始单位）。

对应来源记录中 align_cos 的范围为 0.6734 至 0.8855（沿用原始单位）。

此处描述来源记录的数值范围；因果解释和假设判定由 P8/P10 汇总。

文件：`results/figs/modality_gap_bars.png`；来源：`results/geometry_metrics.csv`。

## F006 — norm_clip_coco_test_image

图注：norm_clip_coco_test_image：组内上游已交付的图。

对应来源记录中 mean 的范围为 10.0662 至 10.6762（沿用原始单位）。

此处描述来源记录的数值范围；因果解释和假设判定由 P8/P10 汇总。

文件：`results/figs/norm_clip_coco_test_image.png`；来源：`results/norm_stats.csv`。

## F007 — norm_clip_coco_test_text

图注：norm_clip_coco_test_text：组内上游已交付的图。

对应来源记录中 mean 的范围为 7.5703 至 8.0343（沿用原始单位）。

此处描述来源记录的数值范围；因果解释和假设判定由 P8/P10 汇总。

文件：`results/figs/norm_clip_coco_test_text.png`；来源：`results/norm_stats.csv`。

## F008 — norm_clip_coco_val_image

图注：norm_clip_coco_val_image：组内上游已交付的图。

对应来源记录中 mean 的范围为 10.0638 至 10.6723（沿用原始单位）。

此处描述来源记录的数值范围；因果解释和假设判定由 P8/P10 汇总。

文件：`results/figs/norm_clip_coco_val_image.png`；来源：`results/norm_stats.csv`。

## F009 — norm_clip_coco_val_text

图注：norm_clip_coco_val_text：组内上游已交付的图。

对应来源记录中 mean 的范围为 7.5947 至 8.0654（沿用原始单位）。

此处描述来源记录的数值范围；因果解释和假设判定由 P8/P10 汇总。

文件：`results/figs/norm_clip_coco_val_text.png`；来源：`results/norm_stats.csv`。

## F010 — norm_clip_flickr_test_image

图注：norm_clip_flickr_test_image：组内上游已交付的图。

对应来源记录中 mean 的范围为 9.9166 至 10.5724（沿用原始单位）。

此处描述来源记录的数值范围；因果解释和假设判定由 P8/P10 汇总。

文件：`results/figs/norm_clip_flickr_test_image.png`；来源：`results/norm_stats.csv`。

## F011 — norm_clip_flickr_test_text

图注：norm_clip_flickr_test_text：组内上游已交付的图。

对应来源记录中 mean 的范围为 7.5853 至 7.8706（沿用原始单位）。

此处描述来源记录的数值范围；因果解释和假设判定由 P8/P10 汇总。

文件：`results/figs/norm_clip_flickr_test_text.png`；来源：`results/norm_stats.csv`。

## F012 — norm_clip_flickr_val_image

图注：norm_clip_flickr_val_image：组内上游已交付的图。

对应来源记录中 mean 的范围为 9.9202 至 11.0512（沿用原始单位）。

此处描述来源记录的数值范围；因果解释和假设判定由 P8/P10 汇总。

文件：`results/figs/norm_clip_flickr_val_image.png`；来源：`results/norm_stats.csv`。

## F013 — norm_clip_flickr_val_text

图注：norm_clip_flickr_val_text：组内上游已交付的图。

对应来源记录中 mean 的范围为 7.5680 至 7.8889（沿用原始单位）。

此处描述来源记录的数值范围；因果解释和假设判定由 P8/P10 汇总。

文件：`results/figs/norm_clip_flickr_val_text.png`；来源：`results/norm_stats.csv`。

## F014 — norm_siglip_coco_test_image

图注：norm_siglip_coco_test_image：组内上游已交付的图。

对应来源记录中 mean 的范围为 10.6407 至 11.5032（沿用原始单位）。

此处描述来源记录的数值范围；因果解释和假设判定由 P8/P10 汇总。

文件：`results/figs/norm_siglip_coco_test_image.png`；来源：`results/norm_stats.csv`。

## F015 — norm_siglip_coco_test_text

图注：norm_siglip_coco_test_text：组内上游已交付的图。

对应来源记录中 mean 的范围为 16.3216 至 18.5237（沿用原始单位）。

此处描述来源记录的数值范围；因果解释和假设判定由 P8/P10 汇总。

文件：`results/figs/norm_siglip_coco_test_text.png`；来源：`results/norm_stats.csv`。

## F016 — norm_siglip_coco_val_image

图注：norm_siglip_coco_val_image：组内上游已交付的图。

对应来源记录中 mean 的范围为 10.6692 至 11.5043（沿用原始单位）。

此处描述来源记录的数值范围；因果解释和假设判定由 P8/P10 汇总。

文件：`results/figs/norm_siglip_coco_val_image.png`；来源：`results/norm_stats.csv`。

## F017 — norm_siglip_coco_val_text

图注：norm_siglip_coco_val_text：组内上游已交付的图。

对应来源记录中 mean 的范围为 16.3583 至 18.5670（沿用原始单位）。

此处描述来源记录的数值范围；因果解释和假设判定由 P8/P10 汇总。

文件：`results/figs/norm_siglip_coco_val_text.png`；来源：`results/norm_stats.csv`。

## F018 — norm_siglip_flickr_test_image

图注：norm_siglip_flickr_test_image：组内上游已交付的图。

对应来源记录中 mean 的范围为 10.3037 至 11.1687（沿用原始单位）。

此处描述来源记录的数值范围；因果解释和假设判定由 P8/P10 汇总。

文件：`results/figs/norm_siglip_flickr_test_image.png`；来源：`results/norm_stats.csv`。

## F019 — norm_siglip_flickr_test_text

图注：norm_siglip_flickr_test_text：组内上游已交付的图。

对应来源记录中 mean 的范围为 16.0817 至 18.4038（沿用原始单位）。

此处描述来源记录的数值范围；因果解释和假设判定由 P8/P10 汇总。

文件：`results/figs/norm_siglip_flickr_test_text.png`；来源：`results/norm_stats.csv`。

## F020 — norm_siglip_flickr_val_image

图注：norm_siglip_flickr_val_image：组内上游已交付的图。

对应来源记录中 mean 的范围为 10.3108 至 11.2396（沿用原始单位）。

此处描述来源记录的数值范围；因果解释和假设判定由 P8/P10 汇总。

文件：`results/figs/norm_siglip_flickr_val_image.png`；来源：`results/norm_stats.csv`。

## F021 — norm_siglip_flickr_val_text

图注：norm_siglip_flickr_val_text：组内上游已交付的图。

对应来源记录中 mean 的范围为 16.1426 至 18.3866（沿用原始单位）。

此处描述来源记录的数值范围；因果解释和假设判定由 P8/P10 汇总。

文件：`results/figs/norm_siglip_flickr_val_text.png`；来源：`results/norm_stats.csv`。

## F022 — p5_drift_image_coco_test

图注：p5_drift_image_coco_test：组内上游已交付的图。

对应来源记录中 mean 的范围为 0.8231 至 1.0000（沿用原始单位）。

此处描述来源记录的数值范围；因果解释和假设判定由 P8/P10 汇总。

文件：`results/figs/p5_drift_image_coco_test.png`；来源：`results/drift_cosine.csv`。

## F023 — p5_drift_image_coco_val

图注：p5_drift_image_coco_val：组内上游已交付的图。

对应来源记录中 mean 的范围为 0.8228 至 1.0000（沿用原始单位）。

此处描述来源记录的数值范围；因果解释和假设判定由 P8/P10 汇总。

文件：`results/figs/p5_drift_image_coco_val.png`；来源：`results/drift_cosine.csv`。

## F024 — p5_drift_image_flickr_test

图注：p5_drift_image_flickr_test：组内上游已交付的图。

对应来源记录中 mean 的范围为 0.8091 至 1.0000（沿用原始单位）。

此处描述来源记录的数值范围；因果解释和假设判定由 P8/P10 汇总。

文件：`results/figs/p5_drift_image_flickr_test.png`；来源：`results/drift_cosine.csv`。

## F025 — p5_drift_image_flickr_val

图注：p5_drift_image_flickr_val：组内上游已交付的图。

对应来源记录中 mean 的范围为 0.7061 至 1.0000（沿用原始单位）。

此处描述来源记录的数值范围；因果解释和假设判定由 P8/P10 汇总。

文件：`results/figs/p5_drift_image_flickr_val.png`；来源：`results/drift_cosine.csv`。

## F026 — p5_drift_text_coco_test

图注：p5_drift_text_coco_test：组内上游已交付的图。

对应来源记录中 mean 的范围为 0.7845 至 1.0000（沿用原始单位）。

此处描述来源记录的数值范围；因果解释和假设判定由 P8/P10 汇总。

文件：`results/figs/p5_drift_text_coco_test.png`；来源：`results/drift_cosine.csv`。

## F027 — p5_drift_text_coco_val

图注：p5_drift_text_coco_val：组内上游已交付的图。

对应来源记录中 mean 的范围为 0.7858 至 1.0000（沿用原始单位）。

此处描述来源记录的数值范围；因果解释和假设判定由 P8/P10 汇总。

文件：`results/figs/p5_drift_text_coco_val.png`；来源：`results/drift_cosine.csv`。

## F028 — p5_drift_text_flickr_test

图注：p5_drift_text_flickr_test：组内上游已交付的图。

对应来源记录中 mean 的范围为 0.7685 至 1.0000（沿用原始单位）。

此处描述来源记录的数值范围；因果解释和假设判定由 P8/P10 汇总。

文件：`results/figs/p5_drift_text_flickr_test.png`；来源：`results/drift_cosine.csv`。

## F029 — p5_drift_text_flickr_val

图注：p5_drift_text_flickr_val：组内上游已交付的图。

对应来源记录中 mean 的范围为 0.7677 至 1.0000（沿用原始单位）。

此处描述来源记录的数值范围；因果解释和假设判定由 P8/P10 汇总。

文件：`results/figs/p5_drift_text_flickr_val.png`；来源：`results/drift_cosine.csv`。

## F030 — p5_drop_coco_test

图注：p5_drop_coco_test：组内上游已交付的图。

对应来源记录中 drop_pp 的范围为 2.4394 至 23.1826（沿用原始单位）。

此处描述来源记录的数值范围；因果解释和假设判定由 P8/P10 汇总。

文件：`results/figs/p5_drop_coco_test.png`；来源：`results/retrieval_drop.csv`。

## F031 — p5_drop_coco_val

图注：p5_drop_coco_val：组内上游已交付的图。

对应来源记录中 drop_pp 的范围为 1.5509 至 24.0307（沿用原始单位）。

此处描述来源记录的数值范围；因果解释和假设判定由 P8/P10 汇总。

文件：`results/figs/p5_drop_coco_val.png`；来源：`results/retrieval_drop.csv`。

## F032 — p5_drop_flickr_test

图注：p5_drop_flickr_test：组内上游已交付的图。

对应来源记录中 drop_pp 的范围为 0.8200 至 24.1000（沿用原始单位）。

此处描述来源记录的数值范围；因果解释和假设判定由 P8/P10 汇总。

文件：`results/figs/p5_drop_flickr_test.png`；来源：`results/retrieval_drop.csv`。

## F033 — p5_drop_flickr_val

图注：p5_drop_flickr_val：组内上游已交付的图。

对应来源记录中 drop_pp 的范围为 -0.2000 至 51.5000（沿用原始单位）。

此处描述来源记录的数值范围；因果解释和假设判定由 P8/P10 汇总。

文件：`results/figs/p5_drop_flickr_val.png`；来源：`results/retrieval_drop.csv`。

## F034 — p5_recall_coco_test

图注：p5_recall_coco_test：组内上游已交付的图。

对应来源记录中 i2t_R1 的范围为 35.8643 至 76.8174（沿用原始单位）。

对应来源记录中 t2i_R1 的范围为 26.0824 至 59.0549（沿用原始单位）。

此处描述来源记录的数值范围；因果解释和假设判定由 P8/P10 汇总。

文件：`results/figs/p5_recall_coco_test.png`；来源：`results/retrieval_test.csv`。

## F035 — p5_recall_coco_val

图注：p5_recall_coco_val：组内上游已交付的图。

对应来源记录中 i2t_R1 的范围为 36.5913 至 75.6058（沿用原始单位）。

对应来源记录中 t2i_R1 的范围为 25.9451 至 57.7706（沿用原始单位）。

此处描述来源记录的数值范围；因果解释和假设判定由 P8/P10 汇总。

文件：`results/figs/p5_recall_coco_val.png`；来源：`results/retrieval_val.csv`。

## F036 — p5_recall_flickr_test

图注：p5_recall_flickr_test：组内上游已交付的图。

对应来源记录中 i2t_R1 的范围为 61.1000 至 91.8000（沿用原始单位）。

对应来源记录中 t2i_R1 的范围为 47.0600 至 78.5200（沿用原始单位）。

此处描述来源记录的数值范围；因果解释和假设判定由 P8/P10 汇总。

文件：`results/figs/p5_recall_flickr_test.png`；来源：`results/retrieval_test.csv`。

## F037 — p5_recall_flickr_val

图注：p5_recall_flickr_val：组内上游已交付的图。

对应来源记录中 i2t_R1 的范围为 33.2000 至 91.0000（沿用原始单位）。

对应来源记录中 t2i_R1 的范围为 28.7000 至 78.1800（沿用原始单位）。

此处描述来源记录的数值范围；因果解释和假设判定由 P8/P10 汇总。

文件：`results/figs/p5_recall_flickr_val.png`；来源：`results/retrieval_val.csv`。

## F038 — p5_strength_noise

图注：p5_strength_noise：组内上游已交付的图。

对应来源记录中 drop_pp 的范围为 -0.2000 至 27.7000（沿用原始单位）。

此处描述来源记录的数值范围；因果解释和假设判定由 P8/P10 汇总。

文件：`results/figs/p5_strength_noise.png`；来源：`results/retrieval_drop.csv`。

## F039 — p5_strength_occl

图注：p5_strength_occl：组内上游已交付的图。

对应来源记录中 drop_pp 的范围为 1.0000 至 51.5000（沿用原始单位）。

此处描述来源记录的数值范围；因果解释和假设判定由 P8/P10 汇总。

文件：`results/figs/p5_strength_occl.png`；来源：`results/retrieval_drop.csv`。

## F040 — strength_alignment

图注：strength_alignment：组内上游已交付的图。

对应来源记录中 gap 的范围为 0.8073 至 1.1021（沿用原始单位）。

对应来源记录中 align_cos 的范围为 0.6747 至 0.8855（沿用原始单位）。

此处描述来源记录的数值范围；因果解释和假设判定由 P8/P10 汇总。

文件：`results/figs/strength_alignment.png`；来源：`results/geometry_metrics.csv`。

## F041 — strength_modality_gap

图注：strength_modality_gap：组内上游已交付的图。

对应来源记录中 gap 的范围为 0.8073 至 1.1021（沿用原始单位）。

对应来源记录中 align_cos 的范围为 0.6747 至 0.8855（沿用原始单位）。

此处描述来源记录的数值范围；因果解释和假设判定由 P8/P10 汇总。

文件：`results/figs/strength_modality_gap.png`；来源：`results/geometry_metrics.csv`。

## F042 — loss

图注：loss：组内上游已交付的图。

对应来源记录中 loss 的范围为 0.1957 至 0.5475（沿用原始单位）。

此处描述来源记录的数值范围；因果解释和假设判定由 P8/P10 汇总。

文件：`results/lit_selfcheck/clip/loss.png`；来源：`results/lit_selfcheck/clip/epochs.csv`。

## F043 — loss

图注：loss：组内上游已交付的图。

对应来源记录中 loss 的范围为 0.5637 至 1.1160（沿用原始单位）。

此处描述来源记录的数值范围；因果解释和假设判定由 P8/P10 汇总。

文件：`results/lit_selfcheck/siglip/loss.png`；来源：`results/lit_selfcheck/siglip/epochs.csv`。

## F044 — norm_clip_coco_test_image

图注：norm_clip_coco_test_image：组内上游已交付的图。

对应来源记录中 mean 的范围为 10.0662 至 10.6762（沿用原始单位）。

此处描述来源记录的数值范围；因果解释和假设判定由 P8/P10 汇总。

文件：`results/p2_stages/clip-finetuned/figs/norm_clip_coco_test_image.png`；来源：`results/p2_stages/clip-finetuned/norm_stats.csv`。

## F045 — norm_clip_coco_test_text

图注：norm_clip_coco_test_text：组内上游已交付的图。

对应来源记录中 mean 的范围为 7.5703 至 7.8181（沿用原始单位）。

此处描述来源记录的数值范围；因果解释和假设判定由 P8/P10 汇总。

文件：`results/p2_stages/clip-finetuned/figs/norm_clip_coco_test_text.png`；来源：`results/p2_stages/clip-finetuned/norm_stats.csv`。

## F046 — norm_clip_coco_val_image

图注：norm_clip_coco_val_image：组内上游已交付的图。

对应来源记录中 mean 的范围为 10.0638 至 10.6723（沿用原始单位）。

此处描述来源记录的数值范围；因果解释和假设判定由 P8/P10 汇总。

文件：`results/p2_stages/clip-finetuned/figs/norm_clip_coco_val_image.png`；来源：`results/p2_stages/clip-finetuned/norm_stats.csv`。

## F047 — norm_clip_coco_val_text

图注：norm_clip_coco_val_text：组内上游已交付的图。

对应来源记录中 mean 的范围为 7.5947 至 7.8427（沿用原始单位）。

此处描述来源记录的数值范围；因果解释和假设判定由 P8/P10 汇总。

文件：`results/p2_stages/clip-finetuned/figs/norm_clip_coco_val_text.png`；来源：`results/p2_stages/clip-finetuned/norm_stats.csv`。

## F048 — norm_clip_flickr_test_image

图注：norm_clip_flickr_test_image：组内上游已交付的图。

对应来源记录中 mean 的范围为 9.9166 至 10.5724（沿用原始单位）。

此处描述来源记录的数值范围；因果解释和假设判定由 P8/P10 汇总。

文件：`results/p2_stages/clip-finetuned/figs/norm_clip_flickr_test_image.png`；来源：`results/p2_stages/clip-finetuned/norm_stats.csv`。

## F049 — norm_clip_flickr_test_text

图注：norm_clip_flickr_test_text：组内上游已交付的图。

对应来源记录中 mean 的范围为 7.5853 至 7.6146（沿用原始单位）。

此处描述来源记录的数值范围；因果解释和假设判定由 P8/P10 汇总。

文件：`results/p2_stages/clip-finetuned/figs/norm_clip_flickr_test_text.png`；来源：`results/p2_stages/clip-finetuned/norm_stats.csv`。

## F050 — norm_clip_flickr_val_image

图注：norm_clip_flickr_val_image：组内上游已交付的图。

对应来源记录中 mean 的范围为 9.9202 至 11.0512（沿用原始单位）。

此处描述来源记录的数值范围；因果解释和假设判定由 P8/P10 汇总。

文件：`results/p2_stages/clip-finetuned/figs/norm_clip_flickr_val_image.png`；来源：`results/p2_stages/clip-finetuned/norm_stats.csv`。

## F051 — norm_clip_flickr_val_text

图注：norm_clip_flickr_val_text：组内上游已交付的图。

对应来源记录中 mean 的范围为 7.5680 至 7.6063（沿用原始单位）。

此处描述来源记录的数值范围；因果解释和假设判定由 P8/P10 汇总。

文件：`results/p2_stages/clip-finetuned/figs/norm_clip_flickr_val_text.png`；来源：`results/p2_stages/clip-finetuned/norm_stats.csv`。

## F052 — norm_clip_coco_test_image

图注：norm_clip_coco_test_image：组内上游已交付的图。

对应来源记录中 mean 的范围为 10.4355 至 10.6762（沿用原始单位）。

此处描述来源记录的数值范围；因果解释和假设判定由 P8/P10 汇总。

文件：`results/p2_stages/pretrained/figs/norm_clip_coco_test_image.png`；来源：`results/p2_stages/pretrained/norm_stats.csv`。

## F053 — norm_clip_coco_test_text

图注：norm_clip_coco_test_text：组内上游已交付的图。

对应来源记录中 mean 的范围为 8.0343 至 8.0343（沿用原始单位）。

此处描述来源记录的数值范围；因果解释和假设判定由 P8/P10 汇总。

文件：`results/p2_stages/pretrained/figs/norm_clip_coco_test_text.png`；来源：`results/p2_stages/pretrained/norm_stats.csv`。

## F054 — norm_clip_coco_val_image

图注：norm_clip_coco_val_image：组内上游已交付的图。

对应来源记录中 mean 的范围为 10.4131 至 10.6723（沿用原始单位）。

此处描述来源记录的数值范围；因果解释和假设判定由 P8/P10 汇总。

文件：`results/p2_stages/pretrained/figs/norm_clip_coco_val_image.png`；来源：`results/p2_stages/pretrained/norm_stats.csv`。

## F055 — norm_clip_coco_val_text

图注：norm_clip_coco_val_text：组内上游已交付的图。

对应来源记录中 mean 的范围为 8.0654 至 8.0654（沿用原始单位）。

此处描述来源记录的数值范围；因果解释和假设判定由 P8/P10 汇总。

文件：`results/p2_stages/pretrained/figs/norm_clip_coco_val_text.png`；来源：`results/p2_stages/pretrained/norm_stats.csv`。

## F056 — norm_clip_flickr_test_image

图注：norm_clip_flickr_test_image：组内上游已交付的图。

对应来源记录中 mean 的范围为 10.3574 至 10.5724（沿用原始单位）。

此处描述来源记录的数值范围；因果解释和假设判定由 P8/P10 汇总。

文件：`results/p2_stages/pretrained/figs/norm_clip_flickr_test_image.png`；来源：`results/p2_stages/pretrained/norm_stats.csv`。

## F057 — norm_clip_flickr_test_text

图注：norm_clip_flickr_test_text：组内上游已交付的图。

对应来源记录中 mean 的范围为 7.8706 至 7.8706（沿用原始单位）。

此处描述来源记录的数值范围；因果解释和假设判定由 P8/P10 汇总。

文件：`results/p2_stages/pretrained/figs/norm_clip_flickr_test_text.png`；来源：`results/p2_stages/pretrained/norm_stats.csv`。

## F058 — norm_clip_flickr_val_image

图注：norm_clip_flickr_val_image：组内上游已交付的图。

对应来源记录中 mean 的范围为 10.3704 至 11.0512（沿用原始单位）。

此处描述来源记录的数值范围；因果解释和假设判定由 P8/P10 汇总。

文件：`results/p2_stages/pretrained/figs/norm_clip_flickr_val_image.png`；来源：`results/p2_stages/pretrained/norm_stats.csv`。

## F059 — norm_clip_flickr_val_text

图注：norm_clip_flickr_val_text：组内上游已交付的图。

对应来源记录中 mean 的范围为 7.8889 至 7.8889（沿用原始单位）。

此处描述来源记录的数值范围；因果解释和假设判定由 P8/P10 汇总。

文件：`results/p2_stages/pretrained/figs/norm_clip_flickr_val_text.png`；来源：`results/p2_stages/pretrained/norm_stats.csv`。

## F060 — norm_siglip_coco_test_image

图注：norm_siglip_coco_test_image：组内上游已交付的图。

对应来源记录中 mean 的范围为 11.1941 至 11.5032（沿用原始单位）。

此处描述来源记录的数值范围；因果解释和假设判定由 P8/P10 汇总。

文件：`results/p2_stages/pretrained/figs/norm_siglip_coco_test_image.png`；来源：`results/p2_stages/pretrained/norm_stats.csv`。

## F061 — norm_siglip_coco_test_text

图注：norm_siglip_coco_test_text：组内上游已交付的图。

对应来源记录中 mean 的范围为 17.5180 至 17.5180（沿用原始单位）。

此处描述来源记录的数值范围；因果解释和假设判定由 P8/P10 汇总。

文件：`results/p2_stages/pretrained/figs/norm_siglip_coco_test_text.png`；来源：`results/p2_stages/pretrained/norm_stats.csv`。

## F062 — norm_siglip_coco_val_image

图注：norm_siglip_coco_val_image：组内上游已交付的图。

对应来源记录中 mean 的范围为 11.2264 至 11.5043（沿用原始单位）。

此处描述来源记录的数值范围；因果解释和假设判定由 P8/P10 汇总。

文件：`results/p2_stages/pretrained/figs/norm_siglip_coco_val_image.png`；来源：`results/p2_stages/pretrained/norm_stats.csv`。

## F063 — norm_siglip_coco_val_text

图注：norm_siglip_coco_val_text：组内上游已交付的图。

对应来源记录中 mean 的范围为 17.5685 至 17.5685（沿用原始单位）。

此处描述来源记录的数值范围；因果解释和假设判定由 P8/P10 汇总。

文件：`results/p2_stages/pretrained/figs/norm_siglip_coco_val_text.png`；来源：`results/p2_stages/pretrained/norm_stats.csv`。

## F064 — norm_siglip_flickr_test_image

图注：norm_siglip_flickr_test_image：组内上游已交付的图。

对应来源记录中 mean 的范围为 10.9109 至 11.1687（沿用原始单位）。

此处描述来源记录的数值范围；因果解释和假设判定由 P8/P10 汇总。

文件：`results/p2_stages/pretrained/figs/norm_siglip_flickr_test_image.png`；来源：`results/p2_stages/pretrained/norm_stats.csv`。

## F065 — norm_siglip_flickr_test_text

图注：norm_siglip_flickr_test_text：组内上游已交付的图。

对应来源记录中 mean 的范围为 16.8203 至 16.8203（沿用原始单位）。

此处描述来源记录的数值范围；因果解释和假设判定由 P8/P10 汇总。

文件：`results/p2_stages/pretrained/figs/norm_siglip_flickr_test_text.png`；来源：`results/p2_stages/pretrained/norm_stats.csv`。

## F066 — norm_siglip_flickr_val_image

图注：norm_siglip_flickr_val_image：组内上游已交付的图。

对应来源记录中 mean 的范围为 10.9375 至 11.2396（沿用原始单位）。

此处描述来源记录的数值范围；因果解释和假设判定由 P8/P10 汇总。

文件：`results/p2_stages/pretrained/figs/norm_siglip_flickr_val_image.png`；来源：`results/p2_stages/pretrained/norm_stats.csv`。

## F067 — norm_siglip_flickr_val_text

图注：norm_siglip_flickr_val_text：组内上游已交付的图。

对应来源记录中 mean 的范围为 16.9145 至 16.9145（沿用原始单位）。

此处描述来源记录的数值范围；因果解释和假设判定由 P8/P10 汇总。

文件：`results/p2_stages/pretrained/figs/norm_siglip_flickr_val_text.png`；来源：`results/p2_stages/pretrained/norm_stats.csv`。

## F068 — norm_siglip_coco_test_image

图注：norm_siglip_coco_test_image：组内上游已交付的图。

对应来源记录中 mean 的范围为 10.6407 至 11.5032（沿用原始单位）。

此处描述来源记录的数值范围；因果解释和假设判定由 P8/P10 汇总。

文件：`results/p2_stages/siglip-finetuned/figs/norm_siglip_coco_test_image.png`；来源：`results/p2_stages/siglip-finetuned/norm_stats.csv`。

## F069 — norm_siglip_coco_test_text

图注：norm_siglip_coco_test_text：组内上游已交付的图。

对应来源记录中 mean 的范围为 16.3216 至 18.5237（沿用原始单位）。

此处描述来源记录的数值范围；因果解释和假设判定由 P8/P10 汇总。

文件：`results/p2_stages/siglip-finetuned/figs/norm_siglip_coco_test_text.png`；来源：`results/p2_stages/siglip-finetuned/norm_stats.csv`。

## F070 — norm_siglip_coco_val_image

图注：norm_siglip_coco_val_image：组内上游已交付的图。

对应来源记录中 mean 的范围为 10.6692 至 11.5043（沿用原始单位）。

此处描述来源记录的数值范围；因果解释和假设判定由 P8/P10 汇总。

文件：`results/p2_stages/siglip-finetuned/figs/norm_siglip_coco_val_image.png`；来源：`results/p2_stages/siglip-finetuned/norm_stats.csv`。

## F071 — norm_siglip_coco_val_text

图注：norm_siglip_coco_val_text：组内上游已交付的图。

对应来源记录中 mean 的范围为 16.3583 至 18.5670（沿用原始单位）。

此处描述来源记录的数值范围；因果解释和假设判定由 P8/P10 汇总。

文件：`results/p2_stages/siglip-finetuned/figs/norm_siglip_coco_val_text.png`；来源：`results/p2_stages/siglip-finetuned/norm_stats.csv`。

## F072 — norm_siglip_flickr_test_image

图注：norm_siglip_flickr_test_image：组内上游已交付的图。

对应来源记录中 mean 的范围为 10.3037 至 11.1687（沿用原始单位）。

此处描述来源记录的数值范围；因果解释和假设判定由 P8/P10 汇总。

文件：`results/p2_stages/siglip-finetuned/figs/norm_siglip_flickr_test_image.png`；来源：`results/p2_stages/siglip-finetuned/norm_stats.csv`。

## F073 — norm_siglip_flickr_test_text

图注：norm_siglip_flickr_test_text：组内上游已交付的图。

对应来源记录中 mean 的范围为 16.0817 至 18.4038（沿用原始单位）。

此处描述来源记录的数值范围；因果解释和假设判定由 P8/P10 汇总。

文件：`results/p2_stages/siglip-finetuned/figs/norm_siglip_flickr_test_text.png`；来源：`results/p2_stages/siglip-finetuned/norm_stats.csv`。

## F074 — norm_siglip_flickr_val_image

图注：norm_siglip_flickr_val_image：组内上游已交付的图。

对应来源记录中 mean 的范围为 10.3108 至 11.2396（沿用原始单位）。

此处描述来源记录的数值范围；因果解释和假设判定由 P8/P10 汇总。

文件：`results/p2_stages/siglip-finetuned/figs/norm_siglip_flickr_val_image.png`；来源：`results/p2_stages/siglip-finetuned/norm_stats.csv`。

## F075 — norm_siglip_flickr_val_text

图注：norm_siglip_flickr_val_text：组内上游已交付的图。

对应来源记录中 mean 的范围为 16.1426 至 18.3866（沿用原始单位）。

此处描述来源记录的数值范围；因果解释和假设判定由 P8/P10 汇总。

文件：`results/p2_stages/siglip-finetuned/figs/norm_siglip_flickr_val_text.png`；来源：`results/p2_stages/siglip-finetuned/norm_stats.csv`。

## F076 — clip_tsne_conditions_lit_coco

图注：CLIP 在 coco val 的 conditions_lit 联合 TSNE 投影。

本图二维图文质心距离最小的面板是 lit/noise（65.23），最大的为 lit/occl（65.82）；这些数值仅描述本投影。

每个面板显示同一批 1000 张图像及其首条文本，灰线连接固定抽取的 30 对图文。

二维距离受降维影响，不能替代原始嵌入的模态间隙，也不能跨图或跨模型比较大小。

文件：`results/p9/figures/clip_tsne_conditions_lit_coco.png`；来源：`results/p9/coordinates/clip_tsne_conditions_lit_coco.csv`。

## F077 — clip_tsne_conditions_lit_coco_classes

图注：CLIP 在 coco val 的 conditions_lit 联合 TSNE 投影。

采样图像中最多的三个类别（COCO 类别 ID）及数量为：0: 333、60: 48、6: 34。

同一类别的图像与首条文本使用相同颜色；只给数量最多的十类着色，其余点为灰色。

类别标签随图像 ID 固定，所有面板使用同一批样本。

文件：`results/p9/figures/clip_tsne_conditions_lit_coco_classes.png`；来源：`results/p9/coordinates/clip_tsne_conditions_lit_coco.csv`。

## F078 — clip_tsne_conditions_lit_flickr

图注：CLIP 在 flickr val 的 conditions_lit 联合 TSNE 投影。

本图二维图文质心距离最小的面板是 lit/clean（75.85），最大的为 lit/noise（76.30）；这些数值仅描述本投影。

每个面板显示同一批 1000 张图像及其首条文本，灰线连接固定抽取的 30 对图文。

二维距离受降维影响，不能替代原始嵌入的模态间隙，也不能跨图或跨模型比较大小。

文件：`results/p9/figures/clip_tsne_conditions_lit_flickr.png`；来源：`results/p9/coordinates/clip_tsne_conditions_lit_flickr.csv`。

## F079 — clip_tsne_conditions_lora_r16_coco

图注：CLIP 在 coco val 的 conditions_lora_r16 联合 TSNE 投影。

本图二维图文质心距离最小的面板是 lora_r16/occl（68.68），最大的为 lora_r16/clean（69.13）；这些数值仅描述本投影。

每个面板显示同一批 1000 张图像及其首条文本，灰线连接固定抽取的 30 对图文。

二维距离受降维影响，不能替代原始嵌入的模态间隙，也不能跨图或跨模型比较大小。

文件：`results/p9/figures/clip_tsne_conditions_lora_r16_coco.png`；来源：`results/p9/coordinates/clip_tsne_conditions_lora_r16_coco.csv`。

## F080 — clip_tsne_conditions_lora_r16_coco_classes

图注：CLIP 在 coco val 的 conditions_lora_r16 联合 TSNE 投影。

采样图像中最多的三个类别（COCO 类别 ID）及数量为：0: 333、60: 48、6: 34。

同一类别的图像与首条文本使用相同颜色；只给数量最多的十类着色，其余点为灰色。

类别标签随图像 ID 固定，所有面板使用同一批样本。

文件：`results/p9/figures/clip_tsne_conditions_lora_r16_coco_classes.png`；来源：`results/p9/coordinates/clip_tsne_conditions_lora_r16_coco.csv`。

## F081 — clip_tsne_conditions_lora_r16_flickr

图注：CLIP 在 flickr val 的 conditions_lora_r16 联合 TSNE 投影。

本图二维图文质心距离最小的面板是 lora_r16/clean（80.69），最大的为 lora_r16/occl（80.82）；这些数值仅描述本投影。

每个面板显示同一批 1000 张图像及其首条文本，灰线连接固定抽取的 30 对图文。

二维距离受降维影响，不能替代原始嵌入的模态间隙，也不能跨图或跨模型比较大小。

文件：`results/p9/figures/clip_tsne_conditions_lora_r16_flickr.png`；来源：`results/p9/coordinates/clip_tsne_conditions_lora_r16_flickr.csv`。

## F082 — clip_tsne_conditions_lora_r8_coco

图注：CLIP 在 coco val 的 conditions_lora_r8 联合 TSNE 投影。

本图二维图文质心距离最小的面板是 lora_r8/occl（70.55），最大的为 lora_r8/clean（70.59）；这些数值仅描述本投影。

每个面板显示同一批 1000 张图像及其首条文本，灰线连接固定抽取的 30 对图文。

二维距离受降维影响，不能替代原始嵌入的模态间隙，也不能跨图或跨模型比较大小。

文件：`results/p9/figures/clip_tsne_conditions_lora_r8_coco.png`；来源：`results/p9/coordinates/clip_tsne_conditions_lora_r8_coco.csv`。

## F083 — clip_tsne_conditions_lora_r8_coco_classes

图注：CLIP 在 coco val 的 conditions_lora_r8 联合 TSNE 投影。

采样图像中最多的三个类别（COCO 类别 ID）及数量为：0: 333、60: 48、6: 34。

同一类别的图像与首条文本使用相同颜色；只给数量最多的十类着色，其余点为灰色。

类别标签随图像 ID 固定，所有面板使用同一批样本。

文件：`results/p9/figures/clip_tsne_conditions_lora_r8_coco_classes.png`；来源：`results/p9/coordinates/clip_tsne_conditions_lora_r8_coco.csv`。

## F084 — clip_tsne_conditions_lora_r8_flickr

图注：CLIP 在 flickr val 的 conditions_lora_r8 联合 TSNE 投影。

本图二维图文质心距离最小的面板是 lora_r8/clean（77.98），最大的为 lora_r8/occl（78.55）；这些数值仅描述本投影。

每个面板显示同一批 1000 张图像及其首条文本，灰线连接固定抽取的 30 对图文。

二维距离受降维影响，不能替代原始嵌入的模态间隙，也不能跨图或跨模型比较大小。

文件：`results/p9/figures/clip_tsne_conditions_lora_r8_flickr.png`；来源：`results/p9/coordinates/clip_tsne_conditions_lora_r8_flickr.csv`。

## F085 — clip_tsne_conditions_pretrained_coco

图注：CLIP 在 coco val 的 conditions_pretrained 联合 TSNE 投影。

本图二维图文质心距离最小的面板是 pretrained/occl（68.70），最大的为 pretrained/noise（70.00）；这些数值仅描述本投影。

每个面板显示同一批 1000 张图像及其首条文本，灰线连接固定抽取的 30 对图文。

二维距离受降维影响，不能替代原始嵌入的模态间隙，也不能跨图或跨模型比较大小。

文件：`results/p9/figures/clip_tsne_conditions_pretrained_coco.png`；来源：`results/p9/coordinates/clip_tsne_conditions_pretrained_coco.csv`。

## F086 — clip_tsne_conditions_pretrained_coco_classes

图注：CLIP 在 coco val 的 conditions_pretrained 联合 TSNE 投影。

采样图像中最多的三个类别（COCO 类别 ID）及数量为：0: 333、60: 48、6: 34。

同一类别的图像与首条文本使用相同颜色；只给数量最多的十类着色，其余点为灰色。

类别标签随图像 ID 固定，所有面板使用同一批样本。

文件：`results/p9/figures/clip_tsne_conditions_pretrained_coco_classes.png`；来源：`results/p9/coordinates/clip_tsne_conditions_pretrained_coco.csv`。

## F087 — clip_tsne_conditions_pretrained_flickr

图注：CLIP 在 flickr val 的 conditions_pretrained 联合 TSNE 投影。

本图二维图文质心距离最小的面板是 pretrained/noise（74.19），最大的为 pretrained/occl（75.87）；这些数值仅描述本投影。

每个面板显示同一批 1000 张图像及其首条文本，灰线连接固定抽取的 30 对图文。

二维距离受降维影响，不能替代原始嵌入的模态间隙，也不能跨图或跨模型比较大小。

文件：`results/p9/figures/clip_tsne_conditions_pretrained_flickr.png`；来源：`results/p9/coordinates/clip_tsne_conditions_pretrained_flickr.csv`。

## F088 — clip_tsne_settings_clean_coco

图注：CLIP 在 coco val 的 settings_clean 联合 TSNE 投影。

本图二维图文质心距离最小的面板是 lit/clean（85.94），最大的为 pretrained/clean（86.32）；这些数值仅描述本投影。

每个面板显示同一批 1000 张图像及其首条文本，灰线连接固定抽取的 30 对图文。

二维距离受降维影响，不能替代原始嵌入的模态间隙，也不能跨图或跨模型比较大小。

文件：`results/p9/figures/clip_tsne_settings_clean_coco.png`；来源：`results/p9/coordinates/clip_tsne_settings_clean_coco.csv`。

## F089 — clip_tsne_settings_clean_coco_classes

图注：CLIP 在 coco val 的 settings_clean 联合 TSNE 投影。

采样图像中最多的三个类别（COCO 类别 ID）及数量为：0: 333、60: 48、6: 34。

同一类别的图像与首条文本使用相同颜色；只给数量最多的十类着色，其余点为灰色。

类别标签随图像 ID 固定，所有面板使用同一批样本。

文件：`results/p9/figures/clip_tsne_settings_clean_coco_classes.png`；来源：`results/p9/coordinates/clip_tsne_settings_clean_coco.csv`。

## F090 — clip_tsne_settings_clean_flickr

图注：CLIP 在 flickr val 的 settings_clean 联合 TSNE 投影。

本图二维图文质心距离最小的面板是 lit/clean（91.39），最大的为 lora_r16/clean（91.45）；这些数值仅描述本投影。

每个面板显示同一批 1000 张图像及其首条文本，灰线连接固定抽取的 30 对图文。

二维距离受降维影响，不能替代原始嵌入的模态间隙，也不能跨图或跨模型比较大小。

文件：`results/p9/figures/clip_tsne_settings_clean_flickr.png`；来源：`results/p9/coordinates/clip_tsne_settings_clean_flickr.csv`。

## F091 — clip_tsne_settings_noise_coco

图注：CLIP 在 coco val 的 settings_noise 联合 TSNE 投影。

本图二维图文质心距离最小的面板是 lora_r16/noise（87.03），最大的为 lit/noise（87.76）；这些数值仅描述本投影。

每个面板显示同一批 1000 张图像及其首条文本，灰线连接固定抽取的 30 对图文。

二维距离受降维影响，不能替代原始嵌入的模态间隙，也不能跨图或跨模型比较大小。

文件：`results/p9/figures/clip_tsne_settings_noise_coco.png`；来源：`results/p9/coordinates/clip_tsne_settings_noise_coco.csv`。

## F092 — clip_tsne_settings_noise_coco_classes

图注：CLIP 在 coco val 的 settings_noise 联合 TSNE 投影。

采样图像中最多的三个类别（COCO 类别 ID）及数量为：0: 333、60: 48、6: 34。

同一类别的图像与首条文本使用相同颜色；只给数量最多的十类着色，其余点为灰色。

类别标签随图像 ID 固定，所有面板使用同一批样本。

文件：`results/p9/figures/clip_tsne_settings_noise_coco_classes.png`；来源：`results/p9/coordinates/clip_tsne_settings_noise_coco.csv`。

## F093 — clip_tsne_settings_noise_flickr

图注：CLIP 在 flickr val 的 settings_noise 联合 TSNE 投影。

本图二维图文质心距离最小的面板是 pretrained/noise（91.18），最大的为 lit/noise（91.22）；这些数值仅描述本投影。

每个面板显示同一批 1000 张图像及其首条文本，灰线连接固定抽取的 30 对图文。

二维距离受降维影响，不能替代原始嵌入的模态间隙，也不能跨图或跨模型比较大小。

文件：`results/p9/figures/clip_tsne_settings_noise_flickr.png`；来源：`results/p9/coordinates/clip_tsne_settings_noise_flickr.csv`。

## F094 — clip_tsne_settings_occl_coco

图注：CLIP 在 coco val 的 settings_occl 联合 TSNE 投影。

本图二维图文质心距离最小的面板是 lit/occl（86.14），最大的为 pretrained/occl（86.78）；这些数值仅描述本投影。

每个面板显示同一批 1000 张图像及其首条文本，灰线连接固定抽取的 30 对图文。

二维距离受降维影响，不能替代原始嵌入的模态间隙，也不能跨图或跨模型比较大小。

文件：`results/p9/figures/clip_tsne_settings_occl_coco.png`；来源：`results/p9/coordinates/clip_tsne_settings_occl_coco.csv`。

## F095 — clip_tsne_settings_occl_coco_classes

图注：CLIP 在 coco val 的 settings_occl 联合 TSNE 投影。

采样图像中最多的三个类别（COCO 类别 ID）及数量为：0: 333、60: 48、6: 34。

同一类别的图像与首条文本使用相同颜色；只给数量最多的十类着色，其余点为灰色。

类别标签随图像 ID 固定，所有面板使用同一批样本。

文件：`results/p9/figures/clip_tsne_settings_occl_coco_classes.png`；来源：`results/p9/coordinates/clip_tsne_settings_occl_coco.csv`。

## F096 — clip_tsne_settings_occl_flickr

图注：CLIP 在 flickr val 的 settings_occl 联合 TSNE 投影。

本图二维图文质心距离最小的面板是 pretrained/occl（91.39），最大的为 lora_r16/occl（92.05）；这些数值仅描述本投影。

每个面板显示同一批 1000 张图像及其首条文本，灰线连接固定抽取的 30 对图文。

二维距离受降维影响，不能替代原始嵌入的模态间隙，也不能跨图或跨模型比较大小。

文件：`results/p9/figures/clip_tsne_settings_occl_flickr.png`；来源：`results/p9/coordinates/clip_tsne_settings_occl_flickr.csv`。

## F097 — clip_tsne_strength_pretrained_flickr

图注：CLIP 在 flickr val 的 strength_pretrained 联合 TSNE 投影。

本图二维图文质心距离最小的面板是 pretrained/noise_hi（91.99），最大的为 pretrained/occl（94.36）；这些数值仅描述本投影。

每个面板显示同一批 1000 张图像及其首条文本，灰线连接固定抽取的 30 对图文。黑色箭头表示所选图像从 clean 到该扰动强度的位移；文本坐标在各面板完全一致。

二维距离受降维影响，不能替代原始嵌入的模态间隙，也不能跨图或跨模型比较大小。

文件：`results/p9/figures/clip_tsne_strength_pretrained_flickr.png`；来源：`results/p9/coordinates/clip_tsne_strength_pretrained_flickr.csv`。

## F098 — clip_umap_conditions_lit_coco

图注：CLIP 在 coco val 的 conditions_lit 联合 UMAP 投影。

本图二维图文质心距离最小的面板是 lit/clean（17.30），最大的为 lit/noise（17.35）；这些数值仅描述本投影。

每个面板显示同一批 1000 张图像及其首条文本，灰线连接固定抽取的 30 对图文。

二维距离受降维影响，不能替代原始嵌入的模态间隙，也不能跨图或跨模型比较大小。

文件：`results/p9/figures/clip_umap_conditions_lit_coco.png`；来源：`results/p9/coordinates/clip_umap_conditions_lit_coco.csv`。

## F099 — clip_umap_conditions_lit_coco_classes

图注：CLIP 在 coco val 的 conditions_lit 联合 UMAP 投影。

采样图像中最多的三个类别（COCO 类别 ID）及数量为：0: 333、60: 48、6: 34。

同一类别的图像与首条文本使用相同颜色；只给数量最多的十类着色，其余点为灰色。

类别标签随图像 ID 固定，所有面板使用同一批样本。

文件：`results/p9/figures/clip_umap_conditions_lit_coco_classes.png`；来源：`results/p9/coordinates/clip_umap_conditions_lit_coco.csv`。

## F100 — clip_umap_conditions_lit_flickr

图注：CLIP 在 flickr val 的 conditions_lit 联合 UMAP 投影。

本图二维图文质心距离最小的面板是 lit/clean（17.49），最大的为 lit/occl（17.51）；这些数值仅描述本投影。

每个面板显示同一批 1000 张图像及其首条文本，灰线连接固定抽取的 30 对图文。

二维距离受降维影响，不能替代原始嵌入的模态间隙，也不能跨图或跨模型比较大小。

文件：`results/p9/figures/clip_umap_conditions_lit_flickr.png`；来源：`results/p9/coordinates/clip_umap_conditions_lit_flickr.csv`。

## F101 — clip_umap_conditions_lora_r16_coco

图注：CLIP 在 coco val 的 conditions_lora_r16 联合 UMAP 投影。

本图二维图文质心距离最小的面板是 lora_r16/occl（19.17），最大的为 lora_r16/clean（19.21）；这些数值仅描述本投影。

每个面板显示同一批 1000 张图像及其首条文本，灰线连接固定抽取的 30 对图文。

二维距离受降维影响，不能替代原始嵌入的模态间隙，也不能跨图或跨模型比较大小。

文件：`results/p9/figures/clip_umap_conditions_lora_r16_coco.png`；来源：`results/p9/coordinates/clip_umap_conditions_lora_r16_coco.csv`。

## F102 — clip_umap_conditions_lora_r16_coco_classes

图注：CLIP 在 coco val 的 conditions_lora_r16 联合 UMAP 投影。

采样图像中最多的三个类别（COCO 类别 ID）及数量为：0: 333、60: 48、6: 34。

同一类别的图像与首条文本使用相同颜色；只给数量最多的十类着色，其余点为灰色。

类别标签随图像 ID 固定，所有面板使用同一批样本。

文件：`results/p9/figures/clip_umap_conditions_lora_r16_coco_classes.png`；来源：`results/p9/coordinates/clip_umap_conditions_lora_r16_coco.csv`。

## F103 — clip_umap_conditions_lora_r16_flickr

图注：CLIP 在 flickr val 的 conditions_lora_r16 联合 UMAP 投影。

本图二维图文质心距离最小的面板是 lora_r16/clean（21.50），最大的为 lora_r16/occl（21.51）；这些数值仅描述本投影。

每个面板显示同一批 1000 张图像及其首条文本，灰线连接固定抽取的 30 对图文。

二维距离受降维影响，不能替代原始嵌入的模态间隙，也不能跨图或跨模型比较大小。

文件：`results/p9/figures/clip_umap_conditions_lora_r16_flickr.png`；来源：`results/p9/coordinates/clip_umap_conditions_lora_r16_flickr.csv`。

## F104 — clip_umap_conditions_lora_r8_coco

图注：CLIP 在 coco val 的 conditions_lora_r8 联合 UMAP 投影。

本图二维图文质心距离最小的面板是 lora_r8/occl（18.67），最大的为 lora_r8/clean（18.74）；这些数值仅描述本投影。

每个面板显示同一批 1000 张图像及其首条文本，灰线连接固定抽取的 30 对图文。

二维距离受降维影响，不能替代原始嵌入的模态间隙，也不能跨图或跨模型比较大小。

文件：`results/p9/figures/clip_umap_conditions_lora_r8_coco.png`；来源：`results/p9/coordinates/clip_umap_conditions_lora_r8_coco.csv`。

## F105 — clip_umap_conditions_lora_r8_coco_classes

图注：CLIP 在 coco val 的 conditions_lora_r8 联合 UMAP 投影。

采样图像中最多的三个类别（COCO 类别 ID）及数量为：0: 333、60: 48、6: 34。

同一类别的图像与首条文本使用相同颜色；只给数量最多的十类着色，其余点为灰色。

类别标签随图像 ID 固定，所有面板使用同一批样本。

文件：`results/p9/figures/clip_umap_conditions_lora_r8_coco_classes.png`；来源：`results/p9/coordinates/clip_umap_conditions_lora_r8_coco.csv`。

## F106 — clip_umap_conditions_lora_r8_flickr

图注：CLIP 在 flickr val 的 conditions_lora_r8 联合 UMAP 投影。

本图二维图文质心距离最小的面板是 lora_r8/clean（17.07），最大的为 lora_r8/occl（17.11）；这些数值仅描述本投影。

每个面板显示同一批 1000 张图像及其首条文本，灰线连接固定抽取的 30 对图文。

二维距离受降维影响，不能替代原始嵌入的模态间隙，也不能跨图或跨模型比较大小。

文件：`results/p9/figures/clip_umap_conditions_lora_r8_flickr.png`；来源：`results/p9/coordinates/clip_umap_conditions_lora_r8_flickr.csv`。

## F107 — clip_umap_conditions_pretrained_coco

图注：CLIP 在 coco val 的 conditions_pretrained 联合 UMAP 投影。

本图二维图文质心距离最小的面板是 pretrained/clean（18.24），最大的为 pretrained/occl（18.30）；这些数值仅描述本投影。

每个面板显示同一批 1000 张图像及其首条文本，灰线连接固定抽取的 30 对图文。

二维距离受降维影响，不能替代原始嵌入的模态间隙，也不能跨图或跨模型比较大小。

文件：`results/p9/figures/clip_umap_conditions_pretrained_coco.png`；来源：`results/p9/coordinates/clip_umap_conditions_pretrained_coco.csv`。

## F108 — clip_umap_conditions_pretrained_coco_classes

图注：CLIP 在 coco val 的 conditions_pretrained 联合 UMAP 投影。

采样图像中最多的三个类别（COCO 类别 ID）及数量为：0: 333、60: 48、6: 34。

同一类别的图像与首条文本使用相同颜色；只给数量最多的十类着色，其余点为灰色。

类别标签随图像 ID 固定，所有面板使用同一批样本。

文件：`results/p9/figures/clip_umap_conditions_pretrained_coco_classes.png`；来源：`results/p9/coordinates/clip_umap_conditions_pretrained_coco.csv`。

## F109 — clip_umap_conditions_pretrained_flickr

图注：CLIP 在 flickr val 的 conditions_pretrained 联合 UMAP 投影。

本图二维图文质心距离最小的面板是 pretrained/noise（17.70），最大的为 pretrained/occl（17.79）；这些数值仅描述本投影。

每个面板显示同一批 1000 张图像及其首条文本，灰线连接固定抽取的 30 对图文。

二维距离受降维影响，不能替代原始嵌入的模态间隙，也不能跨图或跨模型比较大小。

文件：`results/p9/figures/clip_umap_conditions_pretrained_flickr.png`；来源：`results/p9/coordinates/clip_umap_conditions_pretrained_flickr.csv`。

## F110 — clip_umap_settings_clean_coco

图注：CLIP 在 coco val 的 settings_clean 联合 UMAP 投影。

本图二维图文质心距离最小的面板是 lit/clean（16.79），最大的为 pretrained/clean（16.85）；这些数值仅描述本投影。

每个面板显示同一批 1000 张图像及其首条文本，灰线连接固定抽取的 30 对图文。

二维距离受降维影响，不能替代原始嵌入的模态间隙，也不能跨图或跨模型比较大小。

文件：`results/p9/figures/clip_umap_settings_clean_coco.png`；来源：`results/p9/coordinates/clip_umap_settings_clean_coco.csv`。

## F111 — clip_umap_settings_clean_coco_classes

图注：CLIP 在 coco val 的 settings_clean 联合 UMAP 投影。

采样图像中最多的三个类别（COCO 类别 ID）及数量为：0: 333、60: 48、6: 34。

同一类别的图像与首条文本使用相同颜色；只给数量最多的十类着色，其余点为灰色。

类别标签随图像 ID 固定，所有面板使用同一批样本。

文件：`results/p9/figures/clip_umap_settings_clean_coco_classes.png`；来源：`results/p9/coordinates/clip_umap_settings_clean_coco.csv`。

## F112 — clip_umap_settings_clean_flickr

图注：CLIP 在 flickr val 的 settings_clean 联合 UMAP 投影。

本图二维图文质心距离最小的面板是 lora_r16/clean（18.23），最大的为 lit/clean（18.24）；这些数值仅描述本投影。

每个面板显示同一批 1000 张图像及其首条文本，灰线连接固定抽取的 30 对图文。

二维距离受降维影响，不能替代原始嵌入的模态间隙，也不能跨图或跨模型比较大小。

文件：`results/p9/figures/clip_umap_settings_clean_flickr.png`；来源：`results/p9/coordinates/clip_umap_settings_clean_flickr.csv`。

## F113 — clip_umap_settings_noise_coco

图注：CLIP 在 coco val 的 settings_noise 联合 UMAP 投影。

本图二维图文质心距离最小的面板是 pretrained/noise（17.10），最大的为 lit/noise（17.22）；这些数值仅描述本投影。

每个面板显示同一批 1000 张图像及其首条文本，灰线连接固定抽取的 30 对图文。

二维距离受降维影响，不能替代原始嵌入的模态间隙，也不能跨图或跨模型比较大小。

文件：`results/p9/figures/clip_umap_settings_noise_coco.png`；来源：`results/p9/coordinates/clip_umap_settings_noise_coco.csv`。

## F114 — clip_umap_settings_noise_coco_classes

图注：CLIP 在 coco val 的 settings_noise 联合 UMAP 投影。

采样图像中最多的三个类别（COCO 类别 ID）及数量为：0: 333、60: 48、6: 34。

同一类别的图像与首条文本使用相同颜色；只给数量最多的十类着色，其余点为灰色。

类别标签随图像 ID 固定，所有面板使用同一批样本。

文件：`results/p9/figures/clip_umap_settings_noise_coco_classes.png`；来源：`results/p9/coordinates/clip_umap_settings_noise_coco.csv`。

## F115 — clip_umap_settings_noise_flickr

图注：CLIP 在 flickr val 的 settings_noise 联合 UMAP 投影。

本图二维图文质心距离最小的面板是 lit/noise（17.80），最大的为 pretrained/noise（17.81）；这些数值仅描述本投影。

每个面板显示同一批 1000 张图像及其首条文本，灰线连接固定抽取的 30 对图文。

二维距离受降维影响，不能替代原始嵌入的模态间隙，也不能跨图或跨模型比较大小。

文件：`results/p9/figures/clip_umap_settings_noise_flickr.png`；来源：`results/p9/coordinates/clip_umap_settings_noise_flickr.csv`。

## F116 — clip_umap_settings_occl_coco

图注：CLIP 在 coco val 的 settings_occl 联合 UMAP 投影。

本图二维图文质心距离最小的面板是 lora_r16/occl（16.88），最大的为 lit/occl（16.95）；这些数值仅描述本投影。

每个面板显示同一批 1000 张图像及其首条文本，灰线连接固定抽取的 30 对图文。

二维距离受降维影响，不能替代原始嵌入的模态间隙，也不能跨图或跨模型比较大小。

文件：`results/p9/figures/clip_umap_settings_occl_coco.png`；来源：`results/p9/coordinates/clip_umap_settings_occl_coco.csv`。

## F117 — clip_umap_settings_occl_coco_classes

图注：CLIP 在 coco val 的 settings_occl 联合 UMAP 投影。

采样图像中最多的三个类别（COCO 类别 ID）及数量为：0: 333、60: 48、6: 34。

同一类别的图像与首条文本使用相同颜色；只给数量最多的十类着色，其余点为灰色。

类别标签随图像 ID 固定，所有面板使用同一批样本。

文件：`results/p9/figures/clip_umap_settings_occl_coco_classes.png`；来源：`results/p9/coordinates/clip_umap_settings_occl_coco.csv`。

## F118 — clip_umap_settings_occl_flickr

图注：CLIP 在 flickr val 的 settings_occl 联合 UMAP 投影。

本图二维图文质心距离最小的面板是 pretrained/occl（15.10），最大的为 lit/occl（15.10）；这些数值仅描述本投影。

每个面板显示同一批 1000 张图像及其首条文本，灰线连接固定抽取的 30 对图文。

二维距离受降维影响，不能替代原始嵌入的模态间隙，也不能跨图或跨模型比较大小。

文件：`results/p9/figures/clip_umap_settings_occl_flickr.png`；来源：`results/p9/coordinates/clip_umap_settings_occl_flickr.csv`。

## F119 — clip_umap_strength_pretrained_flickr

图注：CLIP 在 flickr val 的 strength_pretrained 联合 UMAP 投影。

本图二维图文质心距离最小的面板是 pretrained/noise_hi（17.06），最大的为 pretrained/occl_hi（17.37）；这些数值仅描述本投影。

每个面板显示同一批 1000 张图像及其首条文本，灰线连接固定抽取的 30 对图文。黑色箭头表示所选图像从 clean 到该扰动强度的位移；文本坐标在各面板完全一致。

二维距离受降维影响，不能替代原始嵌入的模态间隙，也不能跨图或跨模型比较大小。

文件：`results/p9/figures/clip_umap_strength_pretrained_flickr.png`；来源：`results/p9/coordinates/clip_umap_strength_pretrained_flickr.csv`。

## F120 — siglip_tsne_conditions_lit_coco

图注：SIGLIP 在 coco val 的 conditions_lit 联合 TSNE 投影。

本图二维图文质心距离最小的面板是 lit/occl（69.11），最大的为 lit/clean（69.65）；这些数值仅描述本投影。

每个面板显示同一批 1000 张图像及其首条文本，灰线连接固定抽取的 30 对图文。

二维距离受降维影响，不能替代原始嵌入的模态间隙，也不能跨图或跨模型比较大小。

文件：`results/p9/figures/siglip_tsne_conditions_lit_coco.png`；来源：`results/p9/coordinates/siglip_tsne_conditions_lit_coco.csv`。

## F121 — siglip_tsne_conditions_lit_coco_classes

图注：SIGLIP 在 coco val 的 conditions_lit 联合 TSNE 投影。

采样图像中最多的三个类别（COCO 类别 ID）及数量为：0: 333、60: 48、6: 34。

同一类别的图像与首条文本使用相同颜色；只给数量最多的十类着色，其余点为灰色。

类别标签随图像 ID 固定，所有面板使用同一批样本。

文件：`results/p9/figures/siglip_tsne_conditions_lit_coco_classes.png`；来源：`results/p9/coordinates/siglip_tsne_conditions_lit_coco.csv`。

## F122 — siglip_tsne_conditions_lit_flickr

图注：SIGLIP 在 flickr val 的 conditions_lit 联合 TSNE 投影。

本图二维图文质心距离最小的面板是 lit/noise（75.59），最大的为 lit/occl（76.33）；这些数值仅描述本投影。

每个面板显示同一批 1000 张图像及其首条文本，灰线连接固定抽取的 30 对图文。

二维距离受降维影响，不能替代原始嵌入的模态间隙，也不能跨图或跨模型比较大小。

文件：`results/p9/figures/siglip_tsne_conditions_lit_flickr.png`；来源：`results/p9/coordinates/siglip_tsne_conditions_lit_flickr.csv`。

## F123 — siglip_tsne_conditions_lora_r16_coco

图注：SIGLIP 在 coco val 的 conditions_lora_r16 联合 TSNE 投影。

本图二维图文质心距离最小的面板是 lora_r16/occl（70.27），最大的为 lora_r16/noise（70.44）；这些数值仅描述本投影。

每个面板显示同一批 1000 张图像及其首条文本，灰线连接固定抽取的 30 对图文。

二维距离受降维影响，不能替代原始嵌入的模态间隙，也不能跨图或跨模型比较大小。

文件：`results/p9/figures/siglip_tsne_conditions_lora_r16_coco.png`；来源：`results/p9/coordinates/siglip_tsne_conditions_lora_r16_coco.csv`。

## F124 — siglip_tsne_conditions_lora_r16_coco_classes

图注：SIGLIP 在 coco val 的 conditions_lora_r16 联合 TSNE 投影。

采样图像中最多的三个类别（COCO 类别 ID）及数量为：0: 333、60: 48、6: 34。

同一类别的图像与首条文本使用相同颜色；只给数量最多的十类着色，其余点为灰色。

类别标签随图像 ID 固定，所有面板使用同一批样本。

文件：`results/p9/figures/siglip_tsne_conditions_lora_r16_coco_classes.png`；来源：`results/p9/coordinates/siglip_tsne_conditions_lora_r16_coco.csv`。

## F125 — siglip_tsne_conditions_lora_r16_flickr

图注：SIGLIP 在 flickr val 的 conditions_lora_r16 联合 TSNE 投影。

本图二维图文质心距离最小的面板是 lora_r16/clean（79.08），最大的为 lora_r16/occl（80.72）；这些数值仅描述本投影。

每个面板显示同一批 1000 张图像及其首条文本，灰线连接固定抽取的 30 对图文。

二维距离受降维影响，不能替代原始嵌入的模态间隙，也不能跨图或跨模型比较大小。

文件：`results/p9/figures/siglip_tsne_conditions_lora_r16_flickr.png`；来源：`results/p9/coordinates/siglip_tsne_conditions_lora_r16_flickr.csv`。

## F126 — siglip_tsne_conditions_lora_r8_coco

图注：SIGLIP 在 coco val 的 conditions_lora_r8 联合 TSNE 投影。

本图二维图文质心距离最小的面板是 lora_r8/occl（68.86），最大的为 lora_r8/noise（69.15）；这些数值仅描述本投影。

每个面板显示同一批 1000 张图像及其首条文本，灰线连接固定抽取的 30 对图文。

二维距离受降维影响，不能替代原始嵌入的模态间隙，也不能跨图或跨模型比较大小。

文件：`results/p9/figures/siglip_tsne_conditions_lora_r8_coco.png`；来源：`results/p9/coordinates/siglip_tsne_conditions_lora_r8_coco.csv`。

## F127 — siglip_tsne_conditions_lora_r8_coco_classes

图注：SIGLIP 在 coco val 的 conditions_lora_r8 联合 TSNE 投影。

采样图像中最多的三个类别（COCO 类别 ID）及数量为：0: 333、60: 48、6: 34。

同一类别的图像与首条文本使用相同颜色；只给数量最多的十类着色，其余点为灰色。

类别标签随图像 ID 固定，所有面板使用同一批样本。

文件：`results/p9/figures/siglip_tsne_conditions_lora_r8_coco_classes.png`；来源：`results/p9/coordinates/siglip_tsne_conditions_lora_r8_coco.csv`。

## F128 — siglip_tsne_conditions_lora_r8_flickr

图注：SIGLIP 在 flickr val 的 conditions_lora_r8 联合 TSNE 投影。

本图二维图文质心距离最小的面板是 lora_r8/occl（78.23），最大的为 lora_r8/clean（79.31）；这些数值仅描述本投影。

每个面板显示同一批 1000 张图像及其首条文本，灰线连接固定抽取的 30 对图文。

二维距离受降维影响，不能替代原始嵌入的模态间隙，也不能跨图或跨模型比较大小。

文件：`results/p9/figures/siglip_tsne_conditions_lora_r8_flickr.png`；来源：`results/p9/coordinates/siglip_tsne_conditions_lora_r8_flickr.csv`。

## F129 — siglip_tsne_conditions_pretrained_coco

图注：SIGLIP 在 coco val 的 conditions_pretrained 联合 TSNE 投影。

本图二维图文质心距离最小的面板是 pretrained/occl（68.00），最大的为 pretrained/clean（68.53）；这些数值仅描述本投影。

每个面板显示同一批 1000 张图像及其首条文本，灰线连接固定抽取的 30 对图文。

二维距离受降维影响，不能替代原始嵌入的模态间隙，也不能跨图或跨模型比较大小。

文件：`results/p9/figures/siglip_tsne_conditions_pretrained_coco.png`；来源：`results/p9/coordinates/siglip_tsne_conditions_pretrained_coco.csv`。

## F130 — siglip_tsne_conditions_pretrained_coco_classes

图注：SIGLIP 在 coco val 的 conditions_pretrained 联合 TSNE 投影。

采样图像中最多的三个类别（COCO 类别 ID）及数量为：0: 333、60: 48、6: 34。

同一类别的图像与首条文本使用相同颜色；只给数量最多的十类着色，其余点为灰色。

类别标签随图像 ID 固定，所有面板使用同一批样本。

文件：`results/p9/figures/siglip_tsne_conditions_pretrained_coco_classes.png`；来源：`results/p9/coordinates/siglip_tsne_conditions_pretrained_coco.csv`。

## F131 — siglip_tsne_conditions_pretrained_flickr

图注：SIGLIP 在 flickr val 的 conditions_pretrained 联合 TSNE 投影。

本图二维图文质心距离最小的面板是 pretrained/noise（71.55），最大的为 pretrained/occl（72.01）；这些数值仅描述本投影。

每个面板显示同一批 1000 张图像及其首条文本，灰线连接固定抽取的 30 对图文。

二维距离受降维影响，不能替代原始嵌入的模态间隙，也不能跨图或跨模型比较大小。

文件：`results/p9/figures/siglip_tsne_conditions_pretrained_flickr.png`；来源：`results/p9/coordinates/siglip_tsne_conditions_pretrained_flickr.csv`。

## F132 — siglip_tsne_settings_clean_coco

图注：SIGLIP 在 coco val 的 settings_clean 联合 TSNE 投影。

本图二维图文质心距离最小的面板是 lora_r16/clean（86.99），最大的为 lit/clean（87.18）；这些数值仅描述本投影。

每个面板显示同一批 1000 张图像及其首条文本，灰线连接固定抽取的 30 对图文。

二维距离受降维影响，不能替代原始嵌入的模态间隙，也不能跨图或跨模型比较大小。

文件：`results/p9/figures/siglip_tsne_settings_clean_coco.png`；来源：`results/p9/coordinates/siglip_tsne_settings_clean_coco.csv`。

## F133 — siglip_tsne_settings_clean_coco_classes

图注：SIGLIP 在 coco val 的 settings_clean 联合 TSNE 投影。

采样图像中最多的三个类别（COCO 类别 ID）及数量为：0: 333、60: 48、6: 34。

同一类别的图像与首条文本使用相同颜色；只给数量最多的十类着色，其余点为灰色。

类别标签随图像 ID 固定，所有面板使用同一批样本。

文件：`results/p9/figures/siglip_tsne_settings_clean_coco_classes.png`；来源：`results/p9/coordinates/siglip_tsne_settings_clean_coco.csv`。

## F134 — siglip_tsne_settings_clean_flickr

图注：SIGLIP 在 flickr val 的 settings_clean 联合 TSNE 投影。

本图二维图文质心距离最小的面板是 lora_r8/clean（92.22），最大的为 pretrained/clean（92.29）；这些数值仅描述本投影。

每个面板显示同一批 1000 张图像及其首条文本，灰线连接固定抽取的 30 对图文。

二维距离受降维影响，不能替代原始嵌入的模态间隙，也不能跨图或跨模型比较大小。

文件：`results/p9/figures/siglip_tsne_settings_clean_flickr.png`；来源：`results/p9/coordinates/siglip_tsne_settings_clean_flickr.csv`。

## F135 — siglip_tsne_settings_noise_coco

图注：SIGLIP 在 coco val 的 settings_noise 联合 TSNE 投影。

本图二维图文质心距离最小的面板是 pretrained/noise（86.98），最大的为 lora_r16/noise（87.14）；这些数值仅描述本投影。

每个面板显示同一批 1000 张图像及其首条文本，灰线连接固定抽取的 30 对图文。

二维距离受降维影响，不能替代原始嵌入的模态间隙，也不能跨图或跨模型比较大小。

文件：`results/p9/figures/siglip_tsne_settings_noise_coco.png`；来源：`results/p9/coordinates/siglip_tsne_settings_noise_coco.csv`。

## F136 — siglip_tsne_settings_noise_coco_classes

图注：SIGLIP 在 coco val 的 settings_noise 联合 TSNE 投影。

采样图像中最多的三个类别（COCO 类别 ID）及数量为：0: 333、60: 48、6: 34。

同一类别的图像与首条文本使用相同颜色；只给数量最多的十类着色，其余点为灰色。

类别标签随图像 ID 固定，所有面板使用同一批样本。

文件：`results/p9/figures/siglip_tsne_settings_noise_coco_classes.png`；来源：`results/p9/coordinates/siglip_tsne_settings_noise_coco.csv`。

## F137 — siglip_tsne_settings_noise_flickr

图注：SIGLIP 在 flickr val 的 settings_noise 联合 TSNE 投影。

本图二维图文质心距离最小的面板是 pretrained/noise（90.57），最大的为 lit/noise（90.66）；这些数值仅描述本投影。

每个面板显示同一批 1000 张图像及其首条文本，灰线连接固定抽取的 30 对图文。

二维距离受降维影响，不能替代原始嵌入的模态间隙，也不能跨图或跨模型比较大小。

文件：`results/p9/figures/siglip_tsne_settings_noise_flickr.png`；来源：`results/p9/coordinates/siglip_tsne_settings_noise_flickr.csv`。

## F138 — siglip_tsne_settings_occl_coco

图注：SIGLIP 在 coco val 的 settings_occl 联合 TSNE 投影。

本图二维图文质心距离最小的面板是 lora_r16/occl（87.95），最大的为 lit/occl（88.21）；这些数值仅描述本投影。

每个面板显示同一批 1000 张图像及其首条文本，灰线连接固定抽取的 30 对图文。

二维距离受降维影响，不能替代原始嵌入的模态间隙，也不能跨图或跨模型比较大小。

文件：`results/p9/figures/siglip_tsne_settings_occl_coco.png`；来源：`results/p9/coordinates/siglip_tsne_settings_occl_coco.csv`。

## F139 — siglip_tsne_settings_occl_coco_classes

图注：SIGLIP 在 coco val 的 settings_occl 联合 TSNE 投影。

采样图像中最多的三个类别（COCO 类别 ID）及数量为：0: 333、60: 48、6: 34。

同一类别的图像与首条文本使用相同颜色；只给数量最多的十类着色，其余点为灰色。

类别标签随图像 ID 固定，所有面板使用同一批样本。

文件：`results/p9/figures/siglip_tsne_settings_occl_coco_classes.png`；来源：`results/p9/coordinates/siglip_tsne_settings_occl_coco.csv`。

## F140 — siglip_tsne_settings_occl_flickr

图注：SIGLIP 在 flickr val 的 settings_occl 联合 TSNE 投影。

本图二维图文质心距离最小的面板是 pretrained/occl（90.18），最大的为 lora_r16/occl（90.33）；这些数值仅描述本投影。

每个面板显示同一批 1000 张图像及其首条文本，灰线连接固定抽取的 30 对图文。

二维距离受降维影响，不能替代原始嵌入的模态间隙，也不能跨图或跨模型比较大小。

文件：`results/p9/figures/siglip_tsne_settings_occl_flickr.png`；来源：`results/p9/coordinates/siglip_tsne_settings_occl_flickr.csv`。

## F141 — siglip_umap_conditions_lit_coco

图注：SIGLIP 在 coco val 的 conditions_lit 联合 UMAP 投影。

本图二维图文质心距离最小的面板是 lit/clean（15.35），最大的为 lit/noise（15.38）；这些数值仅描述本投影。

每个面板显示同一批 1000 张图像及其首条文本，灰线连接固定抽取的 30 对图文。

二维距离受降维影响，不能替代原始嵌入的模态间隙，也不能跨图或跨模型比较大小。

文件：`results/p9/figures/siglip_umap_conditions_lit_coco.png`；来源：`results/p9/coordinates/siglip_umap_conditions_lit_coco.csv`。

## F142 — siglip_umap_conditions_lit_coco_classes

图注：SIGLIP 在 coco val 的 conditions_lit 联合 UMAP 投影。

采样图像中最多的三个类别（COCO 类别 ID）及数量为：0: 333、60: 48、6: 34。

同一类别的图像与首条文本使用相同颜色；只给数量最多的十类着色，其余点为灰色。

类别标签随图像 ID 固定，所有面板使用同一批样本。

文件：`results/p9/figures/siglip_umap_conditions_lit_coco_classes.png`；来源：`results/p9/coordinates/siglip_umap_conditions_lit_coco.csv`。

## F143 — siglip_umap_conditions_lit_flickr

图注：SIGLIP 在 flickr val 的 conditions_lit 联合 UMAP 投影。

本图二维图文质心距离最小的面板是 lit/occl（18.59），最大的为 lit/noise（18.64）；这些数值仅描述本投影。

每个面板显示同一批 1000 张图像及其首条文本，灰线连接固定抽取的 30 对图文。

二维距离受降维影响，不能替代原始嵌入的模态间隙，也不能跨图或跨模型比较大小。

文件：`results/p9/figures/siglip_umap_conditions_lit_flickr.png`；来源：`results/p9/coordinates/siglip_umap_conditions_lit_flickr.csv`。

## F144 — siglip_umap_conditions_lora_r16_coco

图注：SIGLIP 在 coco val 的 conditions_lora_r16 联合 UMAP 投影。

本图二维图文质心距离最小的面板是 lora_r16/clean（16.69），最大的为 lora_r16/occl（16.76）；这些数值仅描述本投影。

每个面板显示同一批 1000 张图像及其首条文本，灰线连接固定抽取的 30 对图文。

二维距离受降维影响，不能替代原始嵌入的模态间隙，也不能跨图或跨模型比较大小。

文件：`results/p9/figures/siglip_umap_conditions_lora_r16_coco.png`；来源：`results/p9/coordinates/siglip_umap_conditions_lora_r16_coco.csv`。

## F145 — siglip_umap_conditions_lora_r16_coco_classes

图注：SIGLIP 在 coco val 的 conditions_lora_r16 联合 UMAP 投影。

采样图像中最多的三个类别（COCO 类别 ID）及数量为：0: 333、60: 48、6: 34。

同一类别的图像与首条文本使用相同颜色；只给数量最多的十类着色，其余点为灰色。

类别标签随图像 ID 固定，所有面板使用同一批样本。

文件：`results/p9/figures/siglip_umap_conditions_lora_r16_coco_classes.png`；来源：`results/p9/coordinates/siglip_umap_conditions_lora_r16_coco.csv`。

## F146 — siglip_umap_conditions_lora_r16_flickr

图注：SIGLIP 在 flickr val 的 conditions_lora_r16 联合 UMAP 投影。

本图二维图文质心距离最小的面板是 lora_r16/noise（17.66），最大的为 lora_r16/occl（17.77）；这些数值仅描述本投影。

每个面板显示同一批 1000 张图像及其首条文本，灰线连接固定抽取的 30 对图文。

二维距离受降维影响，不能替代原始嵌入的模态间隙，也不能跨图或跨模型比较大小。

文件：`results/p9/figures/siglip_umap_conditions_lora_r16_flickr.png`；来源：`results/p9/coordinates/siglip_umap_conditions_lora_r16_flickr.csv`。

## F147 — siglip_umap_conditions_lora_r8_coco

图注：SIGLIP 在 coco val 的 conditions_lora_r8 联合 UMAP 投影。

本图二维图文质心距离最小的面板是 lora_r8/noise（15.82），最大的为 lora_r8/clean（15.83）；这些数值仅描述本投影。

每个面板显示同一批 1000 张图像及其首条文本，灰线连接固定抽取的 30 对图文。

二维距离受降维影响，不能替代原始嵌入的模态间隙，也不能跨图或跨模型比较大小。

文件：`results/p9/figures/siglip_umap_conditions_lora_r8_coco.png`；来源：`results/p9/coordinates/siglip_umap_conditions_lora_r8_coco.csv`。

## F148 — siglip_umap_conditions_lora_r8_coco_classes

图注：SIGLIP 在 coco val 的 conditions_lora_r8 联合 UMAP 投影。

采样图像中最多的三个类别（COCO 类别 ID）及数量为：0: 333、60: 48、6: 34。

同一类别的图像与首条文本使用相同颜色；只给数量最多的十类着色，其余点为灰色。

类别标签随图像 ID 固定，所有面板使用同一批样本。

文件：`results/p9/figures/siglip_umap_conditions_lora_r8_coco_classes.png`；来源：`results/p9/coordinates/siglip_umap_conditions_lora_r8_coco.csv`。

## F149 — siglip_umap_conditions_lora_r8_flickr

图注：SIGLIP 在 flickr val 的 conditions_lora_r8 联合 UMAP 投影。

本图二维图文质心距离最小的面板是 lora_r8/occl（16.23），最大的为 lora_r8/clean（16.33）；这些数值仅描述本投影。

每个面板显示同一批 1000 张图像及其首条文本，灰线连接固定抽取的 30 对图文。

二维距离受降维影响，不能替代原始嵌入的模态间隙，也不能跨图或跨模型比较大小。

文件：`results/p9/figures/siglip_umap_conditions_lora_r8_flickr.png`；来源：`results/p9/coordinates/siglip_umap_conditions_lora_r8_flickr.csv`。

## F150 — siglip_umap_conditions_pretrained_coco

图注：SIGLIP 在 coco val 的 conditions_pretrained 联合 UMAP 投影。

本图二维图文质心距离最小的面板是 pretrained/clean（16.25），最大的为 pretrained/occl（16.26）；这些数值仅描述本投影。

每个面板显示同一批 1000 张图像及其首条文本，灰线连接固定抽取的 30 对图文。

二维距离受降维影响，不能替代原始嵌入的模态间隙，也不能跨图或跨模型比较大小。

文件：`results/p9/figures/siglip_umap_conditions_pretrained_coco.png`；来源：`results/p9/coordinates/siglip_umap_conditions_pretrained_coco.csv`。

## F151 — siglip_umap_conditions_pretrained_coco_classes

图注：SIGLIP 在 coco val 的 conditions_pretrained 联合 UMAP 投影。

采样图像中最多的三个类别（COCO 类别 ID）及数量为：0: 333、60: 48、6: 34。

同一类别的图像与首条文本使用相同颜色；只给数量最多的十类着色，其余点为灰色。

类别标签随图像 ID 固定，所有面板使用同一批样本。

文件：`results/p9/figures/siglip_umap_conditions_pretrained_coco_classes.png`；来源：`results/p9/coordinates/siglip_umap_conditions_pretrained_coco.csv`。

## F152 — siglip_umap_conditions_pretrained_flickr

图注：SIGLIP 在 flickr val 的 conditions_pretrained 联合 UMAP 投影。

本图二维图文质心距离最小的面板是 pretrained/occl（15.23），最大的为 pretrained/clean（15.27）；这些数值仅描述本投影。

每个面板显示同一批 1000 张图像及其首条文本，灰线连接固定抽取的 30 对图文。

二维距离受降维影响，不能替代原始嵌入的模态间隙，也不能跨图或跨模型比较大小。

文件：`results/p9/figures/siglip_umap_conditions_pretrained_flickr.png`；来源：`results/p9/coordinates/siglip_umap_conditions_pretrained_flickr.csv`。

## F153 — siglip_umap_settings_clean_coco

图注：SIGLIP 在 coco val 的 settings_clean 联合 UMAP 投影。

本图二维图文质心距离最小的面板是 pretrained/clean（16.64），最大的为 lit/clean（16.66）；这些数值仅描述本投影。

每个面板显示同一批 1000 张图像及其首条文本，灰线连接固定抽取的 30 对图文。

二维距离受降维影响，不能替代原始嵌入的模态间隙，也不能跨图或跨模型比较大小。

文件：`results/p9/figures/siglip_umap_settings_clean_coco.png`；来源：`results/p9/coordinates/siglip_umap_settings_clean_coco.csv`。

## F154 — siglip_umap_settings_clean_coco_classes

图注：SIGLIP 在 coco val 的 settings_clean 联合 UMAP 投影。

采样图像中最多的三个类别（COCO 类别 ID）及数量为：0: 333、60: 48、6: 34。

同一类别的图像与首条文本使用相同颜色；只给数量最多的十类着色，其余点为灰色。

类别标签随图像 ID 固定，所有面板使用同一批样本。

文件：`results/p9/figures/siglip_umap_settings_clean_coco_classes.png`；来源：`results/p9/coordinates/siglip_umap_settings_clean_coco.csv`。

## F155 — siglip_umap_settings_clean_flickr

图注：SIGLIP 在 flickr val 的 settings_clean 联合 UMAP 投影。

本图二维图文质心距离最小的面板是 pretrained/clean（16.56），最大的为 lora_r8/clean（16.56）；这些数值仅描述本投影。

每个面板显示同一批 1000 张图像及其首条文本，灰线连接固定抽取的 30 对图文。

二维距离受降维影响，不能替代原始嵌入的模态间隙，也不能跨图或跨模型比较大小。

文件：`results/p9/figures/siglip_umap_settings_clean_flickr.png`；来源：`results/p9/coordinates/siglip_umap_settings_clean_flickr.csv`。

## F156 — siglip_umap_settings_noise_coco

图注：SIGLIP 在 coco val 的 settings_noise 联合 UMAP 投影。

本图二维图文质心距离最小的面板是 lit/noise（16.44），最大的为 pretrained/noise（16.44）；这些数值仅描述本投影。

每个面板显示同一批 1000 张图像及其首条文本，灰线连接固定抽取的 30 对图文。

二维距离受降维影响，不能替代原始嵌入的模态间隙，也不能跨图或跨模型比较大小。

文件：`results/p9/figures/siglip_umap_settings_noise_coco.png`；来源：`results/p9/coordinates/siglip_umap_settings_noise_coco.csv`。

## F157 — siglip_umap_settings_noise_coco_classes

图注：SIGLIP 在 coco val 的 settings_noise 联合 UMAP 投影。

采样图像中最多的三个类别（COCO 类别 ID）及数量为：0: 333、60: 48、6: 34。

同一类别的图像与首条文本使用相同颜色；只给数量最多的十类着色，其余点为灰色。

类别标签随图像 ID 固定，所有面板使用同一批样本。

文件：`results/p9/figures/siglip_umap_settings_noise_coco_classes.png`；来源：`results/p9/coordinates/siglip_umap_settings_noise_coco.csv`。

## F158 — siglip_umap_settings_noise_flickr

图注：SIGLIP 在 flickr val 的 settings_noise 联合 UMAP 投影。

本图二维图文质心距离最小的面板是 lora_r8/noise（16.78），最大的为 lit/noise（16.79）；这些数值仅描述本投影。

每个面板显示同一批 1000 张图像及其首条文本，灰线连接固定抽取的 30 对图文。

二维距离受降维影响，不能替代原始嵌入的模态间隙，也不能跨图或跨模型比较大小。

文件：`results/p9/figures/siglip_umap_settings_noise_flickr.png`；来源：`results/p9/coordinates/siglip_umap_settings_noise_flickr.csv`。

## F159 — siglip_umap_settings_occl_coco

图注：SIGLIP 在 coco val 的 settings_occl 联合 UMAP 投影。

本图二维图文质心距离最小的面板是 lit/occl（16.57），最大的为 pretrained/occl（16.58）；这些数值仅描述本投影。

每个面板显示同一批 1000 张图像及其首条文本，灰线连接固定抽取的 30 对图文。

二维距离受降维影响，不能替代原始嵌入的模态间隙，也不能跨图或跨模型比较大小。

文件：`results/p9/figures/siglip_umap_settings_occl_coco.png`；来源：`results/p9/coordinates/siglip_umap_settings_occl_coco.csv`。

## F160 — siglip_umap_settings_occl_coco_classes

图注：SIGLIP 在 coco val 的 settings_occl 联合 UMAP 投影。

采样图像中最多的三个类别（COCO 类别 ID）及数量为：0: 333、60: 48、6: 34。

同一类别的图像与首条文本使用相同颜色；只给数量最多的十类着色，其余点为灰色。

类别标签随图像 ID 固定，所有面板使用同一批样本。

文件：`results/p9/figures/siglip_umap_settings_occl_coco_classes.png`；来源：`results/p9/coordinates/siglip_umap_settings_occl_coco.csv`。

## F161 — siglip_umap_settings_occl_flickr

图注：SIGLIP 在 flickr val 的 settings_occl 联合 UMAP 投影。

本图二维图文质心距离最小的面板是 lit/occl（17.36），最大的为 lora_r8/occl（17.36）；这些数值仅描述本投影。

每个面板显示同一批 1000 张图像及其首条文本，灰线连接固定抽取的 30 对图文。

二维距离受降维影响，不能替代原始嵌入的模态间隙，也不能跨图或跨模型比较大小。

文件：`results/p9/figures/siglip_umap_settings_occl_flickr.png`；来源：`results/p9/coordinates/siglip_umap_settings_occl_flickr.csv`。
