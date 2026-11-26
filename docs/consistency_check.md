# 图表一致性抽查

种子 42；抽查 10 个数字、10 张图；结果：通过。

数字回查原始 CSV/XLSX 与报告 CSV 的格式化值；图回查输入/输出 SHA-256、坐标文件、首条 caption、300 dpi 和 SVG。这不是对全部上游实验的科学结论验收。

- PASS retrieval_retrieval_val row 13 t2i_R5 ← `results/retrieval_val.csv data row 51`
- PASS drift row 287 n ← `results/drift_cosine.csv data row 192`
- PASS retrieval_retrieval_val row 16 i2t_R1 ← `results/retrieval_val.csv data row 71`
- PASS drift row 203 p95 ← `results/drift_cosine.csv data row 160`
- PASS drift row 51 std ← `results/drift_cosine.csv data row 278`
- PASS drift row 47 std ← `results/drift_cosine.csv data row 274`
- PASS drift row 234 p50 ← `results/drift_cosine.csv data row 447`
- PASS retrieval_retrieval_val row 19 t2i_R5 ← `results/retrieval_val.csv data row 61`
- PASS alignment_uniformity row 14 unif_txt ← `results/geometry_metrics.csv data row 59`
- PASS drift row 348 std ← `results/drift_cosine.csv data row 26`
- PASS results/p9/figures/siglip_tsne_settings_occl_flickr.png ← `results/p9/coordinates/siglip_tsne_settings_occl_flickr.csv`
- PASS results/p9/figures/clip_umap_conditions_pretrained_flickr.png ← `results/p9/coordinates/clip_umap_conditions_pretrained_flickr.csv`
- PASS results/p9/figures/clip_umap_conditions_lora_r8_coco_classes.png ← `results/p9/coordinates/clip_umap_conditions_lora_r8_coco.csv`
- PASS results/p9/figures/siglip_tsne_conditions_pretrained_coco.png ← `results/p9/coordinates/siglip_tsne_conditions_pretrained_coco.csv`
- PASS results/p9/figures/siglip_umap_conditions_lora_r16_coco_classes.png ← `results/p9/coordinates/siglip_umap_conditions_lora_r16_coco.csv`
- PASS results/p9/figures/clip_tsne_settings_clean_flickr.png ← `results/p9/coordinates/clip_tsne_settings_clean_flickr.csv`
- PASS results/p9/figures/clip_umap_settings_clean_coco.png ← `results/p9/coordinates/clip_umap_settings_clean_coco.csv`
- PASS results/p9/figures/clip_umap_settings_noise_flickr.png ← `results/p9/coordinates/clip_umap_settings_noise_flickr.csv`
- PASS results/p9/figures/siglip_tsne_conditions_lit_flickr.png ← `results/p9/coordinates/siglip_tsne_conditions_lit_flickr.csv`
- PASS results/p9/figures/siglip_umap_conditions_pretrained_coco_classes.png ← `results/p9/coordinates/siglip_umap_conditions_pretrained_coco.csv`

仍缺上游表格：svd, strength。后续到齐后重新运行本脚本。
