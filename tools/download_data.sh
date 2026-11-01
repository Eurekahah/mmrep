#!/bin/sh
# 一键下载并检查原始数据，支持断点续传（中断后重新运行即可）。在仓库根目录运行：sh tools/download_data.sh [flickr|coco|cifar|all]
# 国内下载 HF 慢时先：export HF_ENDPOINT=https://hf-mirror.com
set -e
R="${MMREP_DATA_ROOT:-data_raw}"
HF="${HF_ENDPOINT:-https://huggingface.co}"
what="${1:-all}"
fetch() { # url out expected_size
  mkdir -p "$(dirname "$2")"
  for i in 1 2 3 4 5 6 7 8 9 10; do
    curl -fL -C - -o "$2" "$1" || true
    [ "$(wc -c < "$2" 2>/dev/null || echo 0)" = "$3" ] && return 0
    echo "retry $i for $2"; sleep 5
  done
  echo "FAILED: $2 size mismatch (expected $3)"; return 1
}
if [ "$what" = flickr ] || [ "$what" = all ]; then
  fetch "$HF/datasets/nlphuji/flickr30k/resolve/main/flickr_annotations_30k.csv" "$R/flickr30k/flickr_annotations_30k.csv" 12896305
  fetch "$HF/datasets/nlphuji/flickr30k/resolve/main/flickr30k-images.zip" "$R/flickr30k/flickr30k-images.zip" 4390240817
  [ -d "$R/flickr30k/flickr30k-images" ] || unzip -q "$R/flickr30k/flickr30k-images.zip" -d "$R/flickr30k"
  echo "flickr images: $(ls "$R/flickr30k/flickr30k-images" | wc -l)"
fi
if [ "$what" = coco ] || [ "$what" = all ]; then
  fetch http://images.cocodataset.org/zips/val2017.zip "$R/coco/val2017.zip" 815585330
  fetch http://images.cocodataset.org/annotations/annotations_trainval2017.zip "$R/coco/annotations_trainval2017.zip" 252907541
  [ -d "$R/coco/val2017" ] || unzip -q "$R/coco/val2017.zip" -d "$R/coco"
  [ -f "$R/coco/annotations/instances_val2017.json" ] || unzip -q -o "$R/coco/annotations_trainval2017.zip" 'annotations/instances_val2017.json' 'annotations/captions_val2017.json' -d "$R/coco"
  echo "coco images: $(ls "$R/coco/val2017" | wc -l)"
fi
if [ "$what" = cifar ] || [ "$what" = all ]; then
  python -m data.prepare_data cifar100 --root "$R/cifar100" --download --out /tmp/cifar100_classes_check.json
fi
echo "done. 校验划分：见 README“下完后检查图片是否齐全”"
