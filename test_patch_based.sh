#!/bin/bash



# 运行测试脚本
/home/jinfang/anaconda3/envs/my_project/bin/python test_patch_base.py \
    --model-path "/home/jinfang/project/new_CarotidPlaqueStabilityClassification/weights/patch_best.pth" \
    --root-dir /media/data/wjf/data/Carotid_artery \
    --mask-dir /media/data/wjf/data/mask \
    --label-excel /media/data/wjf/data/label_all_250+30+100.xlsx \
    --output-dir ./output/test_patch_based \
    --patch-size 24 \
    --max-patches-per-roi 12 \
    --overlap-ratio 0.5 \
    --feature-dim 64 \
    --batch-size 8 \
    --depth 100 \
    --train-ratio 0.8 \
    --val-ratio 0.1 \
    --seed 42  \
    --num-workers 4

echo ""
echo "测试完成！"
