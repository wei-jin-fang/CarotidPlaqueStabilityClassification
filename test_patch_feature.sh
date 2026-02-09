#!/bin/bash

# Patch-Feature融合颈动脉斑块分类测试脚本
# 使用训练好的模型对测试集进行评估

echo "Starting Patch-Feature Fusion testing..."

MODEL_PATH="/home/jinfang/project/new_CarotidPlaqueStabilityClassification/output/output_patch_feature/train_patch_feature_20260203_172142/models/best_model_48.pth"

/home/jinfang/anaconda3/envs/my_project/bin/python test_patch_feature.py \
    --model-path "$MODEL_PATH" \
    --patch-size 24 \
    --max-patches-per-roi 12 \
    --overlap-ratio 0.5 \
    --image-feature-dim 128 \
    --excel-hidden-dim 64 \
    --fusion-hidden-dim 256 \
    --vit-depth 1 \
    --vit-heads 1 \
    --vit-sub-patch-size 4 \
    --depth 100 \
    --train-ratio 0.8 \
    --val-ratio 0.1 \
    --seed 42 \
    --batch-size 4 \
    --num-workers 4 \
    --save-attention

echo "Patch-Feature Fusion testing completed!"
