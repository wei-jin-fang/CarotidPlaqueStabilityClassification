#!/bin/bash

# Excel特征颈动脉斑块分类训练脚本
# 只使用Excel临床特征进行分类,不使用图像

echo "Starting Feature Classifier training..."

# ============================================================
# 模式选择:
# 1. 默认模式: 使用FeatureCarotidDataset(纯特征,不涉及图像)
# 2. --use-patch-dataset: 使用PatchFeatureCarotidDataset(保持与patch+feature训练数据划分一致)
# ============================================================

# 默认模式: 纯特征数据集
/home/jinfang/anaconda3/envs/my_project/bin/python train_feature.py \
    --excel-hidden-dim 64 \
    --fusion-hidden-dim 256 \
    --epochs 50 \
    --batch-size 4 \
    --lr 1e-3 \
    --weight-decay 1e-4 \
    --train-ratio 0.8 \
    --val-ratio 0.1 \
    --seed 42 \
    --num-workers 4

# ============================================================
# 使用PatchFeatureCarotidDataset模式(保持数据划分一致)
# 取消注释以下内容使用
# ============================================================
# /home/jinfang/anaconda3/envs/my_project/bin/python train_feature.py \
#     --use-patch-dataset \
#     --patch-size 24 \
#     --max-patches-per-roi 12 \
#     --overlap-ratio 0.5 \
#     --depth 100 \
#     --excel-hidden-dim 64 \
#     --fusion-hidden-dim 256 \
#     --epochs 50 \
#     --batch-size 4 \
#     --lr 1e-4 \
#     --weight-decay 1e-4 \
#     --train-ratio 0.8 \
#     --val-ratio 0.1 \
#     --seed 42 \
#     --num-workers 4

echo "Feature Classifier training completed!"
