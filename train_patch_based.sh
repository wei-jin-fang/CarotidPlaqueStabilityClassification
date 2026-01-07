#!/bin/bash

# Patch-based颈动脉斑块分类训练脚本
# 从ROI区域提取小patch，使用attention聚合特征
echo "Starting Patch-based training..."

# python train_patch_based.py \
#     --patch-size 24 \
#     --max-patches-per-roi 12 \
#     --overlap-ratio 0.5 \
#     --feature-dim 128 \
#     --epochs 50 \
#     --batch-size 8 \
#     --lr 1e-3 

# echo "Training completed!"


# python visualize_patch_attention.py \
#       --results-file ./output_patch_based/train_patch_20251226_193043/results/test_predictions_with_attention.pkl \
#       --output-dir ./visualizations_patch_attention

echo "Starting Patch-based hyperparameter search..."

# 定义要循环的参数列表
batch_sizes=(8 4)
learning_rates=(1e-3 1e-4 5e-4)
feature_dims=(256 64 128)

batch_sizes=(8)
learning_rates=(1e-3)
feature_dims=(64)
# 总共组合数：3 × 3 × 3 = 27 次训练
total=$(( ${#batch_sizes[@]} * ${#learning_rates[@]} * ${#feature_dims[@]} ))
count=0

for bs in "${batch_sizes[@]}"; do
    for lr in "${learning_rates[@]}"; do
        for fd in "${feature_dims[@]}"; do
            count=$((count + 1))
            echo "=================================================================="
            echo "[$count/$total] Training with: batch_size=$bs  lr=$lr  feature_dim=$fd"
            echo "=================================================================="

            /home/jinfang/anaconda3/envs/my_project/bin/python train_patch_based.py \
                --patch-size 24 \
                --max-patches-per-roi 12 \
                --overlap-ratio 0.5 \
                --feature-dim $fd \
                --epochs 50 \
                --batch-size $bs \
                --lr $lr

            echo "Finished [$count/$total]: bs=$bs, lr=$lr, fd=$fd"
            echo ""
        done
    done
done

echo "All hyperparameter searches completed!"