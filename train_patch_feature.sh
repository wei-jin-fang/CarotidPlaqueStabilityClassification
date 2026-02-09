#!/bin/bash

# Patch-Feature融合颈动脉斑块分类训练脚本
# 融合图像patch特征和Excel临床特征进行分类
echo "Starting Patch-Feature Fusion training..."

# 基础训练命令示例
# python train_patch_feature.py \
#     --patch-size 24 \
#     --max-patches-per-roi 12 \
#     --overlap-ratio 0.5 \
#     --image-feature-dim 128 \
#     --excel-hidden-dim 128 \
#     --fusion-hidden-dim 256 \
#     --epochs 50 \
#     --batch-size 8 \
#     --lr 1e-3

echo "Starting Patch-Feature Fusion hyperparameter search..."

# 定义要循环的参数列表
batch_sizes=(4)
learning_rates=(1e-4)
image_feature_dims=(128)
excel_hidden_dims=(64)

# # 快速测试配置(取消注释以使用)
# batch_sizes=(8)
# learning_rates=(1e-4)
# image_feature_dims=(64)
# excel_hidden_dims=(64)

# 总共组合数
total=$(( ${#batch_sizes[@]} * ${#learning_rates[@]} * ${#image_feature_dims[@]} * ${#excel_hidden_dims[@]} ))
count=0

for bs in "${batch_sizes[@]}"; do
    for lr in "${learning_rates[@]}"; do
        for ifd in "${image_feature_dims[@]}"; do
            for ehd in "${excel_hidden_dims[@]}"; do
                count=$((count + 1))
                echo "=================================================================="
                echo "[$count/$total] Training with: batch_size=$bs  lr=$lr  image_feat=$ifd  excel_hidden=$ehd"
                echo "=================================================================="

                /home/jinfang/anaconda3/envs/my_project/bin/python train_patch_feature.py \
                    --patch-size 24 \
                    --max-patches-per-roi 12 \
                    --overlap-ratio 0.5 \
                    --image-feature-dim $ifd \
                    --excel-hidden-dim $ehd \
                    --fusion-hidden-dim 256 \
                    --vit-depth 1 \
                    --vit-heads 1 \
                    --vit-sub-patch-size 4 \
                    --epochs 50 \
                    --batch-size $bs \
                    --lr $lr \
                    --weight-decay 1e-4 \
                    --depth 100 \
                    --train-ratio 0.8 \
                    --val-ratio 0.1 \
                    --seed 42 \
                    --num-workers 4

                echo "Finished [$count/$total]: bs=$bs, lr=$lr, ifd=$ifd, ehd=$ehd"
                echo ""
            done
        done
    done
done

echo "All Patch-Feature Fusion hyperparameter searches completed!"
