#!/bin/bash

# Patch-Feature融合颈动脉斑块分类训练脚本 (CLS Token版本)
# 使用CLS Token + Transformer聚合patch特征

echo "Starting Patch-Feature Fusion (CLS Token) training..."

# 定义要循环的参数列表
batch_sizes=(4)
learning_rates=(1e-4)
cls_num_heads=(4)
cls_num_layers=(2)

# 总共组合数
total=$(( ${#batch_sizes[@]} * ${#learning_rates[@]} * ${#cls_num_heads[@]} * ${#cls_num_layers[@]} ))
count=0

for bs in "${batch_sizes[@]}"; do
    for lr in "${learning_rates[@]}"; do
        for heads in "${cls_num_heads[@]}"; do
            for layers in "${cls_num_layers[@]}"; do
                count=$((count + 1))
                echo "=================================================================="
                echo "[$count/$total] Training with: batch_size=$bs  lr=$lr  cls_heads=$heads  cls_layers=$layers"
                echo "=================================================================="

                /home/jinfang/anaconda3/envs/my_project/bin/python train_patch_feature_cls.py \
                    --patch-size 24 \
                    --max-patches-per-roi 12 \
                    --overlap-ratio 0.5 \
                    --image-feature-dim 128 \
                    --excel-hidden-dim 64 \
                    --fusion-hidden-dim 256 \
                    --vit-depth 1 \
                    --vit-heads 1 \
                    --vit-sub-patch-size 4 \
                    --cls-num-heads $heads \
                    --cls-num-layers $layers \
                    --max-seq-len 3000 \
                    --epochs 50 \
                    --batch-size $bs \
                    --lr $lr \
                    --weight-decay 1e-4 \
                    --depth 100 \
                    --train-ratio 0.8 \
                    --val-ratio 0.1 \
                    --seed 42 \
                    --num-workers 4

                echo "Finished [$count/$total]: bs=$bs, lr=$lr, heads=$heads, layers=$layers"
                echo ""
            done
        done
    done
done

echo "All Patch-Feature Fusion (CLS Token) training completed!"
