#!/bin/bash

# 自适应裁剪版本训练脚本
# 使用基于Mask的智能ROI裁剪替代传统resize

python train_adaptive_mask.py \
    --root-dir /media/data/wjf/data/Carotid_artery \
    --mask-dir /media/data/wjf/data/mask \
    --label-excel /media/data/wjf/data/label_all_250+30+100.xlsx \
    --pretrained-path ./weights/resnet_18_23dataset.pth \
    --output-dir ./output_adaptive_mask \
    --freeze-backbone \
    --epochs 50 \
    --batch-size 8 \
    --lr 1e-3 \
    --crop-padding-ratio 0 \
    --crop-strategy adaptive \


