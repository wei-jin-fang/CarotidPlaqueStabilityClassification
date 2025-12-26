#!/bin/bash
# GradCAM可视化运行脚本 - 自适应裁剪版本
# 可视化基于裁剪后的244x244图像

# 配置
MODEL_PATH="/home/jinfang/project/new_CarotidPlaqueStabilityClassification/output_adaptive_mask/train_adaptive_20251225_191020/models/best_model.pth"
OUTPUT_DIR="./gradcam_visualizations_adaptive_crop"
TARGET_LAYER="layer4"  # 可选: layer1, layer2, layer3, layer4
SEED=42  # 与训练时保持一致
SPLIT="test"  # 可选: train, val, test, all

# Mask目录
MASK_DIR="/media/data/wjf/data/mask"

# 自适应裁剪参数（需要与训练时保持一致）
CROP_PADDING_RATIO=0.1  # 裁剪时bbox的padding比例
CROP_STRATEGY="adaptive"  # adaptive | pad_only | resize_only

# 方式1: 可视化测试集的所有患者
# 注意：每个患者的100张切片都会保存为独立文件
# 所有可视化都基于裁剪后的244x244图像（包括原图、mask、热力图、叠加图）
python visualize_with_gradcam_adaptive_crop.py \
    --model-path ${MODEL_PATH} \
    --mask-dir ${MASK_DIR} \
    --output-dir ${OUTPUT_DIR} \
    --target-layer ${TARGET_LAYER} \
    --alpha 0.4 \
    --depth 100 \
    --seed ${SEED} \
    --split ${SPLIT} \
    --train-ratio 0.8 \
    --val-ratio 0.1 \
    --crop-padding-ratio ${CROP_PADDING_RATIO} \
    --crop-strategy ${CROP_STRATEGY} \
    --max-samples 2  # 限制前N个样本,设为-1则全部可视化

# 方式2: 只可视化特定患者名称
# python visualize_with_gradcam_adaptive_crop.py \
#     --model-path ${MODEL_PATH} \
#     --mask-dir ${MASK_DIR} \
#     --output-dir ${OUTPUT_DIR}/specific_patients \
#     --target-layer ${TARGET_LAYER} \
#     --patient-ids "患者姓名1,患者姓名2" \
#     --alpha 0.5 \
#     --depth 100 \
#     --seed ${SEED} \
#     --split test \
#     --crop-padding-ratio ${CROP_PADDING_RATIO} \
#     --crop-strategy ${CROP_STRATEGY}

# 方式3: 可视化验证集
# python visualize_with_gradcam_adaptive_crop.py \
#     --model-path ${MODEL_PATH} \
#     --mask-dir ${MASK_DIR} \
#     --output-dir ${OUTPUT_DIR}_val \
#     --target-layer ${TARGET_LAYER} \
#     --split val \
#     --seed ${SEED} \
#     --crop-padding-ratio ${CROP_PADDING_RATIO} \
#     --crop-strategy ${CROP_STRATEGY} \
#     --max-samples 10

# 方式4: 可视化训练集
# python visualize_with_gradcam_adaptive_crop.py \
#     --model-path ${MODEL_PATH} \
#     --mask-dir ${MASK_DIR} \
#     --output-dir ${OUTPUT_DIR}_train \
#     --target-layer ${TARGET_LAYER} \
#     --split train \
#     --seed ${SEED} \
#     --crop-padding-ratio ${CROP_PADDING_RATIO} \
#     --crop-strategy ${CROP_STRATEGY} \
#     --max-samples 10

echo ""
echo "✓ GradCAM可视化完成！"
echo "✓ 结果保存在: ${OUTPUT_DIR}"
echo "✓ 所有可视化基于裁剪后的244x244图像"
echo "✓ 裁剪策略: ${CROP_STRATEGY} (padding_ratio=${CROP_PADDING_RATIO})"
