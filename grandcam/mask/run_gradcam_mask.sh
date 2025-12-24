#!/bin/bash
# GradCAM可视化运行脚本 - Mask引导模型版本

# 配置
MODEL_PATH="/home/jinfang/project/new_CarotidPlaqueStabilityClassification/output_mask/mask_guide_100_train_mask_20251223_210453/models/best_model.pth"
OUTPUT_DIR="./gradcam_visualizations_mask"
TARGET_LAYER="layer4"  # 可选: layer1, layer2, layer3, layer4
SEED=42  # 与训练时保持一致
SPLIT="test"  # 可选: train, val, test, all

# Mask目录
MASK_DIR="/media/data/wjf/data/mask"

# 方式1: 可视化测试集的所有患者
# 注意：每个患者的100张切片都会保存为独立文件（包含原图、mask、热力图、叠加图）
python visualize_with_gradcam_mask.py \
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
    --max-samples 2  # 限制前N个样本,设为-1则全部可视化

# 方式2: 只可视化特定患者名称
# python visualize_with_gradcam_mask.py \
#     --model-path ${MODEL_PATH} \
#     --mask-dir ${MASK_DIR} \
#     --output-dir ${OUTPUT_DIR}/specific_patients \
#     --target-layer ${TARGET_LAYER} \
#     --patient-ids "何亚玲" \
#     --alpha 0.5 \
#     --depth 100 \
#     --seed ${SEED} \
#     --split test

# 方式3: 可视化验证集
# python visualize_with_gradcam_mask.py \
#     --model-path ${MODEL_PATH} \
#     --mask-dir ${MASK_DIR} \
#     --output-dir ${OUTPUT_DIR}_val \
#     --target-layer ${TARGET_LAYER} \
#     --split val \
#     --seed ${SEED} \
#     --max-samples 10

# 方式4: 可视化训练集
# python visualize_with_gradcam_mask.py \
#     --model-path ${MODEL_PATH} \
#     --mask-dir ${MASK_DIR} \
#     --output-dir ${OUTPUT_DIR}_train \
#     --target-layer ${TARGET_LAYER} \
#     --split train \
#     --seed ${SEED} \
#     --max-samples 10

echo "✓ GradCAM可视化完成！"
echo "✓ 结果保存在: ${OUTPUT_DIR}"
