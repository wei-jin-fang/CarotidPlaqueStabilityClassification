#!/bin/bash

# 基于模型的Patch Attention可视化示例脚本
# 可以直接加载模型，对任意数据集、任意患者进行可视化

# ============================================================
# 配置参数（根据你的实际情况修改）
# ============================================================

# 模型路径（修改为你训练好的模型）
MODEL_PATH="/home/jinfang/project/new_CarotidPlaqueStabilityClassification/output_patch_based/train_patch_20251227_100218/models/best_model.pth"

# 输出目录
OUTPUT_BASE="./visualizations_model_based"

echo "============================================================"
echo "基于模型的Patch Attention可视化"
echo "============================================================"
echo "模型路径: $MODEL_PATH"
echo ""

# ============================================================
# 示例1: 可视化测试集的所有患者（最佳切片模式）
# ============================================================
# echo "示例1: 测试集 - 所有患者 - 最佳切片"
# echo "------------------------------------------------------------"
# python visualize_patch_attention_across_model.py \
#     --model-path "$MODEL_PATH" \
#     --split test \
#     --mode best_slice \
#     --output-dir "$OUTPUT_BASE"

# ============================================================
# 示例2: 可视化验证集的前5个患者（所有切片模式）
# ============================================================
echo ""
echo "示例2: 验证集 - 前5个患者 - 所有切片"
echo "------------------------------------------------------------"
python visualize_patch_attention_across_model.py \
    --model-path "$MODEL_PATH" \
    --split val \
    --mode all_slices \
    --max-samples 5 \
    --output-dir "$OUTPUT_BASE"

# ============================================================
# 示例3: 可视化指定患者（所有切片）
# ============================================================
# echo ""
# echo "示例3: 指定患者 - 所有切片"
# echo "------------------------------------------------------------"
# python visualize_patch_attention_across_model.py \
#     --model-path "$MODEL_PATH" \
#     --split test \
#     --patient-name "A0001" \
#     --mode all_slices \
#     --output-dir "$OUTPUT_BASE"

# ============================================================
# 示例4: 只可视化错误预测的患者（测试集）
# ============================================================
# echo ""
# echo "示例4: 测试集 - 只看错误预测 - 所有切片"
# echo "------------------------------------------------------------"
# python visualize_patch_attention_across_model.py \
#     --model-path "$MODEL_PATH" \
#     --split test \
#     --mode all_slices \
#     --only-errors \
#     --output-dir "$OUTPUT_BASE"

# ============================================================
# 示例5: 可视化训练集的前10个患者（最佳切片）
# ============================================================
# echo ""
# echo "示例5: 训练集 - 前10个患者 - 最佳切片"
# echo "------------------------------------------------------------"
# python visualize_patch_attention_across_model.py \
#     --model-path "$MODEL_PATH" \
#     --split train \
#     --mode best_slice \
#     --max-samples 10 \
#     --output-dir "$OUTPUT_BASE"

# ============================================================
# 示例6: 使用不同的颜色映射
# ============================================================
# echo ""
# echo "示例6: 使用热力图颜色 (hot colormap)"
# echo "------------------------------------------------------------"
# python visualize_patch_attention_across_model.py \
#     --model-path "$MODEL_PATH" \
#     --split test \
#     --mode best_slice \
#     --max-samples 5 \
#     --colormap hot \
#     --output-dir "$OUTPUT_BASE"

echo ""
echo "============================================================"
echo "✓ 可视化完成！"
echo "============================================================"
echo "输出目录: $OUTPUT_BASE"
echo ""
echo "目录结构:"
echo "  $OUTPUT_BASE/"
echo "  ├── test_best_slice/          # 测试集-最佳切片"
echo "  │   ├── 001_patient_A_slice042_correct.png"
echo "  │   └── 002_patient_B_slice035_wrong.png"
echo "  └── val_all_slices/           # 验证集-所有切片"
echo "      ├── 001_patient_C_correct/"
echo "      │   ├── _summary.txt"
echo "      │   ├── slice000_avg0.0123.png"
echo "      │   ├── slice042_BEST_avg0.0850.png"
echo "      │   └── slice099_avg0.0098.png"
echo "      └── 002_patient_D_wrong/"
echo "          └── ..."
echo "============================================================"
