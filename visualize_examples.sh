#!/bin/bash

# Patch Attention 可视化脚本示例
# 根据你的需求选择使用

RESULTS_FILE="./output_patch_based/train_patch_20251226_193043/results/test_predictions_with_attention.pkl"

echo "========================================"
echo "Patch Attention 可视化工具"
echo "========================================"

# 示例1：只可视化最重要的slice（快速浏览）
echo -e "\n1. 快速浏览模式（只显示最重要的slice）"
# python visualize_patch_attention.py \
#     --results-file $RESULTS_FILE \
#     --output-dir ./vis_best_slice \
#     --mode best_slice

# 示例2：可视化所有slice（深度分析，先测试5个患者）
echo -e "\n2. 深度分析模式（测试5个患者的所有slice）"
python visualize_patch_attention.py \
    --results-file $RESULTS_FILE \
    --output-dir ./vis_all_slices_top5 \
    --mode all_slices \
    --max-samples 5

# 示例3：只分析错误预测的患者
# echo -e "\n3. 错误案例分析模式"
# python visualize_patch_attention.py \
#     --results-file $RESULTS_FILE \
#     --output-dir ./vis_errors_all_slices \
#     --mode all_slices \
#     --only-errors

# 示例4：统计分析 + 可视化
echo -e "\n4. 完整分析模式（统计 + 可视化）"
# python visualize_patch_attention.py \
#     --results-file $RESULTS_FILE \
#     --output-dir ./vis_full_analysis \
#     --mode all_slices \
#     --max-samples 10 \
#     --analyze-stats

echo -e "\n========================================"
echo "可视化完成！"
echo "========================================"
echo "输出目录："
echo "  - ./vis_best_slice/          (最重要slice)"
echo "  - ./vis_all_slices_top5/     (前5个患者的所有slice)"
echo "  - ./vis_errors_all_slices/   (错误预测的所有slice)"
echo "  - ./vis_full_analysis/       (完整分析)"
echo "========================================"
