MODEL_PATH="/home/jinfang/project/new_CarotidPlaqueStabilityClassification/weights/patch_best.pth"
echo "============================================================"
echo "基于模型的Patch Attention可视化 - 指定患者和切片"
echo "============================================================"
echo "模型路径: $MODEL_PATH"
python visualize_patch_attention_across_model_person_slice.py \
      --model-path "$MODEL_PATH" \
      --patient-name "李必勇" \
      --slice-idx 43 \
      --split test \
      --feature-dim 64 \
      --patch-size 24 \
      --output-dir ./visualization/person_slice
# # 示例2: 可视化训练集中某患者的第30张切片，使用不同颜色映射
# python visualize_patch_attention_across_model_person_slice.py \
#     --model-path ./output_patch_based/train_patch_20251230_095749/models/best.pth \
#     --patient-name "patient_002" \
#     --slice-idx 30 \
#     --split train \
#     --colormap hot \
#     --output-dir ./my_visualizations