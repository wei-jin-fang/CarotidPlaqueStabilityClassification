"""
新创建的基于模型的可视化工具总结

创建时间: 2025-12-27
"""

# ============================================================
# 新增文件
# ============================================================

1. visualize_patch_attention_across_model.py
   - 主可视化脚本（750+ 行）
   - 直接加载模型进行前向传播和可视化
   - 支持train/val/test三个数据集
   - 支持指定患者名字
   - 两种模式：best_slice / all_slices

2. visualize_model_examples.sh
   - 使用示例脚本
   - 包含6个典型使用场景
   - 可直接运行（需修改MODEL_PATH）

3. VISUALIZATION_MODEL_BASED_GUIDE.md
   - 完整的使用文档
   - 参数说明、使用场景、常见问题
   - 与训练脚本的参数对应表

# ============================================================
# 核心功能
# ============================================================

## 1. 直接加载模型
```python
model = load_model(model_path, patch_size=24, feature_dim=128)
```
- 无需预先运行测试并保存.pkl文件
- 实时计算attention权重
- 灵活切换不同模型

## 2. 多数据集支持
```bash
--split train    # 训练集
--split val      # 验证集
--split test     # 测试集
```
- 自动使用与训练脚本一致的划分逻辑
- 相同种子确保可重复性

## 3. 灵活的样本选择
```bash
# 所有样本
python visualize_patch_attention_across_model.py --model-path xxx

# 指定患者
python visualize_patch_attention_across_model.py --patient-name "A0001"

# 前N个
python visualize_patch_attention_across_model.py --max-samples 10

# 只看错误
python visualize_patch_attention_across_model.py --only-errors
```

## 4. 两种可视化模式

### best_slice 模式
- 每个患者一张图（最重要的切片）
- 快速浏览
- 输出：001_patient_A_slice042_correct.png

### all_slices 模式
- 每个患者一个文件夹
- 包含所有切片的可视化
- 自动标注BEST切片
- 生成_summary.txt统计文件
- 输出：
  ```
  001_patient_A_correct/
  ├── _summary.txt
  ├── slice000_avg0.0123.png
  ├── slice042_BEST_avg0.0850.png
  └── slice099_avg0.0098.png
  ```

## 5. 丰富的可视化内容

每张图包含4个子图：
1. Original Image - 原始图像
2. ROI Mask - 感兴趣区域
3. Attention Heatmap - 热力图
4. Top-K Patches - 标注最重要的patches

# ============================================================
# 使用场景
# ============================================================

场景1: 快速查看测试集
------------------------------------------
python visualize_patch_attention_across_model.py \
    --model-path best_model.pth \
    --split test \
    --mode best_slice

场景2: 深度分析错误预测
------------------------------------------
python visualize_patch_attention_across_model.py \
    --model-path best_model.pth \
    --split test \
    --mode all_slices \
    --only-errors

场景3: 验证集可视化（调优）
------------------------------------------
python visualize_patch_attention_across_model.py \
    --model-path best_model.pth \
    --split val \
    --max-samples 10 \
    --mode all_slices

场景4: 指定患者分析
------------------------------------------
python visualize_patch_attention_across_model.py \
    --model-path best_model.pth \
    --patient-name "A0001" \
    --mode all_slices

场景5: 训练集可视化（检查过拟合）
------------------------------------------
python visualize_patch_attention_across_model.py \
    --model-path best_model.pth \
    --split train \
    --max-samples 20

场景6: 不同颜色映射
------------------------------------------
python visualize_patch_attention_across_model.py \
    --model-path best_model.pth \
    --colormap hot \
    --max-samples 5

# ============================================================
# 与原可视化脚本对比
# ============================================================

visualize_patch_attention.py (原脚本):
✓ 从.pkl读取预先保存的结果
✓ 适合批量查看测试集
✗ 只支持测试集
✗ 不能指定患者
✗ 依赖预先运行evaluate_test_with_attention()

visualize_patch_attention_across_model.py (新脚本):
✓ 直接加载模型，实时计算
✓ 支持train/val/test三个数据集
✓ 可以指定患者名字
✓ 可以只看错误预测
✓ 更灵活，不依赖预先保存的结果
✓ 与训练脚本参数完全对齐

# ============================================================
# 参数对照表（必须与训练时一致）
# ============================================================

训练参数                     可视化参数
--------------------------------------------------
--patch-size 24        →    --patch-size 24
--feature-dim 128      →    --feature-dim 128
--max-patches-per-roi  →    --max-patches-per-roi 12
--overlap-ratio 0.5    →    --overlap-ratio 0.5
--depth 100            →    --depth 100
--train-ratio 0.8      →    --train-ratio 0.8
--val-ratio 0.1        →    --val-ratio 0.1
--seed 42              →    --seed 42

# ============================================================
# 快速开始
# ============================================================

步骤1: 修改示例脚本中的模型路径
--------------------------------------------------
vim visualize_model_examples.sh

# 修改这一行
MODEL_PATH="./output_patch_based/train_patch_YOUR_EXP/models/best_model.pth"

步骤2: 运行示例
--------------------------------------------------
bash visualize_model_examples.sh

步骤3: 查看结果
--------------------------------------------------
ls -lh visualizations_model_based/

# ============================================================
# 输出结果示例
# ============================================================

visualizations_model_based/
├── test_best_slice/                    # 测试集-最佳切片
│   ├── 001_patient_A_slice042_correct.png
│   ├── 002_patient_B_slice035_wrong.png
│   └── ...
├── val_all_slices/                     # 验证集-所有切片
│   ├── 001_patient_C_correct/
│   │   ├── _summary.txt
│   │   ├── slice000_avg0.0123.png
│   │   ├── slice042_BEST_avg0.0850.png
│   │   └── slice099_avg0.0098.png
│   └── 002_patient_D_wrong/
│       └── ...
└── ...

# ============================================================
# 关键技术实现
# ============================================================

1. 模型加载
--------------------------------------------------
def load_model(model_path, patch_size, feature_dim, device):
    model = create_patch_classifier(...)
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
    model.eval()
    return model

2. 数据集创建（与训练一致）
--------------------------------------------------
def create_dataset(split='test', ...):
    # 使用相同的随机种子和划分比例
    full_dataset = PatchBasedCarotidDataset(...)

    # 划分逻辑与train_patch_based.py完全一致
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size+val_size]
    test_indices = indices[train_size+val_size:]

    # 返回指定split
    dataset.persons = [full_dataset.persons[i] for i in selected_indices]
    return dataset

3. 实时计算attention
--------------------------------------------------
def visualize_patient_xxx(model, dataset, idx, ...):
    patches, positions, label = dataset[idx]
    patches_input = patches.unsqueeze(0).to(device)

    # 实时前向传播
    with torch.no_grad():
        logits, attn_weights = model(patches_input, return_attention=True)

    # 可视化
    visualize_single_slice(...)

4. 可视化单个切片
--------------------------------------------------
def visualize_single_slice(img, mask, patches_np, positions,
                          attention_weights, slice_idx, ...):
    # 筛选当前slice的patches
    slice_indices = [i for i, pos in enumerate(positions)
                    if pos['slice_idx'] == slice_idx]

    # 创建attention热力图
    heatmap = np.zeros((h, w))
    for pos, attn in zip(slice_positions, slice_attention):
        x1, y1, x2, y2 = pos['bbox']
        heatmap[y1:y2, x1:x2] += attn

    # 4子图可视化
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    # ... 原图、mask、热力图、top-k patches

5. 患者名字查找
--------------------------------------------------
def get_patient_index(dataset, patient_name):
    for idx, person in enumerate(dataset.persons):
        if person['name'] == patient_name:
            return idx
    return None

# ============================================================
# 使用建议
# ============================================================

1. 快速预览 → best_slice模式
2. 深度分析 → all_slices模式
3. 找bug → --only-errors
4. 论文图表 → 指定patient-name + all_slices
5. 检查过拟合 → --split train
6. 模型对比 → 批量运行多个模型

# ============================================================
# 注意事项
# ============================================================

⚠️ 重要：所有参数必须与训练时一致！

特别是：
- patch_size
- feature_dim
- max_patches_per_roi
- overlap_ratio
- depth

否则：
- 模型加载失败（feature_dim不匹配）
- 数据集不一致（其他参数不匹配）

✓ 推荐做法：
  从训练脚本复制参数，或使用默认值

# ============================================================
# 优势总结
# ============================================================

✅ 直接加载模型，无需预先保存结果
✅ 支持所有数据集（train/val/test）
✅ 灵活指定患者和样本
✅ 实时计算attention（适合调试）
✅ 完整的可视化内容（4子图）
✅ 自动标注最佳切片
✅ 生成统计摘要文件
✅ 与训练脚本参数完全对齐
✅ 丰富的使用场景和文档

# ============================================================
# 文件清单
# ============================================================

新增文件：
1. visualize_patch_attention_across_model.py  (主脚本)
2. visualize_model_examples.sh                (示例)
3. VISUALIZATION_MODEL_BASED_GUIDE.md         (文档)
4. MODEL_BASED_VISUALIZATION_SUMMARY.py       (本文件)

现有文件（无修改）：
- visualize_patch_attention.py        (原可视化脚本，保留)
- train_patch_based.py                (训练脚本)
- utils/dataset_patch_based.py        (数据集)
- models/patch_classifier.py          (模型)

# ============================================================
# 完成时间
# ============================================================

2025-12-27

所有功能已实现并测试！
