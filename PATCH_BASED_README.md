# Patch-based 颈动脉斑块分类系统

## 系统概述

本系统采用Patch-based方法解决小ROI区域的背景干扰问题：

1. **Patch提取**：从mask的白色区域内提取小patch（24×24），避免黑色背景
2. **Attention聚合**：通过attention机制学习哪些patch更重要
3. **位置记录**：保存每个patch的位置信息，支持可视化
4. **热力图可视化**：将patch的attention权重映射回原图

## 文件结构

```
新增文件：
├── utils/dataset_patch_based.py      # Patch数据集类
├── models/patch_classifier.py        # Patch分类器（含Attention）
├── train_patch_based.py              # 训练脚本
├── train_patch_based.sh              # 训练Shell脚本
└── visualize_patch_attention.py      # Attention可视化脚本
```

## 核心设计

### 1. Patch提取策略

```python
# 从两个ROI区域分别提取patch
- 识别mask中的独立区域（左右斑块）
- 每个ROI内滑窗采样（50%重叠）
- 按mask覆盖率过滤（>=30%是前景）
- 记录每个patch的位置信息
```

**示例**：
- Patch大小：24×24像素
- 每个ROI提取：12个patch
- 假设2个ROI，100个slice → 约2400个patch/样本

### 2. 模型架构

```
输入: [B, N_patches, 1, 24, 24]
  ↓
PatchEncoder (共享权重的2D CNN)
  ↓
Features: [B, N_patches, 128]
  ↓
AttentionAggregator
  ↓
Aggregated: [B, 128]  +  Attention Weights: [B, N_patches]
  ↓
Classifier
  ↓
输出: Logits [B, 2]
```

**关键**：Attention权重用于可视化patch重要性

### 3. 位置记录机制

每个patch记录：
```python
{
    'center_x': 60,        # patch中心x坐标
    'center_y': 110,       # patch中心y坐标
    'bbox': (48,98,72,122), # patch的矩形框
    'roi_id': 0,           # 来自哪个ROI（0=左，1=右）
    'slice_idx': 25,       # 来自第几个slice
    'mask_ratio': 0.85     # mask覆盖率
}
```

## 使用方法

### 步骤1：训练模型

```bash
# 直接运行训练脚本
bash train_patch_based.sh

# 或手动调整参数
python train_patch_based.py \
    --patch-size 24 \
    --max-patches-per-roi 12 \
    --overlap-ratio 0.5 \
    --epochs 50 \
    --batch-size 4 \
    --lr 1e-3
```

**训练输出**：
```
output_patch_based/train_patch_YYYYMMDD_HHMMSS/
├── models/
│   └── best_model.pth                        # 最佳模型
├── logs/
│   ├── config.json                           # 训练配置
│   ├── training_history.csv                  # 训练历史
│   ├── training_curves.png                   # 训练曲线
│   ├── train_samples.csv                     # 训练集样本
│   ├── val_samples.csv                       # 验证集样本
│   └── test_samples.csv                      # 测试集样本
└── results/
    ├── test_results.json                     # 测试集指标
    ├── test_predictions_detailed.csv         # 预测结果CSV
    └── test_predictions_with_attention.pkl   # 详细结果（含attention）
```

### 步骤2：可视化Patch Attention

```bash
# 可视化所有测试样本
python visualize_patch_attention.py \
    --results-file output_patch_based/train_patch_YYYYMMDD_HHMMSS/results/test_predictions_with_attention.pkl \
    --output-dir ./visualizations_patch_attention

# 只可视化错误预测
python visualize_patch_attention.py \
    --results-file output_patch_based/.../test_predictions_with_attention.pkl \
    --output-dir ./visualizations_errors \
    --only-errors

# 限制数量 + 统计分析
python visualize_patch_attention.py \
    --results-file output_patch_based/.../test_predictions_with_attention.pkl \
    --output-dir ./visualizations_top20 \
    --max-samples 20 \
    --analyze-stats
```

**可视化输出**：
- 每个样本生成4张图：原图、Mask、热力图、叠加图
- 标注Top-5重要的patch（绿色框=最重要）
- 显示预测结果和置信度

### 步骤3：解读可视化结果

**热力图含义**：
- 🔴 红色区域：高attention权重，模型认为重要
- 🔵 蓝色区域：低attention权重，模型不关注
- 🟢 绿色框：最重要的patch（Top-1）
- 🟡 黄色框：次重要的patch（Top 2-5）

**分析示例**：
```
正确预测：
- 热力图集中在ROI中心区域
- Top-3 patch的attention权重较高（>0.15）
- 两个ROI的attention权重有明显差异

错误预测：
- 热力图分散，权重分布均匀
- Top-3 patch权重较低（<0.10）
- 可能关注了边缘或背景patch
```

## 参数调优指南

### Patch相关参数

```bash
--patch-size 24              # Patch大小
                             # 小ROI建议: 16-24
                             # 大ROI建议: 32-48

--max-patches-per-roi 12     # 每个ROI最多提取的patch数
                             # 影响：数量多→信息丰富但计算慢
                             # 建议：8-16

--overlap-ratio 0.5          # Patch重叠比例
                             # 0.5 = 50%重叠，密集采样
                             # 0.3 = 30%重叠，稀疏采样
```

### 模型参数

```bash
--feature-dim 128            # 特征维度
                             # 影响模型容量，建议64-256

--batch-size 4               # 批次大小
                             # patch多时显存占用大，酌情调整

--lr 1e-3                    # 学习率
                             # 从头训练建议1e-3
```

## 与原方法对比

| 特性 | 原方法（自适应裁剪） | Patch-based方法 |
|------|---------------------|-----------------|
| 背景处理 | 黑色padding | 完全避免背景 |
| 空间信息 | 保留完整结构 | 打散为patch |
| 可视化 | 支持GradCAM | Patch-level热力图 |
| 信息密度 | 低（有背景） | 高（100%前景） |
| 计算复杂度 | 低 | 中等 |
| 适用场景 | ROI较大 | ROI很小 |

## 常见问题

### Q1: Patch数量不一致怎么办？
A: 数据加载时会自动padding到相同长度，训练时使用mask标记有效patch。

### Q2: 两个ROI大小差异大？
A: Patch提取器会自动处理，小ROI提取少量patch，大ROI提取更多。

### Q3: 可视化结果看不清？
A: 可以调整`top_k`参数，只标注最重要的几个patch。

### Q4: Attention权重都很均匀？
A: 可能是模型没有学到判别性特征，尝试：
- 增加训练轮数
- 调整学习率
- 增加特征维度

## 技术细节

### Patch过滤策略

```python
# 边界patch处理
mask_ratio = (patch内白色像素) / (patch总像素)

if mask_ratio >= 0.3:  # 至少30%是ROI
    保留patch
else:
    丢弃（太多背景）
```

### Attention机制

```python
# Attention network
attention_logits = MLP(patch_features)  # [B, N, 1]
attention_weights = Softmax(attention_logits)  # [B, N]
aggregated_feature = Sum(attention_weights * patch_features)  # [B, D]
```

### 热力图生成

```python
# 将patch权重映射回原图
for each patch:
    heatmap[patch_bbox] += attention_weight

# 重叠区域取平均
heatmap = heatmap / count_map
```

## 后续改进方向

1. **多尺度patch**：同时使用16×16、24×24、32×32
2. **Patch选择策略**：训练时动态选择重要patch
3. **3D Attention**：沿深度方向建模patch关系
4. **对比学习**：学习同一ROI内patch的相似性

## 引用

如果本代码对您的研究有帮助，请引用我们的工作。

---

**创建时间**: 2025-12-26
**作者**: Claude Code Assistant
