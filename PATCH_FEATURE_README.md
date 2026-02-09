# Patch-Feature融合分类模型

## 概述

这是一个基于Patch的颈动脉斑块分类模型,融合了图像特征和Excel临床特征。

## 新增文件

所有代码都是新建的,不改动原有文件:

1. **数据加载器**: `utils/dataset_patch_feature.py`
   - 加载图像patches和Excel临床特征
   - 对Excel特征进行归一化(StandardScaler)
   - 只保留同时有图片、mask和Excel特征的样本

2. **模型**: `models/patch_feature_classifier.py`
   - ViTPatchEncoder: 提取图像patch特征
   - AttentionAggregator: 聚合patch特征
   - FeatureMLP: 提取Excel临床特征
   - FusionClassifier: 融合图像和临床特征,分类

3. **训练脚本**: `train_patch_feature.py`
   - 完整的训练、验证、测试流程
   - 保存StandardScaler用于测试
   - 支持attention可视化

4. **Shell脚本**: `train_patch_feature.sh`
   - 超参数搜索脚本
   - 支持批量训练

## 数据要求

### Excel文件格式
- 路径: `/home/jinfang/project/new_CarotidPlaqueStabilityClassification/Test/feature_complete.xlsx`
- 必须包含列: "姓名", "住院号", "斑块性质稳定=0不稳定=1"
- 其他74列作为特征(医学指标)
- 患者姓名必须与图片文件夹名称一致

### 筛选条件
只保留**同时满足**以下条件的样本:
1. 在标签Excel中有记录
2. 有对应的mask文件
3. 有对应的图片文件
4. **在临床特征Excel中有记录**

## 模型架构

```
输入:
├── 图像patches: [B, N_patches, 1, 24, 24]
└── Excel特征: [B, 74] (已归一化)

处理:
├── 图像分支:
│   ├── ViTPatchEncoder: [B*N, 1, 24, 24] -> [B*N, 128]
│   └── AttentionAggregator: [B, N, 128] -> [B, 128]
│
├── 特征分支:
│   └── FeatureMLP: [B, 74] -> [B, 128]
│
└── 融合分类:
    ├── Concat: [B, 128] + [B, 128] -> [B, 256]
    └── Classifier: [B, 256] -> [B, 2]

输出:
├── Logits: [B, 2]
└── Attention weights: [B, N_patches]
```

## 使用方法

### 1. 快速测试

```bash
# 测试单次训练
python train_patch_feature.py \
    --patch-size 24 \
    --max-patches-per-roi 12 \
    --image-feature-dim 128 \
    --excel-hidden-dim 128 \
    --epochs 50 \
    --batch-size 8 \
    --lr 1e-3
```

### 2. 超参数搜索

```bash
# 修改 train_patch_feature.sh 中的参数列表
bash train_patch_feature.sh
```

### 3. 关键参数说明

**Patch参数**:
- `--patch-size`: patch大小 (默认24)
- `--max-patches-per-roi`: 每个ROI最多提取的patch数 (默认12)
- `--overlap-ratio`: patch重叠比例 (默认0.5)

**模型参数**:
- `--image-feature-dim`: 图像特征维度 (默认128)
- `--excel-hidden-dim`: Excel特征提取后的维度 (默认128)
- `--fusion-hidden-dim`: 融合后的隐藏层维度 (默认256)
- `--vit-depth`: ViT Transformer层数 (默认1)
- `--vit-heads`: ViT多头注意力头数 (默认1)

**训练参数**:
- `--epochs`: 训练轮数 (默认50)
- `--batch-size`: 批大小 (默认4)
- `--lr`: 学习率 (默认1e-3)
- `--weight-decay`: 权重衰减 (默认1e-4)

## 输出文件

训练完成后,结果保存在 `output_patch_feature/train_patch_feature_YYYYMMDD_HHMMSS/`:

```
output_patch_feature/train_patch_feature_YYYYMMDD_HHMMSS/
├── models/
│   ├── best_model.pth              # 最佳模型权重
│   └── feature_scaler.pkl          # StandardScaler (用于测试集)
├── logs/
│   ├── config.json                 # 训练配置
│   ├── training_history.csv        # 训练历史
│   ├── training_loss_curves.png    # Loss曲线
│   ├── model_params_summary.txt    # 模型参数统计
│   ├── train_samples.csv           # 训练集样本
│   ├── val_samples.csv             # 验证集样本
│   └── test_samples.csv            # 测试集样本
├── results/
│   ├── test_results.json           # 测试集结果
│   ├── test_predictions_detailed.csv           # 详细预测结果
│   └── test_predictions_with_attention.pkl     # 含attention权重
└── visualizations/                 # 可视化结果(可选)
```

## Excel特征处理

### 归一化
- 使用 `StandardScaler` 对所有74列特征进行归一化
- 训练集fit后的scaler保存为 `feature_scaler.pkl`
- 验证集和测试集使用相同的scaler进行transform

### 特征维度
- 输入: 74维 (去掉"姓名"、"住院号"、"斑块性质稳定=0不稳定=1")
- MLP提取后: 128维 (可通过 `--excel-hidden-dim` 调整)

## 模型参数量

典型配置 (image_feat=128, excel_hidden=128):
- 总参数: ~366K
- 模块分布:
  - PatchEncoder: ~230K
  - Aggregator: ~8K
  - FeatureMLP: ~30K
  - FusionClassifier: ~98K

## 与原有代码的区别

| 特性 | 原有代码 | 新代码 |
|------|---------|--------|
| 输入 | 仅图像patches | 图像patches + Excel特征 |
| 数据集 | PatchBasedCarotidDataset | PatchFeatureCarotidDataset |
| 模型 | PatchBasedClassifier | PatchFeatureFusionClassifier |
| 筛选条件 | 有图片+mask | 有图片+mask+Excel特征 |
| 特征归一化 | 无 | StandardScaler |
| 输出目录 | output_patch_based | output_patch_feature |

## 注意事项

1. **数据一致性**: 确保Excel中的患者姓名与图片文件夹名称完全一致
2. **特征完整性**: 只保留同时有图片和Excel特征的样本
3. **归一化**: 训练集、验证集、测试集使用相同的StandardScaler
4. **GPU显存**: 默认使用GPU 1 (在train_patch_feature.py第12行修改)

## 测试代码

```bash
# 测试模型结构
python models/patch_feature_classifier.py

# 测试数据集加载
python utils/dataset_patch_feature.py
```

## 常见问题

**Q: 样本数量变少了?**
A: 新代码只保留同时有图片、mask和Excel特征的样本。检查Excel中是否有该患者的记录。

**Q: 如何查看哪些样本被筛选掉?**
A: 查看训练输出中的统计信息:
```
✓ 成功加载: X 个样本
✗ 跳过样本(图片数不足): Y 个
✗ 缺少Mask: Z 个
✗ 缺少Excel特征: W 个
```

**Q: 如何使用训练好的模型进行推理?**
A: 需要加载保存的scaler对新样本的Excel特征进行归一化:
```python
from utils.dataset_patch_feature import load_scaler
scaler = load_scaler('path/to/feature_scaler.pkl')
normalized_features = scaler.transform(raw_features.reshape(1, -1))
```

## 作者

Created on 2026-01-07
