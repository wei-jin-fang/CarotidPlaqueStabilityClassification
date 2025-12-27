# Patch Attention 可视化指南

## 功能说明

可视化脚本现在支持两种模式：

### 模式1：best_slice（默认）
只可视化每个患者**最重要的slice**（attention权重最高的patch所在的slice）

**输出结构**：
```
output_dir/
├── 001_patient_A_slice42.png
├── 002_patient_B_slice35.png
├── 003_patient_C_slice58.png
└── ...
```

### 模式2：all_slices（新增）
为每个患者创建文件夹，可视化该患者的**所有slice**

**输出结构**：
```
output_dir/
├── 001_patient_A/
│   ├── _summary.txt                    # 患者统计信息
│   ├── slice000_avg0.0123.png         # 第0个slice，平均attention=0.0123
│   ├── slice001_avg0.0156.png
│   ├── slice042_BEST_avg0.0850.png    # 最重要的slice（标注BEST）
│   ├── ...
│   └── slice099_avg0.0098.png
├── 002_patient_B/
│   ├── _summary.txt
│   ├── slice000_avg0.0145.png
│   └── ...
└── ...
```

## 使用方法

### 1. 只可视化最重要的slice（原有功能）

```bash
python visualize_patch_attention.py \
    --results-file ./output_patch_based/train_patch_XXX/results/test_predictions_with_attention.pkl \
    --output-dir ./vis_best_slice \
    --mode best_slice
```

**效果**：
- 每个患者1张图
- 快速查看模型最关注的区域

### 2. 可视化所有slice（新功能）

```bash
python visualize_patch_attention.py \
    --results-file ./output_patch_based/train_patch_XXX/results/test_predictions_with_attention.pkl \
    --output-dir ./vis_all_slices \
    --mode all_slices
```

**效果**：
- 每个患者一个文件夹
- 包含该患者的所有slice可视化
- 文件名包含平均attention（方便排序）
- 最重要的slice标注"BEST"

### 3. 只可视化错误预测的所有slice

```bash
python visualize_patch_attention.py \
    --results-file ./output_patch_based/train_patch_XXX/results/test_predictions_with_attention.pkl \
    --output-dir ./vis_errors_all_slices \
    --mode all_slices \
    --only-errors
```

**效果**：
- 只处理预测错误的患者
- 用于分析模型失败案例

### 4. 限制患者数量（测试用）

```bash
python visualize_patch_attention.py \
    --results-file ./output_patch_based/train_patch_XXX/results/test_predictions_with_attention.pkl \
    --output-dir ./vis_top5_all_slices \
    --mode all_slices \
    --max-samples 5
```

**效果**：
- 只处理前5个患者
- 快速预览效果

### 5. 完整参数示例

```bash
python visualize_patch_attention.py \
    --results-file ./output_patch_based/train_patch_XXX/results/test_predictions_with_attention.pkl \
    --root-dir /media/data/wjf/data/Carotid_artery \
    --mask-dir /media/data/wjf/data/mask \
    --output-dir ./vis_all_slices_full \
    --mode all_slices \
    --max-samples 10 \
    --analyze-stats
```

## 参数说明

| 参数 | 必需 | 默认值 | 说明 |
|------|------|--------|------|
| --results-file | ✅ | - | 预测结果pkl文件路径 |
| --root-dir | ❌ | /media/data/wjf/data/Carotid_artery | 数据根目录 |
| --mask-dir | ❌ | /media/data/wjf/data/mask | Mask目录 |
| --output-dir | ❌ | ./visualizations_patch_attention | 输出目录 |
| --mode | ❌ | best_slice | 可视化模式：best_slice 或 all_slices |
| --max-samples | ❌ | None | 最多可视化的患者数 |
| --only-errors | ❌ | False | 只可视化错误预测 |
| --analyze-stats | ❌ | False | 分析attention统计信息 |

## 输出文件说明

### 图片文件命名规则（all_slices模式）

```
slice042_BEST_avg0.0850.png
  │     │      │
  │     │      └── 该slice的平均attention权重
  │     └── 标注（仅最重要的slice有）
  └── slice索引（3位数，补0）
```

### _summary.txt 内容

```
患者: patient_A
预测: 1
真实标签: 1
置信度: 0.8542
是否正确: ✓

总切片数: 100
总patch数: 2400
最重要切片: slice042
最高attention权重: 0.0850

各切片平均attention:
  # 1. slice042: 0.0532 ← BEST
  # 2. slice041: 0.0487
  # 3. slice043: 0.0465
  # 4. slice040: 0.0431
  # 5. slice044: 0.0398
  ...
```

## 可视化图片内容

每张图包含4个子图：

```
┌─────────────┬─────────────┬─────────────┬─────────────┐
│  原图       │  Mask       │  热力图     │  叠加图     │
│  (灰度)     │  (白色ROI)  │  (JET色图)  │  (带方框)   │
└─────────────┴─────────────┴─────────────┴─────────────┘
```

**颜色和标注**：
- 🔴 红色热力图 = 高attention（模型认为重要）
- 🔵 蓝色热力图 = 低attention（模型不关注）
- 🟢 绿色粗框 = Top-1最重要的patch
- 🟡 黄色细框 = Top 2-3重要的patch
- 文本标注 = patch的attention权重值

## 使用场景

### 场景1：快速浏览（best_slice模式）
```bash
# 适合：快速查看测试集结果
python visualize_patch_attention.py \
    --results-file results.pkl \
    --mode best_slice
```

### 场景2：深度分析（all_slices模式）
```bash
# 适合：详细分析模型行为
python visualize_patch_attention.py \
    --results-file results.pkl \
    --mode all_slices \
    --max-samples 5
```

### 场景3：错误案例分析
```bash
# 适合：找出模型失败的原因
python visualize_patch_attention.py \
    --results-file results.pkl \
    --mode all_slices \
    --only-errors
```

### 场景4：论文/报告用图
```bash
# 适合：生成高质量可视化图
# 1. 先找到感兴趣的患者（best_slice模式）
python visualize_patch_attention.py \
    --results-file results.pkl \
    --mode best_slice

# 2. 为特定患者生成所有slice（用max-samples控制）
python visualize_patch_attention.py \
    --results-file results.pkl \
    --mode all_slices \
    --max-samples 3
```

## 性能提示

### all_slices模式的计算量

假设：
- 100个测试患者
- 每个患者100个slice
- 总共需要生成：100 × 100 = 10,000张图片

**建议**：
1. 先用`--max-samples 5`测试
2. 确认输出正确后，再处理全部患者
3. 使用`--only-errors`只分析错误案例（通常数量较少）

### 加速技巧

```bash
# 并行处理多个patients（需要手动分割pkl文件）
# 或者分批处理
python visualize_patch_attention.py --results-file results.pkl --mode all_slices --max-samples 10 &
python visualize_patch_attention.py --results-file results.pkl --mode all_slices --max-samples 20 --offset 10 &
```

## 常见问题

### Q1: 生成图片太慢？
A: 使用`--max-samples`限制数量，或者只用`best_slice`模式。

### Q2: 想只看某个特定患者的所有slice？
A: 可以先用`best_slice`模式找到该患者的索引，然后修改代码或手动过滤pkl文件。

### Q3: 如何批量查看最重要的slice？
A: 用`best_slice`模式，文件名包含slice索引，方便排序查看。

### Q4: summary.txt有什么用？
A: 快速了解患者信息，无需打开图片即可看到：
- 预测结果
- 最重要的slice是哪个
- 各slice的attention排名

## 技术细节

### all_slices模式的实现

1. **遍历所有slice**：从该患者的positions中提取所有slice_idx
2. **分组处理**：每个slice单独提取对应的patch和attention
3. **计算平均attention**：用于文件命名和排序
4. **标注最重要slice**：在文件名中加"BEST"
5. **生成summary**：统计信息汇总

### 文件命名的好处

```
slice042_BEST_avg0.0850.png
slice041_avg0.0487.png
slice040_avg0.0431.png
```

- 按文件名排序 = 按slice索引排序
- 一眼看出哪个是最重要的（BEST标记）
- 快速定位高attention的slice

---

**创建时间**: 2025-12-26
**版本**: 2.0 (新增all_slices模式)
