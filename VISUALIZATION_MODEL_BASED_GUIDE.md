# 基于模型的Patch Attention可视化工具

## 文件说明

**新创建的文件:**
- `visualize_patch_attention_across_model.py` - 主可视化脚本（加载模型进行可视化）
- `visualize_model_examples.sh` - 使用示例脚本

**与原有可视化脚本的区别:**

| 特性 | 原脚本 (visualize_patch_attention.py) | 新脚本 (visualize_patch_attention_across_model.py) |
|-----|--------------------------------------|--------------------------------------------------|
| 数据来源 | 读取保存的 `.pkl` 文件 | **直接加载模型进行前向传播** |
| 数据集支持 | 只有测试集 | **train/val/test 三个数据集** |
| 指定患者 | 不支持 | **支持按患者名字可视化** |
| 灵活性 | 低（依赖预先保存的结果） | **高（实时计算）** |
| 适用场景 | 批量查看测试集结果 | **任意数据集、任意患者的可视化** |

## 功能特性

### 1. 直接加载模型
- 无需预先运行测试集并保存结果
- 实时进行模型推理和可视化
- 支持任意训练好的模型权重

### 2. 多数据集支持
```bash
--split train    # 可视化训练集
--split val      # 可视化验证集
--split test     # 可视化测试集
```

### 3. 灵活的患者选择
```bash
# 可视化所有患者
python visualize_patch_attention_across_model.py --model-path xxx

# 可视化指定患者
python visualize_patch_attention_across_model.py --model-path xxx --patient-name "A0001"

# 可视化前N个患者
python visualize_patch_attention_across_model.py --model-path xxx --max-samples 10

# 只可视化错误预测
python visualize_patch_attention_across_model.py --model-path xxx --only-errors
```

### 4. 两种可视化模式

**best_slice 模式** - 快速浏览
- 每个患者只可视化最重要的切片（attention最高）
- 适合快速评估模型关注点
- 输出格式：`001_patient_A_slice042_correct.png`

**all_slices 模式** - 深度分析
- 每个患者可视化所有切片
- 创建患者文件夹，包含完整的切片序列
- 自动标注最佳切片（带BEST标记）
- 生成统计摘要 (`_summary.txt`)
- 输出格式：
  ```
  001_patient_A_correct/
  ├── _summary.txt
  ├── slice000_avg0.0123.png
  ├── slice042_BEST_avg0.0850.png  ← 最重要
  └── slice099_avg0.0098.png
  ```

### 5. 丰富的可视化内容

每张图包含4个子图：
1. **原始图像** - 灰度图
2. **ROI Mask** - 显示感兴趣区域
3. **Attention 热力图** - 显示模型关注的区域
4. **Top-K Patches** - 标注最重要的patches，显示排名和权重

### 6. 自动数据集划分
- 使用与训练脚本**完全一致**的划分逻辑
- 保证train/val/test划分一致性
- 相同的随机种子确保可重复性

## 使用方法

### 基础用法

```bash
python visualize_patch_attention_across_model.py \
    --model-path ./output_patch_based/train_patch_xxx/models/best_model.pth \
    --split test \
    --mode best_slice
```

### 完整参数说明

#### 必需参数
```bash
--model-path PATH          # 模型权重路径（必需）
```

#### 数据集参数（与训练脚本一致）
```bash
--root-dir PATH            # 图像根目录（默认: /media/data/wjf/data/Carotid_artery）
--mask-dir PATH            # Mask目录（默认: /media/data/wjf/data/mask）
--label-excel PATH         # 标签文件（默认: /media/data/wjf/data/label_all_250+30+100.xlsx）
--patch-size INT           # Patch大小（默认: 24）
--max-patches-per-roi INT  # 每个ROI最多patch数（默认: 12）
--overlap-ratio FLOAT      # Patch重叠比例（默认: 0.5）
--depth INT                # 切片深度（默认: 100）
--train-ratio FLOAT        # 训练集比例（默认: 0.8）
--val-ratio FLOAT          # 验证集比例（默认: 0.1）
```

#### 可视化参数
```bash
--split {train,val,test}   # 数据集划分（默认: test）
--patient-name NAME        # 指定患者名字（可选）
--mode {best_slice,all_slices}  # 可视化模式（默认: best_slice）
--max-samples INT          # 最多可视化样本数（可选）
--only-errors              # 只可视化错误预测
--colormap {jet,hot,viridis,plasma,inferno}  # 热力图颜色（默认: jet）
```

#### 其他参数
```bash
--feature-dim INT          # 特征维度（默认: 128）
--output-dir PATH          # 输出目录（默认: ./visualizations_model_based）
--seed INT                 # 随机种子（默认: 42）
--no-cuda                  # 不使用CUDA
```

## 使用场景

### 场景1: 快速查看测试集结果

```bash
# 查看所有测试集样本的最佳切片
python visualize_patch_attention_across_model.py \
    --model-path ./output_patch_based/train_patch_xxx/models/best_model.pth \
    --split test \
    --mode best_slice
```

**适用于:** 论文展示、快速评估

### 场景2: 深度分析错误预测

```bash
# 只看错误预测的患者，显示所有切片
python visualize_patch_attention_across_model.py \
    --model-path ./output_patch_based/train_patch_xxx/models/best_model.pth \
    --split test \
    --mode all_slices \
    --only-errors
```

**适用于:** 调试模型、分析失败案例

### 场景3: 验证集可视化

```bash
# 验证集前10个患者
python visualize_patch_attention_across_model.py \
    --model-path ./output_patch_based/train_patch_xxx/models/best_model.pth \
    --split val \
    --max-samples 10 \
    --mode all_slices
```

**适用于:** 模型调优、超参数选择

### 场景4: 指定患者分析

```bash
# 深度分析某个特定患者
python visualize_patch_attention_across_model.py \
    --model-path ./output_patch_based/train_patch_xxx/models/best_model.pth \
    --split test \
    --patient-name "A0001" \
    --mode all_slices
```

**适用于:** 个案研究、医学解释

### 场景5: 训练集可视化（检查过拟合）

```bash
# 训练集样本可视化
python visualize_patch_attention_across_model.py \
    --model-path ./output_patch_based/train_patch_xxx/models/best_model.pth \
    --split train \
    --max-samples 20 \
    --mode best_slice
```

**适用于:** 检查模型是否记住训练集、过拟合分析

### 场景6: 对比不同颜色映射

```bash
# 使用热力图配色
python visualize_patch_attention_across_model.py \
    --model-path ./output_patch_based/train_patch_xxx/models/best_model.pth \
    --split test \
    --colormap hot \
    --max-samples 5
```

**适用于:** 论文/展示，选择最佳视觉效果

## 快速开始

### 1. 使用示例脚本（推荐）

```bash
# 修改脚本中的MODEL_PATH
vim visualize_model_examples.sh

# 运行示例
bash visualize_model_examples.sh
```

### 2. 手动运行

```bash
# 替换为你的模型路径
MODEL_PATH="./output_patch_based/train_patch_20251226_193043/models/best_model.pth"

# 测试集 - 最佳切片
python visualize_patch_attention_across_model.py \
    --model-path "$MODEL_PATH" \
    --split test \
    --mode best_slice

# 验证集 - 所有切片（前5个）
python visualize_patch_attention_across_model.py \
    --model-path "$MODEL_PATH" \
    --split val \
    --mode all_slices \
    --max-samples 5

# 指定患者
python visualize_patch_attention_across_model.py \
    --model-path "$MODEL_PATH" \
    --split test \
    --patient-name "你的患者名字" \
    --mode all_slices
```

## 输出结果

### best_slice 模式

```
visualizations_model_based/test_best_slice/
├── 001_patient_A_slice042_correct.png
├── 002_patient_B_slice035_wrong.png
├── 003_patient_C_slice050_correct.png
└── ...
```

**文件命名规则:**
- `001` - 序号
- `patient_A` - 患者名字
- `slice042` - 切片索引
- `correct/wrong` - 预测是否正确

### all_slices 模式

```
visualizations_model_based/test_all_slices/
├── 001_patient_A_correct/
│   ├── _summary.txt                      # 统计摘要
│   ├── slice000_avg0.0123.png
│   ├── slice001_avg0.0145.png
│   ├── slice042_BEST_avg0.0850.png      # 最重要的切片
│   └── slice099_avg0.0098.png
├── 002_patient_B_wrong/
│   ├── _summary.txt
│   └── ...
└── ...
```

**_summary.txt 内容:**
```
Patient: patient_A
============================================================

Prediction:
  - True Label: 1 (Unstable)
  - Predicted: 1 (Unstable)
  - Probability: 0.8543
  - Status: ✓ Correct

Statistics:
  - Total Slices: 100
  - Visualized Slices: 85
  - Total Patches: 2048
  - Best Slice: 42

Top-10 Slices by Average Attention:
   1. Slice  42: 0.085032 ← BEST
   2. Slice  43: 0.072145
   3. Slice  41: 0.068234
   ...
```

## 可视化内容解读

### 四个子图说明

1. **Original Image** - 自适应裁剪后的原始图像（244×244）
2. **ROI Mask** - 红色覆盖显示感兴趣区域
3. **Attention Heatmap** - 颜色越热（红/黄）表示attention越高
4. **Top-K Patches** - 用彩色框标注最重要的patches，显示：
   - 排名（#1, #2, ...）
   - Attention权重值

### 标题信息

```
patient_A - Slice 42
True: Unstable | Pred: Unstable (prob=0.854) ✓
Patches: 24 | Avg Attention: 0.0850
```

- **True** - 真实标签
- **Pred** - 模型预测
- **prob** - 预测置信度
- **✓/✗** - 预测正确/错误
- **Patches** - 该切片的patch数量
- **Avg Attention** - 该切片的平均attention

## 性能优化建议

### 对于大数据集

```bash
# 使用max-samples限制数量
python visualize_patch_attention_across_model.py \
    --model-path xxx \
    --split train \
    --max-samples 50 \
    --mode best_slice
```

### 对于all_slices模式

- 每个患者会生成100张图（如果有100个切片）
- 建议先用`--max-samples`限制患者数量
- 或使用`--patient-name`只可视化特定患者

### GPU内存不足

```bash
# 使用CPU
python visualize_patch_attention_across_model.py \
    --model-path xxx \
    --no-cuda
```

## 常见问题

### Q1: 找不到患者名字

**问题:** 指定`--patient-name`后提示找不到

**解决:**
1. 不指定患者名字运行一次，查看输出的可用患者列表
2. 或查看数据集的label Excel文件

### Q2: 数据集划分不一致

**问题:** 可视化的train/val/test与训练时不一致

**解决:**
- 确保`--train-ratio`、`--val-ratio`、`--seed`与训练时一致
- 默认值已经与训练脚本对齐

### Q3: 可视化结果与训练时保存的不同

**原因:**
- 训练时可能使用了dropout
- 解决：模型已经设置为`eval()`模式，确保结果一致

### Q4: 如何查看某个模型在验证集上的表现

```bash
python visualize_patch_attention_across_model.py \
    --model-path ./output_patch_based/train_patch_xxx/models/best_model.pth \
    --split val \
    --mode best_slice
```

查看生成的图片，看correct/wrong的分布

## 进阶用法

### 批量对比多个模型

```bash
# 对比不同超参数的模型
for model in output_patch_based/*/models/best_model.pth; do
    exp_name=$(basename $(dirname $(dirname $model)))
    python visualize_patch_attention_across_model.py \
        --model-path "$model" \
        --split test \
        --max-samples 10 \
        --mode best_slice \
        --output-dir "./vis_comparison/$exp_name"
done
```

### 生成论文用图

```bash
# 选择最佳可视化效果
python visualize_patch_attention_across_model.py \
    --model-path best_model.pth \
    --split test \
    --patient-name "representative_case_1" \
    --mode all_slices \
    --colormap viridis \
    --output-dir ./paper_figures
```

## 与训练脚本的参数对应

| 训练参数 | 可视化参数 | 说明 |
|---------|-----------|------|
| `--patch-size 24` | `--patch-size 24` | 必须一致 |
| `--feature-dim 128` | `--feature-dim 128` | 必须一致 |
| `--max-patches-per-roi 12` | `--max-patches-per-roi 12` | 必须一致 |
| `--overlap-ratio 0.5` | `--overlap-ratio 0.5` | 必须一致 |
| `--depth 100` | `--depth 100` | 必须一致 |
| `--train-ratio 0.8` | `--train-ratio 0.8` | 确保划分一致 |
| `--val-ratio 0.1` | `--val-ratio 0.1` | 确保划分一致 |
| `--seed 42` | `--seed 42` | 确保可重复 |

**重要:** 除了模型路径，其他参数应该与训练时保持一致！

## 总结

**优势:**
- ✅ 直接加载模型，无需预先保存结果
- ✅ 支持train/val/test三个数据集
- ✅ 灵活指定患者和样本数量
- ✅ 实时计算attention权重
- ✅ 完整的可视化内容（4子图 + 详细信息）
- ✅ 与训练脚本参数完全对齐

**适用场景:**
- 模型调试和分析
- 论文图表生成
- 医学解释和个案研究
- 错误案例分析
- 数据集质量检查

**推荐工作流:**
1. 训练模型 → `train_patch_based.sh`
2. 快速查看测试集 → `--split test --mode best_slice`
3. 深度分析错误 → `--only-errors --mode all_slices`
4. 指定患者研究 → `--patient-name xxx --mode all_slices`

创建时间: 2025-12-27
