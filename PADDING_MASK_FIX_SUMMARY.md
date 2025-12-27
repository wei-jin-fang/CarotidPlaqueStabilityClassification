# Padding Mask 修复总结

## 问题描述

在之前的实现中，`pad_patches_batch()` 函数创建了 padding masks 来标识哪些 patches 是真实的（1），哪些是填充的（0），但这个 mask **从未被传递给模型使用**。

### 问题影响

1. **性能影响**：当 batch 中不同样本的 patch 数量不同时，会用零填充到相同长度
   - 例如：样本 A 有 2000 个 patches，样本 B 有 2400 个 patches
   - 样本 A 会被 padding 400 个全零的 patches
   - 模型的 attention 机制会对这 400 个**无效的** padding patches 计算注意力权重

2. **注意力分散**：模型在无效的 padding patches 上浪费了注意力，影响了对真实 patches 的关注

3. **结果不准确**：Attention 权重包含了对 padding patches 的权重，可视化时会显示错误的重要性分布

## 修复方案

### 1. 修改 `AttentionAggregator` (models/patch_classifier.py)

**位置**：第 95-124 行

**关键改动**：
```python
def forward(self, features, mask=None):
    """
    features: [B, N, D]
    mask: [B, N] 可选，1表示有效patch，0表示padding
    """
    attn_logits = self.attention_net(features)  # [B, N, 1]
    attn_logits = attn_logits.squeeze(-1)  # [B, N]

    # ★ 应用mask：将padding位置设为-inf，softmax后权重为0
    if mask is not None:
        mask_value = -1e9
        attn_logits = attn_logits.masked_fill(mask == 0, mask_value)

    # Softmax得到权重（padding位置权重≈0）
    attn_weights = F.softmax(attn_logits, dim=1)

    # 加权求和
    aggregated = torch.bmm(
        attn_weights.unsqueeze(1),
        features
    ).squeeze(1)

    return aggregated, attn_weights
```

**原理**：
- 将 padding 位置的 attention logits 设置为 -1e9（近似 -∞）
- 经过 softmax 后，这些位置的权重接近 0
- 只有真实 patches 获得有意义的 attention 权重

### 2. 修改 `PatchBasedClassifier` (models/patch_classifier.py)

**位置**：第 166-197 行

**关键改动**：
```python
def forward(self, patches, mask=None, return_attention=True):
    """
    patches: [B, N_patches, C, H, W]
    mask: [B, N_patches] 可选，1表示有效patch，0表示padding
    """
    # ... 特征提取 ...

    # 传递mask给aggregator
    aggregated, attn_weights = self.aggregator(features, mask=mask)

    # ... 分类 ...
    return logits, attn_weights
```

### 3. 修改训练循环 (train_patch_based.py)

**train_one_epoch() - 第 138-171 行**：
```python
patches_padded, masks = pad_patches_batch(patches_list)
patches_padded = patches_padded.to(device)
masks = masks.to(device)  # ★ 将mask也移到device
labels = labels.to(device)

# 前向传播（传递mask）
logits, attn_weights = model(patches_padded, mask=masks, return_attention=True)
```

**validate() - 第 174-223 行**：
```python
patches_padded, masks = pad_patches_batch(patches_list)
patches_padded = patches_padded.to(device)
masks = masks.to(device)  # ★ 将mask也移到device
labels = labels.to(device)

# 前向传播（传递mask）
logits, attn_weights = model(patches_padded, mask=masks, return_attention=True)
```

**evaluate_test_with_attention() - 无需修改**：
- 单样本评估，没有 padding
- mask 是可选参数，不传即可

## 修复效果

### 修复前：
```
样本A: [patch1, patch2, ..., patch2000, 0, 0, ..., 0]  ← 400个零patch
                                       ↓
Attention: [0.0004, 0.0005, ..., 0.0003, 0.0001, 0.0001, ...]
           ↑ 真实patches                ↑ padding也有权重！
```

### 修复后：
```
样本A: [patch1, patch2, ..., patch2000, 0, 0, ..., 0]
                                       ↓
Mask:  [1,      1,      ..., 1,       0, 0, ..., 0]
                                       ↓
Attention: [0.0005, 0.0006, ..., 0.0004, ~0, ~0, ...]
           ↑ 真实patches                ↑ padding权重≈0
```

## 预期改进

1. **训练效果提升**：模型不再在无效 patches 上浪费注意力
2. **收敛更快**：注意力集中在有效 patches 上
3. **可视化更准确**：Attention 权重只分布在真实 patches 上
4. **结果更可靠**：消除了 padding 对预测的干扰

## 修改文件清单

```
models/patch_classifier.py
├── AttentionAggregator.forward()    ← 接受并使用mask
└── PatchBasedClassifier.forward()   ← 接受并传递mask

train_patch_based.py
├── train_one_epoch()                ← 传递mask到模型
├── validate()                       ← 传递mask到模型
└── evaluate_test_with_attention()   ← 无需修改（单样本无padding）
```

## 向后兼容性

- mask 参数是**可选的**（默认为 None）
- 不传递 mask 时，模型行为和之前一样（但不推荐）
- 已有的模型权重文件**无需重新训练**即可使用新功能
- 建议重新训练以获得最佳性能

## 使用建议

1. **必须重新训练**：虽然模型结构兼容，但为了获得 padding mask 带来的性能提升，强烈建议重新训练模型
2. **检查 attention 分布**：重新训练后，检查可视化结果，确认 padding patches 的 attention 权重接近 0
3. **性能对比**：对比修复前后的模型性能（准确率、F1、AUC等）

## 技术细节

### 为什么使用 -1e9 而不是 -inf？

```python
mask_value = -1e9  # 而不是 float('-inf')
attn_logits = attn_logits.masked_fill(mask == 0, mask_value)
```

原因：
1. `-inf` 在某些 PyTorch 版本可能导致 NaN
2. `-1e9` 经过 softmax 后约等于 0（exp(-1e9) ≈ 0）
3. 数值上更稳定，避免梯度计算问题

### Masked Softmax 数学原理

```
原始: softmax(x_i) = exp(x_i) / Σ exp(x_j)

应用mask:
- 真实位置: x_i (不变)
- Padding位置: x_i = -1e9

结果:
- 真实位置: exp(x_i) / (Σ_real exp(x_j) + ε)  ← ε ≈ 0
- Padding位置: exp(-1e9) / (...) ≈ 0
```

## 验证方法

训练后，检查 attention 权重：

```python
# 打印某个batch的attention权重和mask
patches_padded, masks = pad_patches_batch(patches_list)
logits, attn_weights = model(patches_padded, mask=masks, return_attention=True)

print("Mask:", masks[0])  # 例如: [1,1,1,...,1,0,0,0]
print("Attention:", attn_weights[0])  # padding位置应该≈0
```

预期结果：
- Mask 为 1 的位置：attention 权重正常（>0）
- Mask 为 0 的位置：attention 权重 ≈ 0（例如 1e-8）

## 修复日期

2025-12-27

## 致谢

感谢用户发现这个关键问题：
> "patches_padded, masks = pad_patches_batch(patches_list) 你这个mask为啥没传给模型使用啊,那你padd出来的token多了很多0token吗"

这个问题的发现和修复对模型性能至关重要！
