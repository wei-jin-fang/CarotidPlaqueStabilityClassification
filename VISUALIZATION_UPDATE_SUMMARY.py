"""
可视化功能扩展总结

已实现的功能：
===============

1. 原有功能（保留）
   - visualize_all_samples()
   - 只可视化最重要的slice（attention最高的patch所在的slice）
   - 快速浏览所有患者

2. 新增功能
   - visualize_all_slices_per_patient()
   - 为每个患者创建文件夹
   - 可视化该患者的所有slice
   - 深度分析模型行为

3. 模式选择（--mode参数）
   - best_slice: 原有模式
   - all_slices: 新增模式

使用方法：
=========

模式1：只看最重要的slice（默认）
--------------------------------
python visualize_patch_attention.py \
    --results-file results.pkl \
    --mode best_slice \
    --output-dir ./vis_best

输出：
./vis_best/
├── 001_patient_A_slice42.png
├── 002_patient_B_slice35.png
└── ...

模式2：查看所有slice（新功能）
------------------------------
python visualize_patch_attention.py \
    --results-file results.pkl \
    --mode all_slices \
    --output-dir ./vis_all

输出：
./vis_all/
├── 001_patient_A/
│   ├── _summary.txt
│   ├── slice000_avg0.0123.png
│   ├── slice042_BEST_avg0.0850.png  ← 最重要的
│   └── slice099_avg0.0098.png
├── 002_patient_B/
│   └── ...
└── ...

关键特性：
=========

1. 文件命名包含信息
   - slice索引（3位数补0，方便排序）
   - BEST标记（最重要的slice）
   - 平均attention值（快速判断重要性）

2. summary.txt文件
   - 患者基本信息（预测、真实标签、置信度）
   - 统计信息（总slice数、总patch数）
   - 最重要slice
   - Top-10 slice的attention排名

3. 灵活的参数控制
   - --max-samples: 限制患者数量
   - --only-errors: 只分析错误预测
   - --analyze-stats: 显示统计信息

实现细节：
=========

1. 保留原有功能
   - visualize_all_samples() 函数不变
   - 默认模式仍是best_slice
   - 向后兼容

2. 新增all_slices模式
   - 遍历患者的所有slice
   - 每个slice单独提取patch和attention
   - 计算平均attention用于排序
   - 标注最重要的slice

3. 文件组织
   - 每个患者一个文件夹
   - summary.txt汇总信息
   - 图片按slice索引命名

适用场景：
=========

best_slice模式：
- 快速浏览测试集结果
- 论文中展示典型案例
- 批量查看模型关注点

all_slices模式：
- 深度分析模型行为
- 研究attention在深度方向的分布
- 找出模型失败的原因
- 医学解释（展示完整病变区域）

示例：
=====

# 快速测试（前5个患者的所有slice）
python visualize_patch_attention.py \
    --results-file results.pkl \
    --mode all_slices \
    --max-samples 5 \
    --output-dir ./test_vis

# 分析错误预测
python visualize_patch_attention.py \
    --results-file results.pkl \
    --mode all_slices \
    --only-errors \
    --output-dir ./error_analysis

# 完整分析前10个患者
python visualize_patch_attention.py \
    --results-file results.pkl \
    --mode all_slices \
    --max-samples 10 \
    --analyze-stats \
    --output-dir ./full_analysis

文件清单：
=========

修改的文件：
- visualize_patch_attention.py
  + 新增 visualize_all_slices_per_patient() 函数
  + 修改 main() 函数，添加 --mode 参数
  + 保留原有 visualize_all_samples() 函数

新增文件：
- VISUALIZATION_GUIDE.md (使用指南)
- visualize_examples.sh (使用示例脚本)

创建时间：2025-12-26
"""
print(__doc__)
