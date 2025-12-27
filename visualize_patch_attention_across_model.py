"""
基于模型的Patch Attention可视化脚本

功能:
1. 直接加载训练好的模型权重
2. 支持可视化 train/val/test 数据集
3. 支持指定患者名字可视化
4. 支持可视化最佳切片或所有切片
5. 使用模型前向传播实时获取attention权重

使用示例:
---------
# 可视化测试集的某个患者（所有切片）
python visualize_patch_attention_across_model.py \
    --model-path ./output_patch_based/train_patch_xxx/models/best_model.pth \
    --split test \
    --patient-name "patient_001" \
    --mode all_slices

# 可视化验证集的前5个患者（最佳切片）
python visualize_patch_attention_across_model.py \
    --model-path ./output_patch_based/train_patch_xxx/models/best_model.pth \
    --split val \
    --max-samples 5 \
    --mode best_slice

# 可视化训练集的所有患者（最佳切片）
python visualize_patch_attention_across_model.py \
    --model-path ./output_patch_based/train_patch_xxx/models/best_model.pth \
    --split train \
    --mode best_slice
"""
import os
import sys
import argparse
import random
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib import cm
import torch
import torchvision.transforms as transforms

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.dataset_patch_based import PatchBasedCarotidDataset, AdaptiveMaskCrop
from models.patch_classifier import create_patch_classifier


def set_seed(seed=42):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_model(model_path, patch_size=24, feature_dim=128, device='cuda'):
    """
    加载训练好的模型

    参数:
        model_path: 模型权重路径
        patch_size: patch大小
        feature_dim: 特征维度
        device: 设备

    返回:
        model: 加载好的模型
    """
    print(f"\n{'='*60}")
    print(f"加载模型: {model_path}")
    print(f"{'='*60}")

    model = create_patch_classifier(
        patch_size=patch_size,
        num_classes=2,
        feature_dim=feature_dim
    )

    # 加载权重
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    print(f"✓ 模型加载成功")
    print(f"  - Patch size: {patch_size}")
    print(f"  - Feature dim: {feature_dim}")
    print(f"  - Device: {device}")

    return model


def create_dataset(root_dir, mask_dir, label_excel, patch_size, max_patches_per_roi,
                   overlap_ratio, depth, split='test', train_ratio=0.8, val_ratio=0.1, seed=42):
    """
    创建数据集并划分

    参数:
        split: 'train' | 'val' | 'test'
        其他参数与训练脚本一致

    返回:
        dataset: 指定split的数据集
    """
    print(f"\n{'='*60}")
    print(f"创建 {split.upper()} 数据集...")
    print(f"{'='*60}")

    # 数据变换
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.35], std=[0.25])
    ])

    # 创建完整数据集
    full_dataset = PatchBasedCarotidDataset(
        root_dir=root_dir,
        mask_dir=mask_dir,
        label_excel=label_excel,
        patch_size=patch_size,
        max_patches_per_roi=max_patches_per_roi,
        overlap_ratio=overlap_ratio,
        keep_middle_n=depth,
        min_imgs_required=depth,
        transform=transform,
        verbose=False
    )

    # 划分数据集（与训练脚本保持一致）
    total_size = len(full_dataset)
    indices = list(range(total_size))
    random.seed(seed)
    random.shuffle(indices)

    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)

    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size+val_size]
    test_indices = indices[train_size+val_size:]

    # 选择对应的split
    if split == 'train':
        selected_indices = train_indices
    elif split == 'val':
        selected_indices = val_indices
    elif split == 'test':
        selected_indices = test_indices
    else:
        raise ValueError(f"Unknown split: {split}")

    # 创建子数据集
    dataset = PatchBasedCarotidDataset(
        root_dir=root_dir,
        mask_dir=mask_dir,
        label_excel=label_excel,
        patch_size=patch_size,
        max_patches_per_roi=max_patches_per_roi,
        overlap_ratio=overlap_ratio,
        keep_middle_n=depth,
        min_imgs_required=depth,
        transform=transform,
        verbose=False
    )
    dataset.persons = [full_dataset.persons[i] for i in selected_indices]

    print(f"✓ {split.upper()} 数据集创建完成")
    print(f"  - 样本数: {len(dataset)}")
    print(f"  - Patch size: {patch_size}")
    print(f"  - Max patches per ROI: {max_patches_per_roi}")
    print(f"  - Overlap ratio: {overlap_ratio}")
    print(f"  - Depth: {depth}")

    return dataset


def get_patient_index(dataset, patient_name):
    """
    根据患者名字获取索引

    参数:
        dataset: 数据集
        patient_name: 患者名字

    返回:
        idx: 患者索引，如果找不到返回None
    """
    for idx, person in enumerate(dataset.persons):
        if person['name'] == patient_name:
            return idx
    return None


def visualize_single_slice(img, mask, positions, attention_weights,
                           slice_idx, output_path, patient_name, label, pred, prob,
                           top_k=10, colormap='jet'):
    """
    可视化单个切片的patch attention（与原始可视化风格一致）

    参数:
        img: 原始图像 [H, W]
        mask: mask图像 [H, W]
        positions: patch位置列表
        attention_weights: attention权重 [N]
        slice_idx: 切片索引
        output_path: 输出路径
        patient_name: 患者名字
        label: 真实标签
        pred: 预测标签
        prob: 预测概率
        top_k: 显示top-k个patch
        colormap: 颜色映射
    """
    # 筛选当前slice的patches
    slice_indices = [i for i, pos in enumerate(positions) if pos['slice_idx'] == slice_idx]

    if len(slice_indices) == 0:
        return False

    slice_positions = [positions[i] for i in slice_indices]
    slice_attention = attention_weights[slice_indices]

    # 创建attention热力图
    h, w = img.shape
    heatmap = np.zeros((h, w), dtype=np.float32)
    count_map = np.zeros((h, w), dtype=np.float32)

    for pos, attn in zip(slice_positions, slice_attention):
        x1, y1, x2, y2 = pos['bbox']
        heatmap[y1:y2, x1:x2] += attn
        count_map[y1:y2, x1:x2] += 1

    # 平均
    valid_mask = count_map > 0
    heatmap[valid_mask] /= count_map[valid_mask]

    # 归一化
    if heatmap.max() > 0:
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())

    # ★ 与原始风格一致：使用cv2创建叠加图
    # 应用colormap
    heatmap_colored = (heatmap * 255).astype(np.uint8)
    heatmap_colored = cv2.applyColorMap(heatmap_colored, cv2.COLORMAP_JET)

    # 转换为RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    # 叠加热力图（50% + 50%）
    overlay = cv2.addWeighted(img_rgb, 0.5, heatmap_colored, 0.5, 0)

    # 标注Top-K重要的patch
    top_indices = np.argsort(slice_attention)[-top_k:][::-1]

    for rank, idx in enumerate(top_indices):
        pos = slice_positions[idx]
        x1, y1, x2, y2 = pos['bbox']
        weight = slice_attention[idx]

        # 绘制边界框（第一名绿色粗框，其他黄色细框）
        color = (0, 255, 0) if rank == 0 else (255, 255, 0)
        thickness = 3 if rank == 0 else 2
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, thickness)

        # 标注权重
        text = f'#{rank+1}: {weight:.3f}'
        cv2.putText(overlay, text, (x1, y1-5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    # ★ 创建与原始风格一致的4子图布局
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # 1. 原始图像
    axes[0].imshow(img, cmap='gray')
    axes[0].set_title('Original Image', fontsize=12)
    axes[0].axis('off')

    # 2. Mask（单独显示，灰度图）
    axes[1].imshow(mask, cmap='gray')
    axes[1].set_title('Mask (ROI)', fontsize=12)
    axes[1].axis('off')

    # 3. Attention热力图
    axes[2].imshow(heatmap, cmap='jet')
    axes[2].set_title('Attention Heatmap', fontsize=12)
    axes[2].axis('off')

    # 4. 叠加图（原图+热力图+top-k框）
    axes[3].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    axes[3].set_title(f'Overlay (Top-{min(top_k, len(slice_indices))} Patches)', fontsize=12)
    axes[3].axis('off')

    # 整体标题
    label_name = 'Stable' if label == 0 else 'Unstable'
    pred_name = 'Stable' if pred == 0 else 'Unstable'
    is_correct = (label == pred)
    status = '✓ Correct' if is_correct else '✗ Wrong'

    title = (f'{patient_name} - Slice {slice_idx} - {status}\n'
            f'True: {label_name}, Pred: {pred_name}, Conf: {prob:.3f}')

    fig.suptitle(title, fontsize=14, fontweight='bold',
                color='green' if is_correct else 'red')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    return True


def visualize_patient_best_slice(model, dataset, idx, output_dir, device='cuda',
                                 patch_size=24, colormap='jet'):
    """
    可视化患者的最佳切片（attention最高的切片）

    参数:
        model: 模型
        dataset: 数据集
        idx: 患者索引
        output_dir: 输出目录
        device: 设备
        patch_size: patch大小
        colormap: 颜色映射
    """
    # 获取数据
    patches, positions, label = dataset[idx]
    person_info = dataset.persons[idx]
    patient_name = person_info['name']

    # 前向传播
    patches_input = patches.unsqueeze(0).to(device)  # [1, N, C, H, W]

    with torch.no_grad():
        logits, attn_weights = model(patches_input, return_attention=True)

    probs = torch.softmax(logits, dim=1)
    pred = logits.argmax(dim=1).item()
    prob = probs[0, pred].item()
    label_val = label.item()

    # 转为numpy
    attn_weights_np = attn_weights[0].cpu().numpy()

    # 找到attention最高的patch所在的slice
    max_attn_idx = np.argmax(attn_weights_np)
    best_slice_idx = positions[max_attn_idx]['slice_idx']

    # 读取该slice的原图和mask
    img_path = person_info['paths'][best_slice_idx]
    mask_path = person_info['mask_path']

    img_original = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    mask_original = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    # 自适应裁剪（与训练一致）
    adaptive_crop = AdaptiveMaskCrop(
        target_size=(244, 244),
        padding_ratio=0,
        strategy='adaptive'
    )
    img, mask = adaptive_crop(img_original, mask_original)

    # 保存路径
    is_correct = (pred == label_val)
    status = 'correct' if is_correct else 'wrong'
    output_path = os.path.join(
        output_dir,
        f'{idx+1:03d}_{patient_name}_slice{best_slice_idx:03d}_{status}.png'
    )

    # 可视化
    visualize_single_slice(
        img, mask, positions, attn_weights_np,
        best_slice_idx, output_path, patient_name,
        label_val, pred, prob, colormap=colormap
    )

    print(f"  ✓ {patient_name} - Slice {best_slice_idx} (Best, Attn={attn_weights_np[max_attn_idx]:.4f})")


def visualize_patient_all_slices(model, dataset, idx, output_dir, device='cuda',
                                 patch_size=24, colormap='jet'):
    """
    可视化患者的所有切片

    参数:
        model: 模型
        dataset: 数据集
        idx: 患者索引
        output_dir: 输出目录
        device: 设备
        patch_size: patch大小
        colormap: 颜色映射
    """
    # 获取数据
    patches, positions, label = dataset[idx]
    person_info = dataset.persons[idx]
    patient_name = person_info['name']

    # 前向传播
    patches_input = patches.unsqueeze(0).to(device)

    with torch.no_grad():
        logits, attn_weights = model(patches_input, return_attention=True)

    probs = torch.softmax(logits, dim=1)
    pred = logits.argmax(dim=1).item()
    prob = probs[0, pred].item()
    label_val = label.item()

    # 转为numpy
    attn_weights_np = attn_weights[0].cpu().numpy()

    # 找到最佳slice
    max_attn_idx = np.argmax(attn_weights_np)
    best_slice_idx = positions[max_attn_idx]['slice_idx']

    # 创建患者文件夹
    is_correct = (pred == label_val)
    status = 'correct' if is_correct else 'wrong'
    patient_dir = os.path.join(output_dir, f'{idx+1:03d}_{patient_name}_{status}')
    os.makedirs(patient_dir, exist_ok=True)

    # 自适应裁剪
    adaptive_crop = AdaptiveMaskCrop(
        target_size=(244, 244),
        padding_ratio=0,
        strategy='adaptive'
    )

    # 统计每个slice的平均attention
    slice_stats = {}
    for i, pos in enumerate(positions):
        slice_idx = pos['slice_idx']
        if slice_idx not in slice_stats:
            slice_stats[slice_idx] = []
        slice_stats[slice_idx].append(attn_weights_np[i])

    slice_avg_attn = {s: np.mean(attns) for s, attns in slice_stats.items()}

    # 可视化所有slice
    num_slices = len(person_info['paths'])
    vis_count = 0

    for slice_idx in range(num_slices):
        if slice_idx not in slice_avg_attn:
            continue  # 该slice没有patch

        # 读取图像
        img_path = person_info['paths'][slice_idx]
        mask_path = person_info['mask_path']

        img_original = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask_original = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        img, mask = adaptive_crop(img_original, mask_original)

        # 文件名
        avg_attn = slice_avg_attn[slice_idx]
        is_best = (slice_idx == best_slice_idx)
        best_tag = '_BEST' if is_best else ''

        output_path = os.path.join(
            patient_dir,
            f'slice{slice_idx:03d}{best_tag}_avg{avg_attn:.4f}.png'
        )

        # 可视化
        success = visualize_single_slice(
            img, mask, positions, attn_weights_np,
            slice_idx, output_path, patient_name,
            label_val, pred, prob, colormap=colormap
        )

        if success:
            vis_count += 1

    # 保存summary
    summary_path = os.path.join(patient_dir, '_summary.txt')
    with open(summary_path, 'w') as f:
        f.write(f"Patient: {patient_name}\n")
        f.write(f"{'='*60}\n\n")
        f.write(f"Prediction:\n")
        f.write(f"  - True Label: {label_val} ({'Stable' if label_val==0 else 'Unstable'})\n")
        f.write(f"  - Predicted: {pred} ({'Stable' if pred==0 else 'Unstable'})\n")
        f.write(f"  - Probability: {prob:.4f}\n")
        f.write(f"  - Status: {'✓ Correct' if is_correct else '✗ Wrong'}\n\n")
        f.write(f"Statistics:\n")
        f.write(f"  - Total Slices: {num_slices}\n")
        f.write(f"  - Visualized Slices: {vis_count}\n")
        f.write(f"  - Total Patches: {len(positions)}\n")
        f.write(f"  - Best Slice: {best_slice_idx}\n\n")
        f.write(f"Top-10 Slices by Average Attention:\n")
        sorted_slices = sorted(slice_avg_attn.items(), key=lambda x: x[1], reverse=True)
        for rank, (s, avg_attn) in enumerate(sorted_slices[:10], 1):
            best_mark = ' ← BEST' if s == best_slice_idx else ''
            f.write(f"  {rank:2d}. Slice {s:3d}: {avg_attn:.6f}{best_mark}\n")

    print(f"  ✓ {patient_name} - All {vis_count} slices visualized")


def main():
    parser = argparse.ArgumentParser(description='基于模型的Patch Attention可视化')

    # 模型参数
    parser.add_argument('--model-path', type=str, required=True,
                       help='模型权重路径')
    parser.add_argument('--patch-size', type=int, default=24,
                       help='Patch大小')
    parser.add_argument('--feature-dim', type=int, default=128,
                       help='特征维度')

    # 数据参数（与训练脚本一致）
    parser.add_argument('--root-dir', type=str,
                       default='/media/data/wjf/data/Carotid_artery',
                       help='图像根目录')
    parser.add_argument('--mask-dir', type=str,
                       default='/media/data/wjf/data/mask',
                       help='Mask目录')
    parser.add_argument('--label-excel', type=str,
                       default='/media/data/wjf/data/label_all_250+30+100.xlsx',
                       help='标签文件')
    parser.add_argument('--max-patches-per-roi', type=int, default=12,
                       help='每个ROI最多提取的patch数')
    parser.add_argument('--overlap-ratio', type=float, default=0.5,
                       help='Patch重叠比例')
    parser.add_argument('--depth', type=int, default=100,
                       help='切片深度')
    parser.add_argument('--train-ratio', type=float, default=0.8,
                       help='训练集比例')
    parser.add_argument('--val-ratio', type=float, default=0.1,
                       help='验证集比例')

    # 可视化参数
    parser.add_argument('--split', type=str, default='test',
                       choices=['train', 'val', 'test'],
                       help='数据集划分：train/val/test')
    parser.add_argument('--patient-name', type=str, default=None,
                       help='指定患者名字（可选，不指定则可视化所有）')
    parser.add_argument('--mode', type=str, default='best_slice',
                       choices=['best_slice', 'all_slices'],
                       help='可视化模式：best_slice只看最佳切片，all_slices看所有切片')
    parser.add_argument('--max-samples', type=int, default=None,
                       help='最多可视化多少个样本（可选）')
    parser.add_argument('--only-errors', action='store_true',
                       help='只可视化错误预测的样本')
    parser.add_argument('--colormap', type=str, default='jet',
                       choices=['jet', 'hot', 'viridis', 'plasma', 'inferno'],
                       help='热力图颜色映射')

    # 输出参数
    parser.add_argument('--output-dir', type=str,
                       default='./visualizations_model_based',
                       help='输出目录')

    # 其他
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子')
    parser.add_argument('--no-cuda', action='store_true',
                       help='不使用CUDA')

    args = parser.parse_args()

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    print(f"\n{'='*60}")
    print(f"Patch Attention 可视化（基于模型）")
    print(f"{'='*60}")
    print(f"设备: {device}")

    # 设置随机种子
    set_seed(args.seed)

    # 加载模型
    model = load_model(
        args.model_path,
        patch_size=args.patch_size,
        feature_dim=args.feature_dim,
        device=device
    )

    # 创建数据集
    dataset = create_dataset(
        root_dir=args.root_dir,
        mask_dir=args.mask_dir,
        label_excel=args.label_excel,
        patch_size=args.patch_size,
        max_patches_per_roi=args.max_patches_per_roi,
        overlap_ratio=args.overlap_ratio,
        depth=args.depth,
        split=args.split,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed
    )

    # 创建输出目录
    output_dir = os.path.join(args.output_dir, f'{args.split}_{args.mode}')
    os.makedirs(output_dir, exist_ok=True)

    # 确定要可视化的样本
    if args.patient_name:
        # 指定患者
        idx = get_patient_index(dataset, args.patient_name)
        if idx is None:
            print(f"\n✗ 找不到患者: {args.patient_name}")
            print(f"可用的患者:")
            for i, person in enumerate(dataset.persons[:10]):
                print(f"  - {person['name']}")
            if len(dataset.persons) > 10:
                print(f"  ... 还有 {len(dataset.persons)-10} 个")
            return
        sample_indices = [idx]
    else:
        # 所有样本或前N个
        sample_indices = list(range(len(dataset)))
        if args.max_samples:
            sample_indices = sample_indices[:args.max_samples]

    # 如果只看错误预测，需要先筛选
    if args.only_errors:
        print(f"\n筛选错误预测的样本...")
        error_indices = []
        for idx in sample_indices:
            patches, positions, label = dataset[idx]
            patches_input = patches.unsqueeze(0).to(device)

            with torch.no_grad():
                logits, _ = model(patches_input, return_attention=True)

            pred = logits.argmax(dim=1).item()
            if pred != label.item():
                error_indices.append(idx)

        sample_indices = error_indices
        print(f"✓ 找到 {len(sample_indices)} 个错误预测")

    # 可视化
    print(f"\n{'='*60}")
    print(f"开始可视化...")
    print(f"  - Split: {args.split}")
    print(f"  - Mode: {args.mode}")
    print(f"  - Samples: {len(sample_indices)}")
    print(f"  - Output: {output_dir}")
    print(f"{'='*60}\n")

    for i, idx in enumerate(sample_indices, 1):
        print(f"[{i}/{len(sample_indices)}]", end=' ')

        if args.mode == 'best_slice':
            visualize_patient_best_slice(
                model, dataset, idx, output_dir,
                device=device, patch_size=args.patch_size,
                colormap=args.colormap
            )
        else:  # all_slices
            visualize_patient_all_slices(
                model, dataset, idx, output_dir,
                device=device, patch_size=args.patch_size,
                colormap=args.colormap
            )

    print(f"\n{'='*60}")
    print(f"✓ 可视化完成！")
    print(f"{'='*60}")
    print(f"输出目录: {output_dir}")
    print(f"可视化样本数: {len(sample_indices)}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
