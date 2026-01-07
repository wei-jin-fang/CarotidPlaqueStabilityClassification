"""
可视化指定患者的指定切片的Patch Attention

功能:
1. 加载训练好的模型
2. 可视化指定患者的指定切片的attention权重
3. 支持指定患者名字和切片索引
4. 生成详细的可视化结果（原图、Mask、热力图、叠加图）

使用示例:
---------
# 可视化测试集中"patient_001"的第50张切片
python visualize_patch_attention_across_model_person_slice.py \
    --model-path ./output_patch_based/train_patch_xxx/models/best_model.pth \
    --patient-name "patient_001" \
    --slice-idx 50 \
    --split test

# 可视化训练集中"patient_002"的第30张切片，保存到指定目录
python visualize_patch_attention_across_model_person_slice.py \
    --model-path ./output_patch_based/train_patch_xxx/models/best_model.pth \
    --patient-name "patient_002" \
    --slice-idx 30 \
    --split train \
    --output-dir ./my_visualizations
"""
import os
import sys
import argparse
import random
import numpy as np
import cv2
import matplotlib.pyplot as plt
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


def load_model(model_path, patch_size=24, feature_dim=64, device='cuda'):
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
    可视化单个切片的patch attention

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
        print(f"✗ 切片 {slice_idx} 没有patch，无法可视化")
        return False

    slice_positions = [positions[i] for i in slice_indices]
    slice_attention = attention_weights[slice_indices]

    print(f"\n切片 {slice_idx} 信息:")
    print(f"  - Patch数量: {len(slice_indices)}")
    print(f"  - Attention范围: [{slice_attention.min():.4f}, {slice_attention.max():.4f}]")
    print(f"  - 平均Attention: {slice_attention.mean():.4f}")

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

    # 使用cv2创建叠加图
    heatmap_colored = (heatmap * 255).astype(np.uint8)
    heatmap_colored = cv2.applyColorMap(heatmap_colored, cv2.COLORMAP_JET)

    # 转换为RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    # 叠加热力图（50% + 50%）
    overlay = cv2.addWeighted(img_rgb, 0.5, heatmap_colored, 0.5, 0)

    # 标注Top-K重要的patch
    top_indices = np.argsort(slice_attention)[-top_k:][::-1]

    print(f"\n  Top-{min(top_k, len(slice_indices))} Patches:")
    for rank, idx in enumerate(top_indices):
        pos = slice_positions[idx]
        x1, y1, x2, y2 = pos['bbox']
        weight = slice_attention[idx]

        print(f"    #{rank+1}: Attention={weight:.4f}, BBox=({x1},{y1},{x2},{y2})")

        # 绘制边界框（第一名绿色粗框，其他黄色细框）
        color = (0, 255, 0) if rank == 0 else (255, 255, 0)
        thickness = 3 if rank == 0 else 2
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, thickness)

        # 标注权重
        text = f'#{rank+1}: {weight:.3f}'
        cv2.putText(overlay, text, (x1, y1-5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    # 创建4子图布局
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # 1. 原始图像
    axes[0].imshow(img, cmap='gray')
    axes[0].set_title('Original Image', fontsize=12)
    axes[0].axis('off')

    # 2. Mask
    axes[1].imshow(mask, cmap='gray')
    axes[1].set_title('Mask (ROI)', fontsize=12)
    axes[1].axis('off')

    # 3. Attention热力图
    axes[2].imshow(heatmap, cmap='jet')
    axes[2].set_title('Attention Heatmap', fontsize=12)
    axes[2].axis('off')

    # 4. 叠加图
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

    print(f"\n✓ 可视化结果已保存: {output_path}")

    return True


def visualize_patient_slice(model, dataset, patient_idx, slice_idx, output_dir,
                            device='cuda', patch_size=24, colormap='jet'):
    """
    可视化指定患者的指定切片

    参数:
        model: 模型
        dataset: 数据集
        patient_idx: 患者索引
        slice_idx: 切片索引
        output_dir: 输出目录
        device: 设备
        patch_size: patch大小
        colormap: 颜色映射
    """
    # 获取数据
    patches, positions, label = dataset[patient_idx]
    person_info = dataset.persons[patient_idx]
    patient_name = person_info['name']

    print(f"\n{'='*60}")
    print(f"患者信息:")
    print(f"{'='*60}")
    print(f"  - 名字: {patient_name}")
    print(f"  - 真实标签: {label.item()} ({'Stable' if label.item()==0 else 'Unstable'})")
    print(f"  - 总切片数: {len(person_info['paths'])}")
    print(f"  - 总Patch数: {len(positions)}")
    print(f"  - 目标切片: {slice_idx}")

    # 检查切片索引是否有效
    if slice_idx >= len(person_info['paths']):
        print(f"\n✗ 错误: 切片索引 {slice_idx} 超出范围 [0, {len(person_info['paths'])-1}]")
        return False

    # 检查该切片是否有patch
    slice_has_patches = any(pos['slice_idx'] == slice_idx for pos in positions)
    if not slice_has_patches:
        print(f"\n✗ 警告: 切片 {slice_idx} 没有提取到patch，可能在ROI外或被过滤")

        # 显示有patch的切片
        slice_set = sorted(set(pos['slice_idx'] for pos in positions))
        print(f"\n有patch的切片索引: {slice_set}")
        return False

    # 前向传播
    patches_input = patches.unsqueeze(0).to(device)

    with torch.no_grad():
        logits, attn_weights = model(patches_input, return_attention=True)

    probs = torch.softmax(logits, dim=1)
    pred = logits.argmax(dim=1).item()
    prob = probs[0, pred].item()
    label_val = label.item()

    print(f"\n模型预测:")
    print(f"  - 预测标签: {pred} ({'Stable' if pred==0 else 'Unstable'})")
    print(f"  - 预测概率: {prob:.4f}")
    print(f"  - 预测状态: {'✓ 正确' if pred==label_val else '✗ 错误'}")

    # 转为numpy
    attn_weights_np = attn_weights[0].cpu().numpy()

    # 读取图像
    img_path = person_info['paths'][slice_idx]
    mask_path = person_info['mask_path']

    img_original = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    mask_original = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    # 自适应裁剪
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
        f'{patient_name}_slice{slice_idx:03d}_{status}.png'
    )

    # 可视化
    success = visualize_single_slice(
        img, mask, positions, attn_weights_np,
        slice_idx, output_path, patient_name,
        label_val, pred, prob, colormap=colormap
    )

    return success


def main():
    parser = argparse.ArgumentParser(description='可视化指定患者的指定切片的Patch Attention')

    # 模型参数
    parser.add_argument('--model-path', type=str, required=True,
                       help='模型权重路径')
    parser.add_argument('--patch-size', type=int, default=24,
                       help='Patch大小')
    parser.add_argument('--feature-dim', type=int, default=64,
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
    parser.add_argument('--patient-name', type=str, required=True,
                       help='患者名字（必须）')
    parser.add_argument('--slice-idx', type=int, required=True,
                       help='切片索引（必须）')
    parser.add_argument('--colormap', type=str, default='jet',
                       choices=['jet', 'hot', 'viridis', 'plasma', 'inferno'],
                       help='热力图颜色映射')

    # 输出参数
    parser.add_argument('--output-dir', type=str,
                       default='./visualizations_single_slice',
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
    print(f"单切片 Patch Attention 可视化")
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

    # 查找患者
    print(f"\n{'='*60}")
    print(f"查找患者: {args.patient_name}")
    print(f"{'='*60}")

    patient_idx = get_patient_index(dataset, args.patient_name)
    if patient_idx is None:
        print(f"\n✗ 找不到患者: {args.patient_name}")
        print(f"\n可用的患者 ({args.split}集):")
        for i, person in enumerate(dataset.persons[:20]):
            print(f"  [{i}] {person['name']} (标签={person['label']})")
        if len(dataset.persons) > 20:
            print(f"  ... 还有 {len(dataset.persons)-20} 个患者")
        return

    print(f"✓ 找到患者: {args.patient_name} (索引={patient_idx})")

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 可视化
    print(f"\n{'='*60}")
    print(f"开始可视化...")
    print(f"{'='*60}")

    success = visualize_patient_slice(
        model, dataset, patient_idx, args.slice_idx,
        args.output_dir, device=device,
        patch_size=args.patch_size,
        colormap=args.colormap
    )

    if success:
        print(f"\n{'='*60}")
        print(f"✓ 可视化完成！")
        print(f"{'='*60}")
        print(f"输出目录: {args.output_dir}")
        print(f"{'='*60}\n")
    else:
        print(f"\n{'='*60}")
        print(f"✗ 可视化失败")
        print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
