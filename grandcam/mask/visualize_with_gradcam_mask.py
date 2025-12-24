"""
使用GradCAM可视化带Mask的颈动脉斑块分类模型的预测结果
"""
import os
import argparse
import random
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# 或者更稳妥的写法（推荐）
current_dir = os.path.dirname(os.path.abspath(__file__))  # grandcam/mask
grandcam_dir = os.path.dirname(current_dir)               # grandcam
project_root = os.path.dirname(grandcam_dir)              # 项目根目录
sys.path.append(project_root)
from utils.dataset_carotid_mask import CarotidPlaqueDatasetWithMask
from models.resnet3D_mask import create_mask_guided_classifier

from gradcam_3d_mask import GradCAM3DWithMask


def set_seed(seed=42):
    """设置随机种子保证可复现性（与训练时保持一致）"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"✓ 随机种子设置为: {seed}")


def split_dataset(full_dataset, train_ratio=0.8, val_ratio=0.1, seed=42):
    """
    使用与训练时相同的方式划分数据集

    返回:
        train_indices, val_indices, test_indices
    """
    # 设置种子确保划分一致
    random.seed(seed)

    total_size = len(full_dataset)
    indices = list(range(total_size))
    random.shuffle(indices)

    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)

    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size+val_size]
    test_indices = indices[train_size+val_size:]

    print(f"\n数据集划分 (seed={seed}):")
    print(f"  训练集: {len(train_indices)} 样本")
    print(f"  验证集: {len(val_indices)} 样本")
    print(f"  测试集: {len(test_indices)} 样本")

    return train_indices, val_indices, test_indices


def visualize_all_patients(args):
    """
    对指定数据集的所有患者进行GradCAM可视化
    """
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    print(f"使用设备: {device}\n")

    # 设置随机种子
    set_seed(args.seed)

    # 数据变换
    transform = transforms.Compose([
        transforms.Resize((244, 244)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.35], std=[0.25])
    ])

    # 加载完整数据集（含mask）
    print("="*60)
    print("加载数据集（含Mask）...")
    print("="*60)

    full_dataset = CarotidPlaqueDatasetWithMask(
        root_dir=args.root_dir,
        mask_dir=args.mask_dir,
        label_excel=args.label_excel,
        transform=transform,
        keep_middle_n=args.depth,
        min_imgs_required=args.depth,
        verbose=True
    )

    # 划分数据集
    train_indices, val_indices, test_indices = split_dataset(
        full_dataset,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed
    )

    # 根据参数选择要可视化的数据集
    split_map = {
        'train': train_indices,
        'val': val_indices,
        'test': test_indices,
        'all': list(range(len(full_dataset)))
    }

    selected_indices = split_map[args.split]

    # 创建目标数据集
    dataset = CarotidPlaqueDatasetWithMask(
        root_dir=args.root_dir,
        mask_dir=args.mask_dir,
        label_excel=args.label_excel,
        transform=transform,
        keep_middle_n=args.depth,
        min_imgs_required=args.depth,
        verbose=False
    )
    dataset.persons = [full_dataset.persons[i] for i in selected_indices]

    print(f"\n✓ 选择数据集: {args.split.upper()}")
    print(f"✓ 样本数量: {len(dataset)}")

    # 创建DataLoader (batch_size=1 for GradCAM)
    dataloader = DataLoader(
        dataset, batch_size=1,
        shuffle=False, num_workers=args.num_workers
    )

    # 加载模型
    print("\n" + "="*60)
    print("加载Mask引导模型...")
    print("="*60)

    model = create_mask_guided_classifier(
        num_classes=2,
        pretrained_path=None,
        freeze_backbone=False
    )

    # 加载训练好的权重
    print(f"加载模型权重: {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()

    print("✓ 模型加载成功\n")

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 初始化GradCAM
    print("="*60)
    print(f"初始化GradCAM (目标层: {args.target_layer})...")
    print("="*60)

    gradcam = GradCAM3DWithMask(model, target_layer=args.target_layer)

    # 对每个患者进行可视化
    print("\n" + "="*60)
    print("开始生成GradCAM可视化（含梯度调试信息）...")
    print("="*60 + "\n")

    total_slices = 0

    for idx, (volume, mask, label) in enumerate(dataloader):
        volume = volume.to(device)  # [1, 1, D, H, W]
        mask = mask.to(device)      # [1, 1, D, H, W]
        label_value = label.item()

        # 前向传播获取预测
        with torch.no_grad():
            output = model(volume, mask)
            pred_class = output.argmax(dim=1).item()
            pred_prob = torch.softmax(output, dim=1)[0, pred_class].item()

        # 生成GradCAM (针对预测类别) + 梯度分布
        cam, gradient_map = gradcam(volume, mask, target_class=pred_class, return_gradients=True)

        # 患者信息
        person_info = dataset.persons[idx]
        patient_name = person_info['name']

        # 患者文件夹名称
        patient_folder_name = f'{patient_name}_label{label_value}_pred{pred_class}_prob{pred_prob:.3f}'

        print(f"[{idx+1}/{len(dataloader)}] 患者: {patient_name} | "
              f"真实标签: {label_value} | 预测: {pred_class} (概率: {pred_prob:.3f})")

        # 保存所有切片为独立文件（包含梯度图）
        num_slices = gradcam.visualize_all_slices(
            volume, mask, cam, args.output_dir,
            patient_id=patient_folder_name,
            alpha=args.alpha,
            verbose=True,
            gradient_map=gradient_map  # 传入梯度图
        )

        total_slices += num_slices

        # 限制可视化数量
        if args.max_samples > 0 and idx + 1 >= args.max_samples:
            print(f"\n已达到最大样本数限制: {args.max_samples}")
            break

    # 清理
    gradcam.remove_hooks()

    print("\n" + "="*60)
    print(f"✓ 完成! 共处理 {idx+1} 个患者，生成 {total_slices} 张切片")
    print(f"✓ 所有可视化已保存到: {args.output_dir}")
    print("="*60 + "\n")


def visualize_specific_patients(args):
    """
    对特定患者名称列表进行GradCAM可视化
    """
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    print(f"使用设备: {device}\n")

    # 设置随机种子
    set_seed(args.seed)

    # 数据变换
    transform = transforms.Compose([
        transforms.Resize((244, 244)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.35], std=[0.25])
    ])

    # 加载完整数据集
    print("="*60)
    print("加载数据集（含Mask）...")
    print("="*60)

    full_dataset = CarotidPlaqueDatasetWithMask(
        root_dir=args.root_dir,
        mask_dir=args.mask_dir,
        label_excel=args.label_excel,
        transform=transform,
        keep_middle_n=args.depth,
        min_imgs_required=args.depth,
        verbose=True
    )

    # 划分数据集
    train_indices, val_indices, test_indices = split_dataset(
        full_dataset,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed
    )

    # 根据参数选择搜索范围
    split_map = {
        'train': train_indices,
        'val': val_indices,
        'test': test_indices,
        'all': list(range(len(full_dataset)))
    }

    search_indices = split_map[args.split]

    # 创建搜索用的数据集
    dataset = CarotidPlaqueDatasetWithMask(
        root_dir=args.root_dir,
        mask_dir=args.mask_dir,
        label_excel=args.label_excel,
        transform=transform,
        keep_middle_n=args.depth,
        min_imgs_required=args.depth,
        verbose=False
    )
    dataset.persons = [full_dataset.persons[i] for i in search_indices]

    print(f"\n✓ 搜索范围: {args.split.upper()} ({len(dataset)} 个样本)")
    print(f"✓ 查找患者: {args.patient_ids}")

    # 加载模型
    print("\n" + "="*60)
    print("加载Mask引导模型...")
    print("="*60)

    model = create_mask_guided_classifier(
        num_classes=2,
        pretrained_path=None,
        freeze_backbone=False
    )

    print(f"加载模型权重: {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()

    print("✓ 模型加载成功\n")

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 初始化GradCAM
    print("="*60)
    print(f"初始化GradCAM (目标层: {args.target_layer})...")
    print("="*60)

    gradcam = GradCAM3DWithMask(model, target_layer=args.target_layer)

    # 找到指定患者名称的索引
    target_patients = args.patient_ids.split(',')
    patient_indices = []

    for idx, person in enumerate(dataset.persons):
        if person['name'] in target_patients:
            patient_indices.append(idx)

    print(f"\n找到 {len(patient_indices)} 个匹配的患者")

    # 对指定患者进行可视化
    print("\n" + "="*60)
    print("开始生成GradCAM可视化（含梯度调试信息）...")
    print("="*60 + "\n")

    total_slices = 0

    for count, idx in enumerate(patient_indices):
        # 获取单个样本
        volume, mask, label = dataset[idx]
        volume = volume.unsqueeze(0).to(device)  # [1, 1, D, H, W]
        mask = mask.unsqueeze(0).to(device)      # [1, 1, D, H, W]
        label_value = label

        # 前向传播获取预测
        with torch.no_grad():
            output = model(volume, mask)
            pred_class = output.argmax(dim=1).item()
            pred_prob = torch.softmax(output, dim=1)[0, pred_class].item()

        # 生成GradCAM + 梯度分布
        cam, gradient_map = gradcam(volume, mask, target_class=pred_class, return_gradients=True)

        # 患者信息
        person_info = dataset.persons[idx]
        patient_name = person_info['name']

        # 患者文件夹名称
        patient_folder_name = f'{patient_name}_label{label_value}_pred{pred_class}_prob{pred_prob:.3f}'

        print(f"[{count+1}/{len(patient_indices)}] 患者: {patient_name} | "
              f"真实标签: {label_value} | 预测: {pred_class} (概率: {pred_prob:.3f})")

        # 保存所有切片为独立文件（包含梯度图）
        num_slices = gradcam.visualize_all_slices(
            volume, mask, cam, args.output_dir,
            patient_id=patient_folder_name,
            alpha=args.alpha,
            verbose=True,
            gradient_map=gradient_map  # 传入梯度图
        )

        total_slices += num_slices

    # 清理
    gradcam.remove_hooks()

    print("\n" + "="*60)
    print(f"✓ 完成! 共处理 {len(patient_indices)} 个患者，生成 {total_slices} 张切片")
    print(f"✓ 所有可视化已保存到: {args.output_dir}")
    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(description='GradCAM可视化 - Mask引导模型')

    # 数据参数
    parser.add_argument('--root-dir', type=str,
                       default='/media/data/wjf/data/Carotid_artery',
                       help='数据根目录')
    parser.add_argument('--mask-dir', type=str,
                       default='/media/data/wjf/data/mask',
                       help='Mask目录')
    parser.add_argument('--label-excel', type=str,
                       default='/media/data/wjf/data/label_all_250+30+100.xlsx',
                       help='标签Excel文件')
    parser.add_argument('--output-dir', type=str,
                       default='./gradcam_visualizations_mask',
                       help='可视化输出目录')

    # 模型参数
    parser.add_argument('--model-path', type=str,
                       required=True,
                       help='训练好的模型权重路径')
    parser.add_argument('--target-layer', type=str,
                       default='layer4',
                       help='GradCAM目标层 (layer1/layer2/layer3/layer4)')

    # 可视化参数
    parser.add_argument('--depth', type=int, default=100,
                       help='3D volume深度')
    parser.add_argument('--alpha', type=float, default=0.4,
                       help='热力图叠加透明度')
    parser.add_argument('--max-samples', type=int, default=-1,
                       help='最大可视化样本数 (-1表示全部)')

    # 数据集划分参数
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子')
    parser.add_argument('--split', type=str, default='test',
                       choices=['train', 'val', 'test', 'all'],
                       help='选择要可视化的数据集')
    parser.add_argument('--train-ratio', type=float, default=0.8,
                       help='训练集比例')
    parser.add_argument('--val-ratio', type=float, default=0.1,
                       help='验证集比例')

    # 特定患者可视化
    parser.add_argument('--patient-ids', type=str,
                       help='指定患者名称列表，用逗号分隔')

    # 其他参数
    parser.add_argument('--num-workers', type=int, default=4,
                       help='数据加载线程数')
    parser.add_argument('--no-cuda', action='store_true',
                       help='不使用GPU')

    args = parser.parse_args()

    # 根据是否指定患者ID选择不同的可视化函数
    if args.patient_ids:
        visualize_specific_patients(args)
    else:
        visualize_all_patients(args)


if __name__ == '__main__':
    main()
