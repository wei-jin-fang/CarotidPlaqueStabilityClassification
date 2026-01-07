"""
基于Patch的颈动脉斑块分类测试脚本

功能:
1. 加载训练好的模型
2. 在测试集上评估
3. 保存详细的预测结果和attention权重
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import argparse
import json
import random
import numpy as np
import pandas as pd
from datetime import datetime
import pickle

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)

from utils.dataset_patch_based import PatchBasedCarotidDataset
from models.patch_classifier import create_patch_classifier


def set_seed(seed=42):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"✓ 随机种子设置为: {seed}")


def create_output_folder(base_dir="./output_patch_based"):
    """创建测试输出文件夹"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(base_dir, f"test_patch_{timestamp}")

    sub_dirs = {
        'root': exp_dir,
        'results': os.path.join(exp_dir, 'results'),
        'logs': os.path.join(exp_dir, 'logs'),
    }

    for dir_path in sub_dirs.values():
        os.makedirs(dir_path, exist_ok=True)

    print(f"\n✓ 创建测试输出文件夹: {exp_dir}\n")
    return exp_dir, sub_dirs


def collate_fn_patch(batch):
    """
    自定义collate函数，处理不同数量的patch

    返回:
        patches: list of [N_i, C, H, W] tensors
        positions: list of list of dict
        labels: [B] tensor
    """
    patches_list = []
    positions_list = []
    labels_list = []

    for patches, positions, label in batch:
        patches_list.append(patches)
        positions_list.append(positions)
        labels_list.append(label)

    labels = torch.stack(labels_list)

    return patches_list, positions_list, labels


def pad_patches_batch(patches_list, max_patches=None):
    """
    将不同数量的patch padding到相同长度

    参数:
        patches_list: list of [N_i, C, H, W] tensors
        max_patches: 最大patch数（None则使用batch中的最大值）

    返回:
        padded: [B, max_N, C, H, W] tensor
        masks: [B, max_N] tensor, 1表示有效patch，0表示padding
    """
    batch_size = len(patches_list)

    if max_patches is None:
        max_patches = max(p.size(0) for p in patches_list)

    # 获取patch尺寸
    C, H, W = patches_list[0].shape[1:]

    # 创建padding后的tensor
    padded = torch.zeros(batch_size, max_patches, C, H, W)
    masks = torch.zeros(batch_size, max_patches)

    for i, patches in enumerate(patches_list):
        n = patches.size(0)
        n = min(n, max_patches)  # 限制最大数量
        padded[i, :n] = patches[:n]
        masks[i, :n] = 1

    return padded, masks


def validate(model, dataloader, criterion, device):
    """验证模型"""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for patches_list, positions_list, labels in dataloader:
            # Padding
            patches_padded, masks = pad_patches_batch(patches_list)
            patches_padded = patches_padded.to(device)
            masks = masks.to(device)
            labels = labels.to(device)

            # 前向传播
            logits, attn_weights = model(patches_padded, mask=masks, return_attention=True)
            loss = criterion(logits, labels)

            total_loss += loss.item()
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='binary', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='binary', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='binary', zero_division=0)

    try:
        auc = roc_auc_score(all_labels, all_probs)
    except:
        auc = 0.0

    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'predictions': all_preds,
        'labels': all_labels,
        'probabilities': all_probs
    }


def evaluate_test_with_attention(model, test_dataset, device, save_dir):
    """
    详细评估测试集，保存每个样本的预测结果和attention权重

    关键：保存positions信息用于可视化
    """
    model.eval()
    results = []

    print("\n正在评估测试集...")

    with torch.no_grad():
        for idx in range(len(test_dataset)):
            patches, positions, label = test_dataset[idx]

            # 添加batch维度
            patches = patches.unsqueeze(0).to(device)  # [1, N, C, H, W]

            # 前向传播
            logits, attn_weights = model(patches, return_attention=True)
            probs = torch.softmax(logits, dim=1)
            pred = logits.argmax(dim=1).item()

            prob_class0 = probs[0, 0].item()
            prob_class1 = probs[0, 1].item()

            person_info = test_dataset.persons[idx]
            patient_name = person_info['name']
            true_label = label.item()
            is_correct = (pred == true_label)

            # 获取attention权重（numpy数组）
            attn_weights_np = attn_weights[0].cpu().numpy()

            results.append({
                'patient_name': patient_name,
                'true_label': true_label,
                'predicted_label': pred,
                'is_correct': is_correct,
                'prob_class_0': prob_class0,
                'prob_class_1': prob_class1,
                'confidence': max(prob_class0, prob_class1),
                'num_patches': len(positions),
                'attention_weights': attn_weights_np,
                'positions': positions  # 保存位置信息！
            })

            if (idx + 1) % 5 == 0:
                print(f"  已评估 {idx + 1}/{len(test_dataset)} 个样本")

    # 保存详细结果（包含attention和positions）
    results_path = os.path.join(save_dir, 'test_predictions_with_attention.pkl')
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"\n✓ 详细结果（含attention）已保存: {results_path}")

    # 保存简化版CSV（不含positions）
    df_data = []
    for r in results:
        df_data.append({
            'patient_name': r['patient_name'],
            'true_label': r['true_label'],
            'predicted_label': r['predicted_label'],
            'is_correct': r['is_correct'],
            'prob_class_0': r['prob_class_0'],
            'prob_class_1': r['prob_class_1'],
            'confidence': r['confidence'],
            'num_patches': r['num_patches'],
            'top3_attention': str(np.sort(r['attention_weights'])[-3:][::-1])
        })

    df = pd.DataFrame(df_data)
    csv_path = os.path.join(save_dir, 'test_predictions_detailed.csv')
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"✓ CSV结果已保存: {csv_path}")

    # 统计
    accuracy = df['is_correct'].sum() / len(df)
    print(f"\n测试集准确率: {accuracy:.4f}")
    print(f"正确预测: {df['is_correct'].sum()}/{len(df)}")

    return df, results


def save_test_samples(test_dataset, save_dir):
    """保存测试集样本名称"""
    samples = []
    for person in test_dataset.persons:
        samples.append({
            'patient_name': person['name'],
            'label': person['label'],
            'mask_path': person['mask_path'],
            'num_slices': len(person['paths'])
        })

    df = pd.DataFrame(samples)
    csv_path = os.path.join(save_dir, 'test_samples.csv')
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"✓ 测试集样本已保存: {csv_path} ({len(samples)} 样本)")


def test(args):
    """主测试函数"""
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    print(f"使用设备: {device}")

    set_seed(args.seed)

    exp_dir, sub_dirs = create_output_folder(args.output_dir)

    # 保存配置
    config_path = os.path.join(sub_dirs['logs'], 'test_config.json')
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=4)

    # 数据变换
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.35], std=[0.25])
    ])

    # 加载数据集
    print("\n" + "="*60)
    print("加载Patch-based测试数据集...")
    print("="*60)

    full_dataset = PatchBasedCarotidDataset(
        root_dir=args.root_dir,
        mask_dir=args.mask_dir,
        label_excel=args.label_excel,
        patch_size=args.patch_size,
        max_patches_per_roi=args.max_patches_per_roi,
        overlap_ratio=args.overlap_ratio,
        keep_middle_n=args.depth,
        min_imgs_required=args.depth,
        transform=transform,
        verbose=True
    )

    # 划分数据集（保持与训练时相同的划分）
    total_size = len(full_dataset)
    indices = list(range(total_size))
    random.shuffle(indices)

    train_size = int(args.train_ratio * total_size)
    val_size = int(args.val_ratio * total_size)

    test_indices = indices[train_size+val_size:]

    print(f"\n测试集大小: {len(test_indices)} 样本")

    # 创建测试数据集
    test_dataset = PatchBasedCarotidDataset(
        root_dir=args.root_dir,
        mask_dir=args.mask_dir,
        label_excel=args.label_excel,
        patch_size=args.patch_size,
        max_patches_per_roi=args.max_patches_per_roi,
        overlap_ratio=args.overlap_ratio,
        keep_middle_n=args.depth,
        min_imgs_required=args.depth,
        transform=transform,
        verbose=False
    )
    test_dataset.persons = [full_dataset.persons[i] for i in test_indices]

    # 保存测试样本信息
    save_test_samples(test_dataset, sub_dirs['logs'])

    # DataLoader
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers,
        collate_fn=collate_fn_patch, pin_memory=True
    )

    # 创建模型
    print("\n" + "="*60)
    print("创建Patch-based分类器...")
    print("="*60)

    model = create_patch_classifier(
        patch_size=args.patch_size,
        num_classes=2,
        feature_dim=args.feature_dim
    )
    model = model.to(device)

    # 加载模型权重
    print(f"\n加载模型权重: {args.model_path}")
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    print("✓ 模型加载成功")

    # 损失函数
    criterion = nn.CrossEntropyLoss()

    # 测试集评估
    print("\n" + "="*60)
    print("测试集评估...")
    print("="*60)

    test_metrics = validate(model, test_loader, criterion, device)

    print(f"\n测试集结果:")
    print(f"  准确率: {test_metrics['accuracy']:.4f}")
    print(f"  精确率: {test_metrics['precision']:.4f}")
    print(f"  召回率: {test_metrics['recall']:.4f}")
    print(f"  F1分数: {test_metrics['f1']:.4f}")
    print(f"  AUC:   {test_metrics['auc']:.4f}")

    cm = confusion_matrix(test_metrics['labels'], test_metrics['predictions'])
    print(f"\n混淆矩阵:\n{cm}")

    # 保存结果
    results = {
        'model_path': args.model_path,
        'test_accuracy': float(test_metrics['accuracy']),
        'test_precision': float(test_metrics['precision']),
        'test_recall': float(test_metrics['recall']),
        'test_f1': float(test_metrics['f1']),
        'test_auc': float(test_metrics['auc']),
        'confusion_matrix': cm.tolist(),
        'patch_size': args.patch_size,
        'max_patches_per_roi': args.max_patches_per_roi
    }

    results_path = os.path.join(sub_dirs['results'], 'test_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"\n✓ 测试结果已保存: {results_path}")

    # 详细评估（含attention）
    print("\n" + "="*60)
    print("保存详细预测结果和attention权重...")
    print("="*60)

    test_df, test_results = evaluate_test_with_attention(
        model, test_dataset, device, sub_dirs['results']
    )

    print("\n" + "="*60)
    print(f"测试完成！所有结果保存在: {exp_dir}")
    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Patch-based颈动脉斑块分类测试')

    # 数据参数
    parser.add_argument('--root-dir', type=str,
                       default='/media/data/wjf/data/Carotid_artery')
    parser.add_argument('--mask-dir', type=str,
                       default='/media/data/wjf/data/mask')
    parser.add_argument('--label-excel', type=str,
                       default='/media/data/wjf/data/label_all_250+30+100.xlsx')
    parser.add_argument('--output-dir', type=str, default='./output_patch_based')

    # 模型参数
    parser.add_argument('--model-path', type=str, required=True,
                       help='训练好的模型权重路径')

    # Patch参数（需要与训练时一致）
    parser.add_argument('--patch-size', type=int, default=24,
                       help='patch大小')
    parser.add_argument('--max-patches-per-roi', type=int, default=12,
                       help='每个ROI最多提取的patch数')
    parser.add_argument('--overlap-ratio', type=float, default=0.5,
                       help='patch重叠比例')

    # 模型参数（需要与训练时一致）
    parser.add_argument('--feature-dim', type=int, default=128,
                       help='特征维度')

    # 测试参数
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)

    # 数据集参数（需要与训练时一致）
    parser.add_argument('--depth', type=int, default=100)
    parser.add_argument('--train-ratio', type=float, default=0.8)
    parser.add_argument('--val-ratio', type=float, default=0.1)

    # 其他
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--no-cuda', action='store_true')

    args = parser.parse_args()
    test(args)


if __name__ == '__main__':
    main()
