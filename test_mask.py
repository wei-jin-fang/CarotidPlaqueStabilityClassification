"""
测试脚本：加载训练好的Mask引导模型进行测试
"""
import os
import argparse
import json
import random
import numpy as np
import pandas as pd
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

from utils.dataset_carotid_mask import CarotidPlaqueDatasetWithMask
from models.resnet3D_mask import create_mask_guided_classifier


def set_seed(seed=42):
    """设置随机种子保证可复现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"✓ 随机种子设置为: {seed}")


def create_test_output_folder(base_dir="./test_output_mask"):
    """创建测试输出文件夹"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    test_dir = os.path.join(base_dir, f"test_mask_{timestamp}")

    os.makedirs(test_dir, exist_ok=True)

    print(f"\n✓ 创建测试输出文件夹: {test_dir}\n")
    return test_dir


def evaluate_test_set(model, dataloader, device):
    """评估测试集"""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for volumes, masks, labels in dataloader:
            volumes = volumes.to(device)
            masks = masks.to(device)
            labels = labels.to(device)

            outputs = model(volumes, masks)
            probs = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='binary', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='binary', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='binary', zero_division=0)

    try:
        auc = roc_auc_score(all_labels, all_probs)
    except:
        auc = 0.0

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'predictions': all_preds,
        'labels': all_labels,
        'probabilities': all_probs
    }


def evaluate_test_set_detailed(model, test_dataset, device, save_dir):
    """详细评估测试集"""
    model.eval()
    results = []

    print("\n正在评估测试集每个样本...")

    with torch.no_grad():
        for idx in range(len(test_dataset)):
            volume, mask, label = test_dataset[idx]
            volume = volume.unsqueeze(0).to(device)
            mask = mask.unsqueeze(0).to(device)

            output = model(volume, mask)
            probs = torch.softmax(output, dim=1)
            pred = output.argmax(dim=1).item()

            prob_class0 = probs[0, 0].item()
            prob_class1 = probs[0, 1].item()

            person_info = test_dataset.persons[idx]
            patient_name = person_info['name']
            true_label = label.item() if isinstance(label, torch.Tensor) else label

            is_correct = (pred == true_label)

            results.append({
                'patient_name': patient_name,
                'true_label': true_label,
                'predicted_label': pred,
                'is_correct': is_correct,
                'prob_class_0': prob_class0,
                'prob_class_1': prob_class1,
                'confidence': max(prob_class0, prob_class1),
                'mask_path': person_info['mask_path']
            })

            if (idx + 1) % 5 == 0:
                print(f"  已评估 {idx + 1}/{len(test_dataset)} 个样本")

    df = pd.DataFrame(results)
    accuracy = df['is_correct'].sum() / len(df)

    print(f"\n✓ 测试集详细评估完成")
    print(f"  总样本数: {len(df)}")
    print(f"  正确预测: {df['is_correct'].sum()}")
    print(f"  错误预测: {(~df['is_correct']).sum()}")
    print(f"  准确率: {accuracy:.4f}")

    csv_path = os.path.join(save_dir, 'test_predictions_detailed.csv')
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"\n✓ 详细预测结果已保存: {csv_path}")

    wrong_predictions = df[~df['is_correct']]
    if len(wrong_predictions) > 0:
        wrong_csv_path = os.path.join(save_dir, 'test_predictions_errors.csv')
        wrong_predictions.to_csv(wrong_csv_path, index=False, encoding='utf-8-sig')
        print(f"✓ 错误预测案例已保存: {wrong_csv_path}")

        print(f"\n错误预测详情:")
        for _, row in wrong_predictions.iterrows():
            print(f"  - {row['patient_name']}: 真实={row['true_label']}, "
                  f"预测={row['predicted_label']}, 置信度={row['confidence']:.3f}")

    return df


def test(args):
    """主测试函数"""
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    print(f"使用设备: {device}")

    set_seed(args.seed)

    test_dir = create_test_output_folder(args.output_dir)

    config_path = os.path.join(test_dir, 'test_config.json')
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=4)

    # 数据变换
    test_transform = transforms.Compose([
        transforms.Resize((244, 244)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.35], std=[0.25])
    ])

    # 加载数据集
    print("\n" + "="*60)
    print("加载数据集（含Mask）...")
    print("="*60)

    full_dataset = CarotidPlaqueDatasetWithMask(
        root_dir=args.root_dir,
        mask_dir=args.mask_dir,
        label_excel=args.label_excel,
        transform=test_transform,
        keep_middle_n=args.depth,
        min_imgs_required=args.depth,
        verbose=True
    )

    # 划分数据集
    total_size = len(full_dataset)
    indices = list(range(total_size))
    random.shuffle(indices)

    train_size = int(args.train_ratio * total_size)
    val_size = int(args.val_ratio * total_size)

    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size+val_size]
    test_indices = indices[train_size+val_size:]

    print(f"\n数据集划分:")
    print(f"  训练集: {len(train_indices)} 样本")
    print(f"  验证集: {len(val_indices)} 样本")
    print(f"  测试集: {len(test_indices)} 样本")

    # 创建测试数据集
    test_dataset = CarotidPlaqueDatasetWithMask(
        root_dir=args.root_dir,
        mask_dir=args.mask_dir,
        label_excel=args.label_excel,
        transform=test_transform,
        keep_middle_n=args.depth,
        min_imgs_required=args.depth,
        verbose=False
    )
    test_dataset.persons = [full_dataset.persons[i] for i in test_indices]

    # 保存测试集样本列表
    test_samples = []
    for person in test_dataset.persons:
        test_samples.append({
            'patient_name': person['name'],
            'label': person['label'],
            'mask_path': person['mask_path']
        })
    test_df = pd.DataFrame(test_samples)
    test_samples_path = os.path.join(test_dir, 'test_samples.csv')
    test_df.to_csv(test_samples_path, index=False, encoding='utf-8-sig')
    print(f"\n✓ 测试集样本列表已保存: {test_samples_path}")

    # 创建DataLoader
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers, pin_memory=True
    )

    # 创建模型
    print("\n" + "="*60)
    print("加载Mask引导模型...")
    print("="*60)

    model = create_mask_guided_classifier(
        num_classes=2,
        pretrained_path=None,
        freeze_backbone=False
    )

    # 加载训练好的模型权重
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"模型文件不存在: {args.model_path}")

    print(f"加载模型权重: {args.model_path}")
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model = model.to(device)
    print("✓ 模型加载成功")

    # 评估测试集
    print("\n" + "="*60)
    print("测试集评估...")
    print("="*60)

    test_metrics = evaluate_test_set(model, test_loader, device)

    print(f"\n测试集整体结果:")
    print(f"  准确率:   {test_metrics['accuracy']:.4f}")
    print(f"  精确率:   {test_metrics['precision']:.4f}")
    print(f"  召回率:   {test_metrics['recall']:.4f}")
    print(f"  F1分数:   {test_metrics['f1']:.4f}")
    print(f"  AUC:      {test_metrics['auc']:.4f}")

    cm = confusion_matrix(test_metrics['labels'], test_metrics['predictions'])
    print(f"\n混淆矩阵:")
    print(cm)

    print(f"\n分类报告:")
    print(classification_report(test_metrics['labels'], test_metrics['predictions'],
                                target_names=['Class 0', 'Class 1']))

    # 保存测试结果
    results = {
        'test_accuracy': float(test_metrics['accuracy']),
        'test_precision': float(test_metrics['precision']),
        'test_recall': float(test_metrics['recall']),
        'test_f1': float(test_metrics['f1']),
        'test_auc': float(test_metrics['auc']),
        'confusion_matrix': cm.tolist(),
        'model_path': args.model_path
    }

    results_path = os.path.join(test_dir, 'test_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"\n✓ 测试结果已保存: {results_path}")

    # 详细评估
    print("\n" + "="*60)
    print("保存测试集详细预测结果...")
    print("="*60)

    test_predictions_df = evaluate_test_set_detailed(
        model, test_dataset, device, test_dir
    )

    print("\n" + "="*60)
    print("测试完成！")
    print(f"所有结果保存在: {test_dir}")
    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(description='颈动脉斑块分类测试 - Mask引导')

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
    parser.add_argument('--output-dir', type=str, default='./test_output_mask',
                       help='测试输出目录')

    # 模型参数
    parser.add_argument('--model-path', type=str, required=True,
                       help='训练好的模型权重路径（必需）')

    # 测试参数
    parser.add_argument('--batch-size', type=int, default=4, help='批次大小')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')

    # 数据集参数
    parser.add_argument('--depth', type=int, default=100,
                       help='3D volume深度')
    parser.add_argument('--train-ratio', type=float, default=0.8,
                       help='训练集比例')
    parser.add_argument('--val-ratio', type=float, default=0.1,
                       help='验证集比例')

    # 其他参数
    parser.add_argument('--num-workers', type=int, default=4, help='数据加载线程数')
    parser.add_argument('--no-cuda', action='store_true', help='不使用GPU')

    args = parser.parse_args()
    test(args)


if __name__ == '__main__':
    main()
