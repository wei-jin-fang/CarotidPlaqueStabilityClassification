"""
基于Patch的颈动脉斑块分类测试脚本(融合Excel临床特征)

功能:
1. 加载训练好的模型
2. 对测试集进行评估
3. 保存详细的预测结果和attention权重
4. 生成混淆矩阵和评估指标
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import argparse
import json
import pickle
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

from utils.dataset_patch_feature import PatchFeatureCarotidDataset
from models.patch_feature_classifier import create_patch_feature_classifier


def set_seed(seed=42):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"✓ 随机种子设置为: {seed}")


def collate_fn_patch_feature(batch):
    """
    自定义collate函数,处理不同数量的patch和Excel特征
    """
    patches_list = []
    positions_list = []
    features_list = []
    labels_list = []

    for patches, positions, features, label in batch:
        patches_list.append(patches)
        positions_list.append(positions)
        features_list.append(features)
        labels_list.append(label)

    features = torch.stack(features_list)
    labels = torch.stack(labels_list)

    return patches_list, positions_list, features, labels


def pad_patches_batch(patches_list, max_patches=None):
    """
    将不同数量的patch padding到相同长度
    """
    batch_size = len(patches_list)

    if max_patches is None:
        max_patches = max(p.size(0) for p in patches_list)

    C, H, W = patches_list[0].shape[1:]

    padded = torch.zeros(batch_size, max_patches, C, H, W)
    masks = torch.zeros(batch_size, max_patches)

    for i, patches in enumerate(patches_list):
        n = patches.size(0)
        n = min(n, max_patches)
        padded[i, :n] = patches[:n]
        masks[i, :n] = 1

    return padded, masks


def evaluate_batch(model, dataloader, criterion, device):
    """批量评估模型"""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for patches_list, positions_list, features, labels in dataloader:
            patches_padded, masks = pad_patches_batch(patches_list)
            patches_padded = patches_padded.to(device)
            masks = masks.to(device)
            features = features.to(device)
            labels = labels.to(device)

            logits, attn_weights = model(patches_padded, features, mask=masks, return_attention=True)
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


def evaluate_with_attention(model, dataset, device, save_dir):
    """
    详细评估测试集,保存每个样本的预测结果和attention权重
    """
    model.eval()
    results = []

    print("\n正在评估测试集...")

    with torch.no_grad():
        for idx in range(len(dataset)):
            patches, positions, features, label = dataset[idx]

            patches = patches.unsqueeze(0).to(device)
            features = features.unsqueeze(0).to(device)

            logits, attn_weights = model(patches, features, return_attention=True)
            probs = torch.softmax(logits, dim=1)
            pred = logits.argmax(dim=1).item()

            prob_class0 = probs[0, 0].item()
            prob_class1 = probs[0, 1].item()

            person_info = dataset.persons[idx]
            patient_name = person_info['name']
            true_label = label.item()
            is_correct = (pred == true_label)

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
                'positions': positions
            })

            if (idx + 1) % 10 == 0:
                print(f"  已评估 {idx + 1}/{len(dataset)} 个样本")

    # 保存详细结果(包含attention和positions)
    results_path = os.path.join(save_dir, 'test_predictions_with_attention.pkl')
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"\n✓ 详细结果(含attention)已保存: {results_path}")

    # 保存CSV结果
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

    return df, results


def print_evaluation_results(metrics, labels, predictions, save_path=None):
    """打印和保存评估结果"""
    lines = []
    lines.append("=" * 60)
    lines.append("测试集评估结果")
    lines.append("=" * 60)
    lines.append(f"准确率 (Accuracy):  {metrics['accuracy']:.4f}")
    lines.append(f"精确率 (Precision): {metrics['precision']:.4f}")
    lines.append(f"召回率 (Recall):    {metrics['recall']:.4f}")
    lines.append(f"F1分数 (F1-Score):  {metrics['f1']:.4f}")
    lines.append(f"AUC:                {metrics['auc']:.4f}")
    lines.append(f"损失 (Loss):        {metrics['loss']:.4f}")

    lines.append("\n" + "-" * 60)
    lines.append("混淆矩阵:")
    lines.append("-" * 60)
    cm = confusion_matrix(labels, predictions)
    lines.append(f"              预测负类  预测正类")
    lines.append(f"实际负类      {cm[0][0]:8d}  {cm[0][1]:8d}")
    lines.append(f"实际正类      {cm[1][0]:8d}  {cm[1][1]:8d}")

    lines.append("\n" + "-" * 60)
    lines.append("分类报告:")
    lines.append("-" * 60)
    report = classification_report(labels, predictions, target_names=['稳定(0)', '不稳定(1)'])
    lines.append(report)
    lines.append("=" * 60)

    result_text = "\n".join(lines)
    print(result_text)

    if save_path:
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(result_text)
        print(f"\n✓ 评估结果已保存: {save_path}")

    return cm


def test(args):
    """主测试函数"""
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    print(f"使用设备: {device}")

    # 设置随机种子(与训练时一致)
    set_seed(args.seed)

    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"test_patch_feature_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"输出目录: {output_dir}")

    # 数据变换
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.35], std=[0.25])
    ])

    # 加载完整数据集
    print("\n" + "=" * 60)
    print("加载数据集...")
    print("=" * 60)

    full_dataset = PatchFeatureCarotidDataset(
        root_dir=args.root_dir,
        mask_dir=args.mask_dir,
        label_excel=args.label_excel,
        feature_excel=args.feature_excel,
        patch_size=args.patch_size,
        max_patches_per_roi=args.max_patches_per_roi,
        overlap_ratio=args.overlap_ratio,
        keep_middle_n=args.depth,
        min_imgs_required=args.depth,
        transform=transform,
        verbose=True
    )

    # 使用与训练相同的方式划分数据集
    total_size = len(full_dataset)
    indices = list(range(total_size))
    random.shuffle(indices)

    train_size = int(args.train_ratio * total_size)
    val_size = int(args.val_ratio * total_size)

    test_indices = indices[train_size + val_size:]

    print(f"\n数据集划分 (seed={args.seed}):")
    print(f"  总样本数: {total_size}")
    print(f"  训练集: {train_size} 样本")
    print(f"  验证集: {val_size} 样本")
    print(f"  测试集: {len(test_indices)} 样本")

    # 获取scaler
    scaler = full_dataset.get_scaler()

    # 创建测试子数据集
    test_dataset = PatchFeatureCarotidDataset(
        root_dir=args.root_dir,
        mask_dir=args.mask_dir,
        label_excel=args.label_excel,
        feature_excel=args.feature_excel,
        patch_size=args.patch_size,
        max_patches_per_roi=args.max_patches_per_roi,
        overlap_ratio=args.overlap_ratio,
        keep_middle_n=args.depth,
        min_imgs_required=args.depth,
        transform=transform,
        verbose=False,
        scaler=scaler
    )
    test_dataset.persons = [full_dataset.persons[i] for i in test_indices]

    print(f"\n测试集样本数: {len(test_dataset)}")

    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers,
        collate_fn=collate_fn_patch_feature, pin_memory=True
    )

    # 创建模型
    print("\n" + "=" * 60)
    print("创建模型...")
    print("=" * 60)

    model = create_patch_feature_classifier(
        patch_size=args.patch_size,
        num_classes=2,
        image_feature_dim=args.image_feature_dim,
        excel_feature_dim=test_dataset.get_feature_dim(),
        excel_hidden_dim=args.excel_hidden_dim,
        vit_depth=args.vit_depth,
        vit_heads=args.vit_heads,
        vit_sub_patch_size=args.vit_sub_patch_size,
        fusion_hidden_dim=args.fusion_hidden_dim
    )

    # 加载模型权重
    if os.path.exists(args.model_path):
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        print(f"✓ 已加载模型权重: {args.model_path}")
    else:
        raise FileNotFoundError(f"模型权重文件不存在: {args.model_path}")

    model = model.to(device)
    model.eval()

    # 损失函数
    criterion = nn.CrossEntropyLoss()

    # 批量评估
    print("\n" + "=" * 60)
    print("开始测试评估...")
    print("=" * 60)

    metrics = evaluate_batch(model, test_loader, criterion, device)

    # 打印结果
    report_path = os.path.join(output_dir, 'evaluation_report.txt')
    cm = print_evaluation_results(
        metrics,
        metrics['labels'],
        metrics['predictions'],
        save_path=report_path
    )

    # 详细评估(包含attention)
    if args.save_attention:
        print("\n" + "=" * 60)
        print("保存详细预测结果和attention权重...")
        print("=" * 60)
        test_df, test_results = evaluate_with_attention(
            model, test_dataset, device, output_dir
        )

    # 保存汇总结果
    summary = {
        'model_path': args.model_path,
        'test_samples': len(test_dataset),
        'accuracy': float(metrics['accuracy']),
        'precision': float(metrics['precision']),
        'recall': float(metrics['recall']),
        'f1': float(metrics['f1']),
        'auc': float(metrics['auc']),
        'loss': float(metrics['loss']),
        'confusion_matrix': cm.tolist(),
        'config': {
            'patch_size': args.patch_size,
            'max_patches_per_roi': args.max_patches_per_roi,
            'image_feature_dim': args.image_feature_dim,
            'excel_hidden_dim': args.excel_hidden_dim,
            'fusion_hidden_dim': args.fusion_hidden_dim
        }
    }

    summary_path = os.path.join(output_dir, 'test_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=4)
    print(f"\n✓ 测试汇总已保存: {summary_path}")

    print("\n" + "=" * 60)
    print(f"测试完成! 所有结果保存在: {output_dir}")
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Patch-Feature融合颈动脉斑块分类测试')

    # 模型路径
    parser.add_argument('--model-path', type=str, required=True,
                       help='模型权重文件路径')

    # 数据参数
    parser.add_argument('--root-dir', type=str,
                       default='/media/data/wjf/data/Carotid_artery')
    parser.add_argument('--mask-dir', type=str,
                       default='/media/data/wjf/data/mask')
    parser.add_argument('--label-excel', type=str,
                       default='/media/data/wjf/data/label_all_250+30+100.xlsx')
    parser.add_argument('--feature-excel', type=str,
                       default='/home/jinfang/project/new_CarotidPlaqueStabilityClassification/Test/feature_complete.xlsx')
    parser.add_argument('--output-dir', type=str, default='./output_patch_feature_test')

    # Patch参数
    parser.add_argument('--patch-size', type=int, default=24,
                       help='patch大小')
    parser.add_argument('--max-patches-per-roi', type=int, default=12,
                       help='每个ROI最多提取的patch数')
    parser.add_argument('--overlap-ratio', type=float, default=0.5,
                       help='patch重叠比例')

    # 模型参数
    parser.add_argument('--image-feature-dim', type=int, default=128,
                       help='图像特征维度')
    parser.add_argument('--excel-hidden-dim', type=int, default=128,
                       help='Excel特征提取后的隐藏层维度')
    parser.add_argument('--fusion-hidden-dim', type=int, default=256,
                       help='融合后的隐藏层维度')
    parser.add_argument('--vit-depth', type=int, default=1,
                       help='ViT Transformer层数')
    parser.add_argument('--vit-heads', type=int, default=1,
                       help='ViT多头注意力头数')
    parser.add_argument('--vit-sub-patch-size', type=int, default=4,
                       help='ViT sub-patch大小')

    # 数据集参数
    parser.add_argument('--depth', type=int, default=100)
    parser.add_argument('--train-ratio', type=float, default=0.8)
    parser.add_argument('--val-ratio', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)

    # 测试参数
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--no-cuda', action='store_true')
    parser.add_argument('--save-attention', action='store_true', default=True,
                       help='是否保存attention权重和详细预测结果')

    args = parser.parse_args()
    test(args)


if __name__ == '__main__':
    main()
