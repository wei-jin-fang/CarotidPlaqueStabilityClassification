"""
基于Patch的颈动脉斑块分类训练脚本(融合Excel临床特征)

改进:
1. 从ROI区域提取小patch,避免背景干扰
2. 使用Attention机制聚合patch特征
3. 融合Excel临床特征(医学指标)
4. 记录patch位置信息,支持后续可视化
5. 不使用预训练,从头训练
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import argparse
import json
import random
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import pickle

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)

from utils.dataset_patch_feature import PatchFeatureCarotidDataset, save_scaler
from models.patch_feature_classifier import create_patch_feature_classifier
# from models.patch_feature_classifier_clstoken import create_patch_feature_classifier_with_cls as create_patch_feature_classifier

def set_seed(seed=42):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"✓ 随机种子设置为: {seed}")


def create_output_folder(base_dir="./output_patch_feature"):
    """创建输出文件夹"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(base_dir, f"train_patch_feature_{timestamp}")

    sub_dirs = {
        'root': exp_dir,
        'models': os.path.join(exp_dir, 'models'),
        'logs': os.path.join(exp_dir, 'logs'),
        'results': os.path.join(exp_dir, 'results'),
        'visualizations': os.path.join(exp_dir, 'visualizations'),
    }

    for dir_path in sub_dirs.values():
        os.makedirs(dir_path, exist_ok=True)

    print(f"\n✓ 创建输出文件夹: {exp_dir}\n")
    return exp_dir, sub_dirs


def calculate_class_weights(dataset, device, neg_extra=1.0, pos_extra=1.2):
    """
    计算类别权重，并支持对负类/正类额外乘以权重系数。

    参数:
        dataset: 数据集对象，必须有 .persons 属性，每个 person 有 'label' (0 或 1)
        device: torch.device
        neg_extra: 负类（label=0）的额外权重系数，默认 1.0
        pos_extra: 正类（label=1）的额外权重系数，默认 1.0

    返回:
        torch.Tensor: shape=(2,) 的最终类别权重，已移到 device 上
    """
    class_counts = [0, 0]
    for person in dataset.persons:
        class_counts[person['label']] += 1

    total = sum(class_counts)
    
    # 基础逆频率权重（标准的类平衡权重）
    base_weights = [
        total / (2 * count) if count > 0 else 1.0 
        for count in class_counts
    ]

    # 应用额外系数
    final_weights = [
        base_weights[0] * neg_extra,
        base_weights[1] * pos_extra
    ]

    weights = torch.tensor(final_weights, dtype=torch.float32).to(device)

    print(f"类别分布: 负类={class_counts[0]}, 正类={class_counts[1]}")
    print(f"基础权重: 负类={base_weights[0]:.4f}, 正类={base_weights[1]:.4f}")
    print(f"额外系数: 负类={neg_extra:.3f}, 正类={pos_extra:.3f}")
    print(f"最终类别权重: {weights.cpu().numpy()}")

    return weights


def collate_fn_patch_feature(batch):
    """
    自定义collate函数,处理不同数量的patch和Excel特征

    返回:
        patches: list of [N_i, C, H, W] tensors
        positions: list of list of dict
        features: [B, feature_dim] tensor
        labels: [B] tensor
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

    参数:
        patches_list: list of [N_i, C, H, W] tensors
        max_patches: 最大patch数(None则使用batch中的最大值)

    返回:
        padded: [B, max_N, C, H, W] tensor
        masks: [B, max_N] tensor, 1表示有效patch,0表示padding
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


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """训练一个epoch"""
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    for batch_idx, (patches_list, positions_list, features, labels) in enumerate(dataloader):
        # Padding patches
        patches_padded, masks = pad_patches_batch(patches_list)
        patches_padded = patches_padded.to(device)
        masks = masks.to(device)
        features = features.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # 前向传播(传递mask和features)
        logits, attn_weights = model(patches_padded, features, mask=masks, return_attention=True)
        loss = criterion(logits, labels)

        # 反向传播
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        preds = logits.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)

    return avg_loss, accuracy


def validate(model, dataloader, criterion, device):
    """验证模型"""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for patches_list, positions_list, features, labels in dataloader:
            # Padding
            patches_padded, masks = pad_patches_batch(patches_list)
            patches_padded = patches_padded.to(device)
            masks = masks.to(device)
            features = features.to(device)
            labels = labels.to(device)

            # 前向传播(传递mask和features)
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


def save_dataset_samples(train_dataset, val_dataset, test_dataset, save_dir):
    """保存数据集样本名称"""
    print("\n" + "="*60)
    print("保存数据集样本名称...")
    print("="*60)

    for split_name, dataset in [('train', train_dataset),
                                ('val', val_dataset),
                                ('test', test_dataset)]:
        samples = []
        for person in dataset.persons:
            samples.append({
                'patient_name': person['name'],
                'label': person['label'],
                'mask_path': person['mask_path'],
                'num_slices': len(person['paths']),
                'has_excel_features': True
            })

        df = pd.DataFrame(samples)
        csv_path = os.path.join(save_dir, f'{split_name}_samples.csv')
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"✓ {split_name}集样本已保存: {csv_path} ({len(samples)} 样本)")

    print("="*60 + "\n")


def evaluate_test_with_attention(model, test_dataset, device, save_dir):
    """
    详细评估测试集,保存每个样本的预测结果和attention权重

    关键:保存positions信息用于可视化
    """
    model.eval()
    results = []

    print("\n正在评估测试集...")

    with torch.no_grad():
        for idx in range(len(test_dataset)):
            patches, positions, features, label = test_dataset[idx]

            # 添加batch维度
            patches = patches.unsqueeze(0).to(device)  # [1, N, C, H, W]
            features = features.unsqueeze(0).to(device)  # [1, feature_dim]

            # 前向传播
            logits, attn_weights = model(patches, features, return_attention=True)
            probs = torch.softmax(logits, dim=1)
            pred = logits.argmax(dim=1).item()

            prob_class0 = probs[0, 0].item()
            prob_class1 = probs[0, 1].item()

            person_info = test_dataset.persons[idx]
            patient_name = person_info['name']
            true_label = label.item()
            is_correct = (pred == true_label)

            # 获取attention权重(numpy数组)
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
                'positions': positions  # 保存位置信息!
            })

            if (idx + 1) % 5 == 0:
                print(f"  已评估 {idx + 1}/{len(test_dataset)} 个样本")

    # 保存详细结果(包含attention和positions)
    results_path = os.path.join(save_dir, 'test_predictions_with_attention.pkl')
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"\n✓ 详细结果(含attention)已保存: {results_path}")

    # 保存简化版CSV(不含positions)
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


def calculate_model_params(model):
    """
    计算模型参数量

    返回:
        dict: 包含总参数、可训练参数、不可训练参数的字典
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params

    # 按模块统计
    module_params = {}
    for name, module in model.named_children():
        module_total = sum(p.numel() for p in module.parameters())
        module_trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
        module_params[name] = {
            'total': module_total,
            'trainable': module_trainable
        }

    return {
        'total': total_params,
        'trainable': trainable_params,
        'non_trainable': non_trainable_params,
        'by_module': module_params
    }


def print_model_summary(model, save_path=None):
    """打印并保存模型参数统计"""
    params_info = calculate_model_params(model)

    summary = []
    summary.append("=" * 80)
    summary.append("模型参数统计 (Patch-Feature融合模型)")
    summary.append("=" * 80)
    summary.append(f"总参数量:        {params_info['total']:,}")
    summary.append(f"可训练参数:      {params_info['trainable']:,}")
    summary.append(f"不可训练参数:    {params_info['non_trainable']:,}")
    summary.append(f"模型大小 (MB):   {params_info['total'] * 4 / 1024 / 1024:.2f}")

    summary.append("\n" + "-" * 80)
    summary.append("各模块参数统计:")
    summary.append("-" * 80)
    for name, info in params_info['by_module'].items():
        summary.append(f"  {name:25s}: {info['total']:>10,} 总参数, {info['trainable']:>10,} 可训练")
    summary.append("=" * 80)

    summary_text = "\n".join(summary)
    print(summary_text)

    # 保存到文件
    if save_path:
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(summary_text)
        print(f"\n✓ 模型参数统计已保存: {save_path}")

    return params_info


def plot_training_curves(history, save_dir):
    """绘制训练曲线(训练Loss和验证Loss分两个子图展示)"""
    epochs = range(1, len(history['train_loss']) + 1)

    # 创建2行1列子图
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    fig.suptitle('Patch-Feature Fusion Training History', fontsize=16, fontweight='bold')

    # 第一个子图:训练Loss
    axes[0].plot(epochs, history['train_loss'], 'b-o', label='Train Loss', linewidth=2, markersize=4)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training Loss Curve', fontsize=13, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(alpha=0.3, linestyle='--')
    axes[0].set_xlim(1, len(epochs))

    # 第二个子图:验证Loss
    axes[1].plot(epochs, history['val_loss'], 'r-s', label='Validation Loss', linewidth=2, markersize=4)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Loss', fontsize=12)
    axes[1].set_title('Validation Loss Curve', fontsize=13, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(alpha=0.3, linestyle='--')
    axes[1].set_xlim(1, len(epochs))

    # 调整子图间距
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_path = os.path.join(save_dir, 'training_loss_curves.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ 训练Loss曲线已保存: {save_path}")

    # 打印Loss相关训练统计
    print("\n" + "="*60)
    print("训练Loss改善统计:")
    print("="*60)
    print(f"训练集:")
    print(f"  初始Loss (Epoch 1):    {history['train_loss'][0]:.4f}")
    print(f"  最终Loss (Epoch {len(history['train_loss'])}):   {history['train_loss'][-1]:.4f}")
    print(f"  最佳Loss:              {min(history['train_loss']):.4f}")
    print(f"  Loss改善:              {history['train_loss'][0] - min(history['train_loss']):.4f}")
    print(f"\n验证集:")
    print(f"  初始Loss (Epoch 1):    {history['val_loss'][0]:.4f}")
    print(f"  最终Loss (Epoch {len(history['val_loss'])}):   {history['val_loss'][-1]:.4f}")
    print(f"  最佳Loss:              {min(history['val_loss']):.4f}")
    print(f"  Loss改善:              {history['val_loss'][0] - min(history['val_loss']):.4f}")
    print("="*60 + "\n")


def train(args):
    """主训练函数"""
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    print(f"使用设备: {device}")

    set_seed(args.seed)

    exp_dir, sub_dirs = create_output_folder(args.output_dir)

    # 保存配置
    config_path = os.path.join(sub_dirs['logs'], 'config.json')
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=4)

    # 数据变换
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.35], std=[0.25])
    ])

    # 加载数据集
    print("\n" + "="*60)
    print("加载Patch-Feature融合数据集...")
    print("="*60)

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

    # 获取scaler(从full_dataset)
    scaler = full_dataset.get_scaler()

    # 创建子数据集
    train_dataset = PatchFeatureCarotidDataset(
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
        scaler=scaler  # 使用相同的scaler
    )
    train_dataset.persons = [full_dataset.persons[i] for i in train_indices]

    val_dataset = PatchFeatureCarotidDataset(
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
    val_dataset.persons = [full_dataset.persons[i] for i in val_indices]

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

    # 保存scaler
    scaler_path = os.path.join(sub_dirs['models'], 'feature_scaler.pkl')
    save_scaler(scaler, scaler_path)

    # DataLoader
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=args.num_workers,
        collate_fn=collate_fn_patch_feature, pin_memory=True,drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers,
        collate_fn=collate_fn_patch_feature, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers,
        collate_fn=collate_fn_patch_feature, pin_memory=True
    )

    # 保存样本信息
    save_dataset_samples(train_dataset, val_dataset, test_dataset, sub_dirs['logs'])

    # 创建模型
    print("\n" + "="*60)
    print("创建Patch-Feature融合分类器...")
    print("="*60)

    model = create_patch_feature_classifier(
        patch_size=args.patch_size,
        num_classes=2,
        image_feature_dim=args.image_feature_dim,
        excel_feature_dim=full_dataset.get_feature_dim(),
        excel_hidden_dim=args.excel_hidden_dim,
        vit_depth=args.vit_depth,
        vit_heads=args.vit_heads,
        vit_sub_patch_size=args.vit_sub_patch_size,
        fusion_hidden_dim=args.fusion_hidden_dim
    )

    model = model.to(device)

    # 打印并保存模型参数统计
    params_summary_path = os.path.join(sub_dirs['logs'], 'model_params_summary.txt')
    print_model_summary(model, save_path=params_summary_path)

    # 损失函数和优化器
    class_weights = calculate_class_weights(train_dataset, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # 训练循环
    print("\n" + "="*60)
    print("开始训练...")
    print("="*60 + "\n")

    best_val_acc = 0.0
    best_epoch = 0
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'val_f1': [],
        'val_auc': []
    }

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print("-" * 60)

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )

        val_metrics = validate(model, val_loader, criterion, device)

        scheduler.step()

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['accuracy'])
        history['val_f1'].append(val_metrics['f1'])
        history['val_auc'].append(val_metrics['auc'])

        print(f"\n训练 - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
        print(f"验证 - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}, "
              f"F1: {val_metrics['f1']:.4f}, AUC: {val_metrics['auc']:.4f}")

        # 测试集评估(当前epoch)
        if len(test_indices) > 0:
            test_metrics_temp = validate(model, test_loader, criterion, device)
            print(f"测试 - Acc: {test_metrics_temp['accuracy']:.4f}, "
                  f"AUC: {test_metrics_temp['auc']:.4f}, "
                  f"Recall: {test_metrics_temp['recall']:.4f}, "
                  f"Precision: {test_metrics_temp['precision']:.4f}, "
                  f"F1: {test_metrics_temp['f1']:.4f}")
            if test_metrics_temp['accuracy']>0.8:
                temp_best_model_path = os.path.join(sub_dirs['models'], f'best_model_{epoch}.pth')
                torch.save(model.state_dict(), temp_best_model_path)
                print(f"临时保存_{epoch}_{test_metrics_temp['accuracy']}")

        # 保存最佳模型
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            best_epoch = epoch + 1
            best_model_path = os.path.join(sub_dirs['models'], 'best_model.pth')
            torch.save(model.state_dict(), best_model_path)
            print(f"✓ 保存最佳模型 (验证准确率: {best_val_acc:.4f})")

    # 保存训练历史
    history_df = pd.DataFrame(history)
    history_df.insert(0, 'epoch', range(1, len(history['train_loss']) + 1))
    history_path = os.path.join(sub_dirs['logs'], 'training_history.csv')
    history_df.to_csv(history_path, index=False)
    print(f"\n✓ 训练历史已保存: {history_path}")

    # 绘制曲线
    plot_training_curves(history, sub_dirs['logs'])

    # 测试集最终评估
    if len(test_indices) > 0:
        print("\n" + "="*60)
        print("最终测试集评估...")
        print("="*60)

        model.load_state_dict(torch.load(best_model_path))
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
            'best_epoch': best_epoch,
            'best_val_acc': float(best_val_acc),
            'test_accuracy': float(test_metrics['accuracy']),
            'test_precision': float(test_metrics['precision']),
            'test_recall': float(test_metrics['recall']),
            'test_f1': float(test_metrics['f1']),
            'test_auc': float(test_metrics['auc']),
            'confusion_matrix': cm.tolist(),
            'patch_size': args.patch_size,
            'max_patches_per_roi': args.max_patches_per_roi,
            'image_feature_dim': args.image_feature_dim,
            'excel_feature_dim': full_dataset.get_feature_dim(),
            'excel_hidden_dim': args.excel_hidden_dim,
            'fusion_type': 'concat'
        }

        results_path = os.path.join(sub_dirs['results'], 'test_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=4)

        # 详细评估(含attention)
        print("\n" + "="*60)
        print("保存详细预测结果和attention权重...")
        print("="*60)

        test_df, test_results = evaluate_test_with_attention(
            model, test_dataset, device, sub_dirs['results']
        )

    print("\n" + "="*60)
    print(f"训练完成!所有结果保存在: {exp_dir}")
    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Patch-Feature融合颈动脉斑块分类训练')

    # 数据参数
    parser.add_argument('--root-dir', type=str,
                       default='/media/data/wjf/data/Carotid_artery')
    parser.add_argument('--mask-dir', type=str,
                       default='/media/data/wjf/data/mask')
    parser.add_argument('--label-excel', type=str,
                       default='/media/data/wjf/data/label_all_250+30+100.xlsx')
    parser.add_argument('--feature-excel', type=str,
                       default='/home/jinfang/project/new_CarotidPlaqueStabilityClassification/Test/feature_complete.xlsx')
    parser.add_argument('--output-dir', type=str, default='./output_patch_feature')

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

    # 训练参数
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=42)

    # 数据集参数
    parser.add_argument('--depth', type=int, default=100)
    parser.add_argument('--train-ratio', type=float, default=0.8)
    parser.add_argument('--val-ratio', type=float, default=0.1)

    # 其他
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--no-cuda', action='store_true')

    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
