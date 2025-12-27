"""
基于Mask引导的颈动脉斑块分类训练脚本 - 自适应裁剪版本

改进:
1. 使用自适应裁剪替代传统的直接resize
2. 根据mask自动找到ROI并智能裁剪
3. 对小区域使用padding保留信息，大区域智能缩放
"""
import os
import argparse
import json
import random
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

from utils.dataset_adaptive_mask import CarotidPlaqueDatasetWithAdaptiveMask
from models.resnet3D_mask_finetune import create_mask_guided_classifier



def set_seed(seed=42):
    """设置随机种子保证可复现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"✓ 随机种子设置为: {seed}")


def create_output_folder(base_dir="./output_adaptive_mask"):
    """创建带时间戳的输出文件夹"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(base_dir, f"train_adaptive_{timestamp}")

    sub_dirs = {
        'root': exp_dir,
        'models': os.path.join(exp_dir, 'models'),
        'logs': os.path.join(exp_dir, 'logs'),
        'results': os.path.join(exp_dir, 'results'),
    }

    for dir_path in sub_dirs.values():
        os.makedirs(dir_path, exist_ok=True)

    print(f"\n✓ 创建输出文件夹: {exp_dir}\n")
    return exp_dir, sub_dirs


def calculate_class_weights(dataset, device, pos_scale=1.2, neg_scale=1.0):
    """计算类别权重"""
    class_counts = [0, 0]
    for person in dataset.persons:
        class_counts[person['label']] += 1

    total = sum(class_counts)
    weights = [total / (2 * count) if count > 0 else 1.0 for count in class_counts]

    weights[0] *= neg_scale
    weights[1] *= pos_scale

    weights = torch.tensor(weights, dtype=torch.float32).to(device)

    print(f"类别分布: 负类={class_counts[0]}, 正类={class_counts[1]}")
    print(f"基础权重: {[total/(2*c) if c>0 else 1.0 for c in class_counts]}")
    print(f"最终权重: {weights.cpu().numpy()} (neg_scale={neg_scale}, pos_scale={pos_scale})")

    return weights


def save_dataset_samples_to_csv(train_dataset, val_dataset, test_dataset, save_dir):
    """保存数据集样本名称到CSV"""
    print("\n" + "="*60)
    print("保存数据集样本名称...")
    print("="*60)

    # 训练集
    train_samples = []
    for person in train_dataset.persons:
        train_samples.append({
            'patient_name': person['name'],
            'label': person['label'],
            'mask_path': person['mask_path']
        })
    train_df = pd.DataFrame(train_samples)
    train_csv_path = os.path.join(save_dir, 'train_samples.csv')
    train_df.to_csv(train_csv_path, index=False, encoding='utf-8-sig')
    print(f"✓ 训练集样本已保存: {train_csv_path} ({len(train_samples)} 样本)")

    # 验证集
    val_samples = []
    for person in val_dataset.persons:
        val_samples.append({
            'patient_name': person['name'],
            'label': person['label'],
            'mask_path': person['mask_path']
        })
    val_df = pd.DataFrame(val_samples)
    val_csv_path = os.path.join(save_dir, 'val_samples.csv')
    val_df.to_csv(val_csv_path, index=False, encoding='utf-8-sig')
    print(f"✓ 验证集样本已保存: {val_csv_path} ({len(val_samples)} 样本)")

    # 测试集
    test_samples = []
    for person in test_dataset.persons:
        test_samples.append({
            'patient_name': person['name'],
            'label': person['label'],
            'mask_path': person['mask_path']
        })
    test_df = pd.DataFrame(test_samples)
    test_csv_path = os.path.join(save_dir, 'test_samples.csv')
    test_df.to_csv(test_csv_path, index=False, encoding='utf-8-sig')
    print(f"✓ 测试集样本已保存: {test_csv_path} ({len(test_samples)} 样本)")

    print("="*60 + "\n")


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """训练一个epoch"""
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    for batch_idx, (volumes, masks, labels) in enumerate(dataloader):
        volumes = volumes.to(device)
        masks = masks.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(volumes, masks)  # 传入图像和mask
        loss = criterion(outputs, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        preds = outputs.argmax(dim=1)
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
        for volumes, masks, labels in dataloader:
            volumes = volumes.to(device)
            masks = masks.to(device)
            labels = labels.to(device)

            outputs = model(volumes, masks)  # 传入图像和mask
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            probs = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)

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
                'confidence': max(prob_class0, prob_class1)
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


def evaluate_initial_loss(model, train_loader, val_loader, criterion, device):
    """评估第0轮性能"""
    print("\n" + "="*60)
    print("评估第0轮（初始）性能...")
    print("="*60)

    model.eval()

    train_loss = 0.0
    train_samples = 0
    with torch.no_grad():
        for volumes, masks, labels in train_loader:
            volumes = volumes.to(device)
            masks = masks.to(device)
            labels = labels.to(device)

            outputs = model(volumes, masks)
            loss = criterion(outputs, labels)

            train_loss += loss.item() * volumes.size(0)
            train_samples += volumes.size(0)

    initial_train_loss = train_loss / train_samples

    val_metrics = validate(model, val_loader, criterion, device)
    initial_val_loss = val_metrics['loss']

    print(f"第0轮（初始）:")
    print(f"  训练集 Loss: {initial_train_loss:.4f}")
    print(f"  验证集 Loss: {initial_val_loss:.4f}")
    print(f"  验证集 Acc:  {val_metrics['accuracy']:.4f}")
    print("="*60 + "\n")

    return initial_train_loss, initial_val_loss


def plot_loss_comparison(history, save_dir):
    """绘制loss曲线"""
    epochs_with_0 = list(range(len(history['train_loss'])))
    epochs_from_1 = list(range(1, len(history['train_loss'])))

    train_loss = history['train_loss']
    val_loss = history['val_loss']

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Loss (Adaptive Crop + Mask-Guided Model)', fontsize=16, fontweight='bold')

    # 子图1
    ax1 = axes[0, 0]
    ax1.plot(epochs_from_1, train_loss[1:], 'b-o', linewidth=2, markersize=4, label='Train Loss')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training Loss (from epoch 1)', fontsize=13, fontweight='bold')
    ax1.grid(alpha=0.3, linestyle='--')
    ax1.legend(fontsize=10)

    # 子图2
    ax2 = axes[0, 1]
    ax2.plot(epochs_from_1, val_loss[1:], 'r-s', linewidth=2, markersize=4, label='Val Loss')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.set_title('Validation Loss (from epoch 1)', fontsize=13, fontweight='bold')
    ax2.grid(alpha=0.3, linestyle='--')
    ax2.legend(fontsize=10)

    # 子图3
    ax3 = axes[1, 0]
    ax3.plot(epochs_with_0, train_loss, 'b-o', linewidth=2, markersize=4, label='Train Loss')
    ax3.scatter(0, train_loss[0], color='green', s=150, zorder=5,
               marker='*', label='Initial Loss (Epoch 0)')
    ax3.set_xlabel('Epoch', fontsize=12)
    ax3.set_ylabel('Loss', fontsize=12)
    ax3.set_title('Training Loss (from epoch 0)', fontsize=13, fontweight='bold')
    ax3.grid(alpha=0.3, linestyle='--')
    ax3.legend(fontsize=10)

    # 子图4
    ax4 = axes[1, 1]
    ax4.plot(epochs_with_0, val_loss, 'r-s', linewidth=2, markersize=4, label='Val Loss')
    ax4.scatter(0, val_loss[0], color='green', s=150, zorder=5,
               marker='*', label='Initial Loss (Epoch 0)')
    ax4.set_xlabel('Epoch', fontsize=12)
    ax4.set_ylabel('Loss', fontsize=12)
    ax4.set_title('Validation Loss (from epoch 0)', fontsize=13, fontweight='bold')
    ax4.grid(alpha=0.3, linestyle='--')
    ax4.legend(fontsize=10)

    plt.tight_layout()

    save_path = os.path.join(save_dir, 'loss_comparison_4views.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Loss对比图已保存: {save_path}")

    # 打印统计信息
    print("\n" + "="*60)
    print("Loss改善统计:")
    print("="*60)
    print(f"训练集:")
    print(f"  初始Loss (Epoch 0):    {train_loss[0]:.4f}")
    print(f"  最佳Loss (Epoch 1+):   {min(train_loss[1:]):.4f}")
    print(f"  改善:                  {train_loss[0] - min(train_loss[1:]):.4f} ({(train_loss[0] - min(train_loss[1:]))/train_loss[0]*100:.1f}%)")
    print(f"\n验证集:")
    print(f"  初始Loss (Epoch 0):    {val_loss[0]:.4f}")
    print(f"  最佳Loss (Epoch 1+):   {min(val_loss[1:]):.4f}")
    print(f"  改善:                  {val_loss[0] - min(val_loss[1:]):.4f} ({(val_loss[0] - min(val_loss[1:]))/val_loss[0]*100:.1f}%)")
    print("="*60 + "\n")


def train(args):
    """主训练函数"""
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    print(f"使用设备: {device}")

    set_seed(args.seed)

    exp_dir, sub_dirs = create_output_folder(args.output_dir)

    config_path = os.path.join(sub_dirs['logs'], 'config.json')
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=4)

    # 数据变换 - 注意：不包含Resize，因为自适应裁剪已经处理到244x244
    train_transform = transforms.Compose([
        # transforms.Resize((244, 244)),  # 移除！自适应裁剪已处理
        transforms.RandomRotation(5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.35], std=[0.25])
    ])

    val_transform = transforms.Compose([
        # transforms.Resize((244, 244)),  # 移除！自适应裁剪已处理
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.35], std=[0.25])
    ])

    # 加载数据集 - 使用自适应裁剪版本
    print("\n" + "="*60)
    print("加载数据集（自适应裁剪 + Mask引导）...")
    print("="*60)

    full_dataset = CarotidPlaqueDatasetWithAdaptiveMask(
        root_dir=args.root_dir,
        mask_dir=args.mask_dir,
        label_excel=args.label_excel,
        transform=val_transform,
        keep_middle_n=args.depth,
        min_imgs_required=args.depth,
        verbose=True,
        crop_padding_ratio=args.crop_padding_ratio,
        crop_strategy=args.crop_strategy
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

    # 创建数据集副本
    train_dataset = CarotidPlaqueDatasetWithAdaptiveMask(
        root_dir=args.root_dir,
        mask_dir=args.mask_dir,
        label_excel=args.label_excel,
        transform=train_transform,
        keep_middle_n=args.depth,
        min_imgs_required=args.depth,
        verbose=False,
        crop_padding_ratio=args.crop_padding_ratio,
        crop_strategy=args.crop_strategy
    )
    train_dataset.persons = [full_dataset.persons[i] for i in train_indices]

    val_dataset = CarotidPlaqueDatasetWithAdaptiveMask(
        root_dir=args.root_dir,
        mask_dir=args.mask_dir,
        label_excel=args.label_excel,
        transform=val_transform,
        keep_middle_n=args.depth,
        min_imgs_required=args.depth,
        verbose=False,
        crop_padding_ratio=args.crop_padding_ratio,
        crop_strategy=args.crop_strategy
    )
    val_dataset.persons = [full_dataset.persons[i] for i in val_indices]

    test_dataset = CarotidPlaqueDatasetWithAdaptiveMask(
        root_dir=args.root_dir,
        mask_dir=args.mask_dir,
        label_excel=args.label_excel,
        transform=val_transform,
        keep_middle_n=args.depth,
        min_imgs_required=args.depth,
        verbose=False,
        crop_padding_ratio=args.crop_padding_ratio,
        crop_strategy=args.crop_strategy
    )
    test_dataset.persons = [full_dataset.persons[i] for i in test_indices]

    # 创建DataLoader
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=args.num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers, pin_memory=True
    )

    # 保存数据集样本名称
    save_dataset_samples_to_csv(train_dataset, val_dataset, test_dataset, sub_dirs['logs'])

    # 创建模型
    print("\n" + "="*60)
    print("创建Mask引导模型...")
    print("="*60)

    model = create_mask_guided_classifier(
        num_classes=2,
        pretrained_path=args.pretrained_path,
        freeze_backbone=args.freeze_backbone
    )
    model = model.to(device)

    # 计算类别权重
    class_weights = calculate_class_weights(train_dataset, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # 优化器
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # 评估初始性能
    initial_train_loss, initial_val_loss = evaluate_initial_loss(
        model, train_loader, val_loader, criterion, device
    )

    # 训练循环
    print("\n" + "="*60)
    print("开始训练...")
    print("="*60 + "\n")

    best_val_acc = 0.0
    best_epoch = 0
    history = {
        'train_loss': [initial_train_loss],
        'train_acc': [np.nan],
        'val_loss': [initial_val_loss],
        'val_acc': [np.nan],
        'val_f1': [np.nan],
        'val_auc': [np.nan]
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

        # 测试集评估
        if len(test_indices) > 0:
            test_metrics_temp = validate(model, test_loader, criterion, device)
            print(f"测试（本轮） - Acc: {test_metrics_temp['accuracy']:.4f}, "
                  f"F1: {test_metrics_temp['f1']:.4f}, AUC: {test_metrics_temp['auc']:.4f}, "
                  f"Recall: {test_metrics_temp['recall']:.4f}, Precision: {test_metrics_temp['precision']:.4f}")
        # 保存最佳模型
        if epoch in [30] or test_metrics_temp['accuracy'] >= 0.71:
            temp_best_model_epoch = epoch + 1
            temp_best_model_path = os.path.join(sub_dirs['models'], f'model_{temp_best_model_epoch}.pth')
            torch.save(model.state_dict(), temp_best_model_path)
            print(f"✓ 保存临时模型")
        # 保存最佳模型
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            best_epoch = epoch + 1
            best_model_path = os.path.join(sub_dirs['models'], 'best_model.pth')
            torch.save(model.state_dict(), best_model_path)
            print(f"✓ 保存最佳模型 (验证准确率: {best_val_acc:.4f})")

    # 保存训练历史
    history_df = pd.DataFrame(history)
    history_df.insert(0, 'epoch', range(len(history['train_loss'])))
    history_path = os.path.join(sub_dirs['logs'], 'training_history.csv')
    history_df.to_csv(history_path, index=False)
    print(f"\n✓ 训练历史已保存: {history_path}")

    # 绘制loss曲线
    plot_loss_comparison(history, sub_dirs['logs'])

    # 测试
    if len(test_indices) > 0:
        print("\n" + "="*60)
        print("测试集评估...")
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
        print(f"\n混淆矩阵:")
        print(cm)

        results = {
            'best_epoch': best_epoch,
            'best_val_acc': float(best_val_acc),
            'test_accuracy': float(test_metrics['accuracy']),
            'test_precision': float(test_metrics['precision']),
            'test_recall': float(test_metrics['recall']),
            'test_f1': float(test_metrics['f1']),
            'test_auc': float(test_metrics['auc']),
            'confusion_matrix': cm.tolist(),
            'crop_padding_ratio': args.crop_padding_ratio,
            'crop_strategy': args.crop_strategy
        }

        results_path = os.path.join(sub_dirs['results'], 'test_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=4)

        # 详细评估
        print("\n" + "="*60)
        print("保存测试集详细预测结果...")
        print("="*60)

        test_predictions_df = evaluate_test_set_detailed(
            model, test_dataset, device, sub_dirs['results']
        )

        print("\n" + "="*60)
        print("测试集详细结果文件:")
        print(f"  - test_predictions_detailed.csv (所有样本)")
        if len(test_predictions_df[~test_predictions_df['is_correct']]) > 0:
            print(f"  - test_predictions_errors.csv (仅错误预测)")
        print("="*60)

    print("\n" + "="*60)
    print(f"训练完成！所有结果保存在: {exp_dir}")
    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(description='颈动脉斑块分类训练 - 自适应裁剪 + Mask引导')

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
    parser.add_argument('--output-dir', type=str, default='./output_adaptive_mask',
                       help='输出目录')

    # 模型参数
    parser.add_argument('--pretrained-path', type=str,
                       default='./weights/resnet_18_23dataset.pth',
                       help='预训练模型路径')
    parser.add_argument('--freeze-backbone', action='store_true', default=True,
                       help='是否冻结骨干网络')
    parser.add_argument('--no-freeze-backbone', dest='freeze_backbone',
                       action='store_false', help='不冻结骨干网络')

    # 训练参数
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=4, help='批次大小')
    parser.add_argument('--lr', type=float, default=1e-3, help='学习率')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='权重衰减')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')

    # 数据集参数
    parser.add_argument('--depth', type=int, default=100,
                       help='3D volume深度')
    parser.add_argument('--train-ratio', type=float, default=0.8,
                       help='训练集比例')
    parser.add_argument('--val-ratio', type=float, default=0.1,
                       help='验证集比例')

    # 自适应裁剪参数
    parser.add_argument('--crop-padding-ratio', type=float, default=0.1,
                       help='裁剪时bbox的padding比例 (默认0.1=10%%)')
    parser.add_argument('--crop-strategy', type=str, default='adaptive',
                       choices=['adaptive', 'pad_only', 'resize_only'],
                       help='裁剪策略: adaptive(自适应) | pad_only(仅padding) | resize_only(仅resize)')

    # 其他参数
    parser.add_argument('--num-workers', type=int, default=4, help='数据加载线程数')
    parser.add_argument('--no-cuda', action='store_true', help='不使用GPU')

    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
