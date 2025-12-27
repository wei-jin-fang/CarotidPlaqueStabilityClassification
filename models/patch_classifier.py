"""
基于Patch的分类器模型

架构：
1. PatchEncoder: 对每个patch提取特征（共享权重的2D CNN）
2. AttentionAggregator: 通过attention机制聚合patch特征
3. Classifier: 最终分类层

关键：forward返回attention weights用于可视化patch重要性
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEncoder(nn.Module):
    """
    Patch特征提取器
    输入：单个patch [B, C, H, W]
    输出：特征向量 [B, feature_dim]
    """

    def __init__(self, patch_size=24, in_channels=1, feature_dim=128):
        super().__init__()

        # 简单的CNN backbone
        self.features = nn.Sequential(
            # Block 1: 24x24 -> 12x12
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 2: 12x12 -> 6x6
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 3: 6x6 -> 3x3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # 投影到特征维度
        self.fc = nn.Sequential(
            nn.Linear(128, feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )

    def forward(self, x):
        """
        x: [B, C, H, W]
        返回: [B, feature_dim]
        """
        features = self.features(x)  # [B, 128, 3, 3]
        pooled = self.global_pool(features).squeeze(-1).squeeze(-1)  # [B, 128]
        out = self.fc(pooled)  # [B, feature_dim]
        return out


class AttentionAggregator(nn.Module):
    """
    Attention机制聚合多个patch的特征

    输入：[B, N_patches, feature_dim]
    输出：
        - aggregated: [B, feature_dim] 聚合后的特征
        - attention_weights: [B, N_patches] 每个patch的重要性权重
    """

    def __init__(self, feature_dim=128, hidden_dim=64):
        super().__init__()

        # Attention network
        self.attention_net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, features, mask=None):
        """
        features: [B, N, D] where N=num_patches, D=feature_dim
        mask: [B, N] 可选，1表示有效patch，0表示padding

        返回:
            aggregated: [B, D]
            attention_weights: [B, N]
        """
        # 计算attention logits
        attn_logits = self.attention_net(features)  # [B, N, 1]
        attn_logits = attn_logits.squeeze(-1)  # [B, N]

        # ★ 应用mask：将padding位置设为-inf，softmax后权重为0
        if mask is not None:
            # mask: 1=有效, 0=padding
            # 将0位置设为-1e9（近似-inf）
            mask_value = -1e9
            attn_logits = attn_logits.masked_fill(mask == 0, mask_value)

        # Softmax得到权重
        attn_weights = F.softmax(attn_logits, dim=1)  # [B, N]

        # 加权求和
        aggregated = torch.bmm(
            attn_weights.unsqueeze(1),  # [B, 1, N]
            features  # [B, N, D]
        ).squeeze(1)  # [B, D]

        return aggregated, attn_weights


class PatchBasedClassifier(nn.Module):
    """
    基于Patch的分类器

    完整流程：
    1. 每个patch通过共享的PatchEncoder提取特征
    2. Attention聚合所有patch特征
    3. 分类器输出预测

    训练时返回：logits, attention_weights
    """

    def __init__(self, patch_size=24, num_classes=2, feature_dim=128):
        super().__init__()

        # Patch特征提取器（所有patch共享）
        self.patch_encoder = PatchEncoder(
            patch_size=patch_size,
            in_channels=1,
            feature_dim=feature_dim
        )








        # Attention聚合器
        self.aggregator = AttentionAggregator(
            feature_dim=feature_dim,
            hidden_dim=64
        )
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, patches, mask=None, return_attention=True):
        """
        patches: [B, N_patches, C, H, W]
                 B=batch_size, N_patches=每个样本的patch数
        mask: [B, N_patches] 可选，1表示有效patch，0表示padding

        返回:
            logits: [B, num_classes]
            attention_weights: [B, N_patches] (如果return_attention=True)
        """
        B, N, C, H, W = patches.shape

        # 1. 展平batch和patch维度
        patches_flat = patches.view(B * N, C, H, W)

        # 2. 提取每个patch的特征（共享权重）
        features = self.patch_encoder(patches_flat)  # [B*N, feature_dim]

        # 3. 重塑为 [B, N, feature_dim]
        feature_dim = features.size(1)
        features = features.view(B, N, feature_dim)

        # 4. Attention聚合（应用mask）
        aggregated, attn_weights = self.aggregator(features, mask=mask)  # [B, D], [B, N]

        # 5. 分类
        logits = self.classifier(aggregated)  # [B, num_classes]

        if return_attention:
            return logits, attn_weights
        else:
            return logits

    def get_feature_dim(self):
        """返回特征维度（用于调试）"""
        return self.patch_encoder.fc[0].out_features


def create_patch_classifier(patch_size=24, num_classes=2, feature_dim=128):
    """
    创建Patch-based分类器的工厂函数

    参数:
        patch_size: patch大小
        num_classes: 分类数
        feature_dim: 特征维度

    返回:
        model: PatchBasedClassifier实例
    """
    model = PatchBasedClassifier(
        patch_size=patch_size,
        num_classes=num_classes,
        feature_dim=feature_dim
    )

    # 初始化权重
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    return model


if __name__ == '__main__':
    # 测试模型
    print("测试Patch-based分类器...")

    # 创建模型
    model = create_patch_classifier(patch_size=24, num_classes=2)
    model.eval()

    # 模拟输入
    batch_size = 4
    num_patches = 24  # 每个样本24个patch
    patches = torch.randn(batch_size, num_patches, 1, 24, 24)

    # 前向传播
    with torch.no_grad():
        logits, attention_weights = model(patches, return_attention=True)

    print(f"输入shape: {patches.shape}")
    print(f"输出logits shape: {logits.shape}")
    print(f"Attention weights shape: {attention_weights.shape}")
    print(f"Attention weights示例:\n{attention_weights[0]}")
    print(f"Attention weights sum: {attention_weights[0].sum():.4f} (应该≈1.0)")

    # 参数统计
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n模型参数统计:")
    print(f"  总参数: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")
