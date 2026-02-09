"""
基于Patch的分类器模型

架构：
1. ViTPatchEncoder: 对每个patch提取特征（使用Vision Transformer）
2. AttentionAggregator: 通过attention机制聚合patch特征
3. Classifier: 最终分类层

关键：forward返回attention weights用于可视化patch重要性
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ViTPatchEncoder(nn.Module):
    """
    基于Vision Transformer的Patch特征提取器

    输入：[B*N, C, H, W] - B*N个patches展平后的batch
    输出：[B*N, feature_dim] - 每个patch对应一个特征向量

    架构：
    1. 将每个patch分成sub-patches (例如24x24 -> 6x6个4x4的sub-patches，共36个tokens)
    2. Linear projection到embedding空间
    3. 添加position embedding
    4. 多层Transformer encoder处理这36个tokens
    5. 通过mean pooling得到该patch的特征向量

    使用方式：
        patches_flat = patches.view(B * N, C, H, W)  # [B*N, C, H, W]
        features = self.patch_encoder(patches_flat)   # [B*N, feature_dim]
        features = features.view(B, N, feature_dim)   # reshape回去
    """

    def __init__(self, patch_size=24, in_channels=1, feature_dim=128,
                 sub_patch_size=4, depth=4, num_heads=4, mlp_ratio=4.0, dropout=0.1):
        super().__init__()

        self.patch_size = patch_size
        self.sub_patch_size = sub_patch_size
        self.in_channels = in_channels
        self.feature_dim = feature_dim

        # 计算sub-patch的数量
        assert patch_size % sub_patch_size == 0, "patch_size必须能被sub_patch_size整除"
        self.num_sub_patches = (patch_size // sub_patch_size) ** 2  # 例如 (24//4)^2 = 36

        # Sub-patch embedding: 将每个sub-patch投影到embedding空间
        # 使用卷积实现 patch embedding
        self.patch_embedding = nn.Conv2d(
            in_channels, feature_dim,
            kernel_size=sub_patch_size,
            stride=sub_patch_size
        )

        # Position embedding: 为每个sub-patch添加位置信息（不需要[CLS] token）
        self.pos_embedding = nn.Parameter(
            torch.randn(1, self.num_sub_patches, feature_dim)
        )

        self.pos_dropout = nn.Dropout(dropout)

        # Transformer Encoder Layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=num_heads,
            dim_feedforward=int(feature_dim * mlp_ratio),
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=depth
        )

        # Layer normalization
        self.norm = nn.LayerNorm(feature_dim)

    def forward(self, x):
        """
        x: [B*N, C, H, W] 例如 [96, 1, 24, 24]（假设B=4, N=24）
        返回: [B*N, feature_dim]
        """
        batch_size = x.shape[0]  # 这是B*N

        # 1. Patch Embedding: [B*N, C, H, W] -> [B*N, feature_dim, num_patches_h, num_patches_w]
        x = self.patch_embedding(x)  # [B*N, feature_dim, 6, 6]

        # 2. Flatten: [B*N, feature_dim, H', W'] -> [B*N, feature_dim, num_sub_patches]
        x = x.flatten(2)  # [B*N, feature_dim, 36]

        # 3. Transpose: [B*N, feature_dim, num_sub_patches] -> [B*N, num_sub_patches, feature_dim]
        x = x.transpose(1, 2)  # [B*N, 36, feature_dim]

        # 4. 添加position embedding
        x = x + self.pos_embedding  # [B*N, 36, feature_dim]
        x = self.pos_dropout(x)

        # 5. Transformer encoding
        x = self.transformer_encoder(x)  # [B*N, 36, feature_dim]

        # 6. Mean pooling: 对所有sub-patch tokens取平均
        x = x.mean(dim=1)  # [B*N, feature_dim]

        # 7. Layer normalization
        x = self.norm(x)  # [B*N, feature_dim]

        return x


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
        # print(attn_weights[0].max(), attn_weights[0].min(), attn_weights[0].mean())
        # exit()
    #         ------------------------------------------------------------
    # tensor([0.0007, 0.0005, 0.0006,  ..., 0.0001, 0.0002, 0.0003], device='cuda:0',
    #     grad_fn=<SelectBackward0>)

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

    def __init__(self, patch_size=24, num_classes=2, feature_dim=128,
                 vit_depth=1, vit_heads=1, vit_sub_patch_size=4):
        super().__init__()

        # Patch特征提取器（所有patch共享） - 使用ViT Encoder
        self.patch_encoder = ViTPatchEncoder(
            patch_size=patch_size,
            in_channels=1,
            feature_dim=feature_dim,
            sub_patch_size=vit_sub_patch_size,
            depth=vit_depth,
            num_heads=vit_heads,
            mlp_ratio=4.0,
            dropout=0.1
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
        # print(B, N, C, H, W)
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
        return self.patch_encoder.feature_dim


def create_patch_classifier(patch_size=24, num_classes=2, feature_dim=128,
                           vit_depth=1, vit_heads=1, vit_sub_patch_size=4):
    """
    创建Patch-based分类器的工厂函数

    使用两阶段架构：
    1. ViTPatchEncoder: 对每个patch提取特征（使用Transformer）
    2. AttentionAggregator: 聚合所有patch特征

    参数:
        patch_size: patch大小 (例如24)
        num_classes: 分类数
        feature_dim: 特征维度
        vit_depth: Transformer层数
        vit_heads: 多头注意力头数
        vit_sub_patch_size: sub-patch大小 (例如4)

    返回:
        model: PatchBasedClassifier实例
    """
    model = PatchBasedClassifier(
        patch_size=patch_size,
        num_classes=num_classes,
        feature_dim=feature_dim,
        vit_depth=vit_depth,
        vit_heads=vit_heads,
        vit_sub_patch_size=vit_sub_patch_size
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
    # 测试两阶段架构
    print("="*60)
    print("测试两阶段架构: ViTPatchEncoder + AttentionAggregator")
    print("="*60)

    # 创建模型
    model = create_patch_classifier(
        patch_size=24,
        num_classes=2,
        feature_dim=128,
        vit_depth=4,
        vit_heads=4,
        vit_sub_patch_size=4
    )
    model.eval()

    # 模拟输入
    batch_size = 4
    num_patches = 24  # 每个样本24个patch
    patches = torch.randn(batch_size, num_patches, 1, 24, 24)

    # 前向传播
    with torch.no_grad():
        logits, attention_weights = model(patches, return_attention=True)

    print(f"\n输入shape: {patches.shape}")
    print(f"  -> [batch_size, num_patches, channels, height, width]")
    print(f"  -> [{batch_size}, {num_patches}, 1, 24, 24]")

    print(f"\n中间处理:")
    print(f"  1. 展平: [{batch_size}*{num_patches}, 1, 24, 24] = [{batch_size*num_patches}, 1, 24, 24]")
    print(f"  2. ViTPatchEncoder:")
    print(f"     - 每个24x24 patch -> 36个4x4 sub-patches")
    print(f"     - Transformer处理36个tokens")
    print(f"     - Mean pooling -> [{batch_size*num_patches}, {model.patch_encoder.feature_dim}]")
    print(f"  3. Reshape: [{batch_size}, {num_patches}, {model.patch_encoder.feature_dim}]")
    print(f"  4. AttentionAggregator: [{batch_size}, {model.patch_encoder.feature_dim}]")

    print(f"\n输出logits shape: {logits.shape}")
    print(f"  -> [batch_size, num_classes]")
    print(f"\nAttention weights shape: {attention_weights.shape}")
    print(f"  -> [batch_size, num_patches]")
    print(f"\nAttention weights示例 (第一个样本):")
    print(f"  {attention_weights[0]}")
    print(f"  Sum: {attention_weights[0].sum():.4f} (应该≈1.0)")

    # 参数统计
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n模型参数统计:")
    print(f"  总参数: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")

    print("\n" + "="*60)
    print("架构说明:")
    print("  阶段1 - ViTPatchEncoder (局部特征提取):")
    print("    每个patch内部用Transformer提取特征")
    print("  阶段2 - AttentionAggregator (全局聚合):")
    print("    学习不同patches的重要性并聚合")
    print("="*60)
