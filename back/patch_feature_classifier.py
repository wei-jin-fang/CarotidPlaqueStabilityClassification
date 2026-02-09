"""
基于Patch的分类器模型(融合Excel临床特征)

架构:
1. ViTPatchEncoder: 对每个patch提取特征(使用Vision Transformer)
2. AttentionAggregator: 通过attention机制聚合patch特征
3. FeatureMLP: 提取Excel临床特征
4. FusionClassifier: 融合图像特征和临床特征,最终分类

关键: forward返回attention weights用于可视化patch重要性
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# 导入原有的模块
from models.patch_classifier import ViTPatchEncoder, AttentionAggregator


class FeatureMLP(nn.Module):
    """
    MLP提取Excel临床特征

    输入: [B, feature_dim] - 归一化后的临床特征
    输出: [B, hidden_dim] - 提取后的特征向量
    """

    def __init__(self, input_dim, hidden_dim=128, dropout=0.3):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        """
        x: [B, input_dim]
        返回: [B, hidden_dim]
        """
        return self.mlp(x)


class PatchFeatureFusionClassifier(nn.Module):
    """
    基于Patch的分类器(融合Excel临床特征)

    完整流程:
    1. 图像分支: 每个patch通过共享的PatchEncoder提取特征 -> Attention聚合
    2. 特征分支: Excel临床特征通过MLP提取
    3. 融合: 将图像特征和临床特征concat
    4. 分类器输出预测

    训练时返回: logits, attention_weights
    """

    def __init__(self, patch_size=24, num_classes=2,
                 image_feature_dim=128, excel_feature_dim=74,
                 excel_hidden_dim=128,
                 vit_depth=1, vit_heads=1, vit_sub_patch_size=4,
                 fusion_hidden_dim=256):
        super().__init__()

        self.image_feature_dim = image_feature_dim
        self.excel_feature_dim = excel_feature_dim
        self.excel_hidden_dim = excel_hidden_dim

        # 1. 图像分支: Patch特征提取器(所有patch共享) - 使用ViT Encoder
        self.patch_encoder = ViTPatchEncoder(
            patch_size=patch_size,
            in_channels=1,
            feature_dim=image_feature_dim,
            sub_patch_size=vit_sub_patch_size,
            depth=vit_depth,
            num_heads=vit_heads,
            mlp_ratio=4.0,
            dropout=0.1
        )

        # 2. 图像分支: Attention聚合器
        self.aggregator = AttentionAggregator(
            feature_dim=image_feature_dim,
            hidden_dim=64
        )

        # 3. 特征分支: MLP提取Excel特征
        self.feature_mlp = FeatureMLP(
            input_dim=excel_feature_dim,
            hidden_dim=excel_hidden_dim,
            dropout=0.3
        )

        # 4. 融合后的特征维度
        fused_dim = image_feature_dim + excel_hidden_dim

        # 5. 融合分类头
        self.fusion_classifier = nn.Sequential(
            nn.Linear(fused_dim, fusion_hidden_dim),
            nn.BatchNorm1d(fusion_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),

            nn.Linear(fusion_hidden_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),

            nn.Linear(128, num_classes)
        )

    def forward(self, patches, excel_features, mask=None, return_attention=True):
        """
        patches: [B, N_patches, C, H, W]
                 B=batch_size, N_patches=每个样本的patch数
        excel_features: [B, excel_feature_dim] Excel临床特征(已归一化)
        mask: [B, N_patches] 可选,1表示有效patch,0表示padding

        返回:
            logits: [B, num_classes]
            attention_weights: [B, N_patches] (如果return_attention=True)
        """
        B, N, C, H, W = patches.shape

        # === 图像分支 ===
        # 1. 展平batch和patch维度
        patches_flat = patches.view(B * N, C, H, W)

        # 2. 提取每个patch的特征(共享权重)
        image_features = self.patch_encoder(patches_flat)  # [B*N, image_feature_dim]

        # 3. 重塑为 [B, N, image_feature_dim]
        image_features = image_features.view(B, N, self.image_feature_dim)

        # 4. Attention聚合(应用mask)
        aggregated_image, attn_weights = self.aggregator(image_features, mask=mask)  # [B, image_feature_dim], [B, N]

        # === 特征分支 ===
        # 5. 提取Excel特征
        excel_hidden = self.feature_mlp(excel_features)  # [B, excel_hidden_dim]

        # === 融合 ===
        # 6. Concat图像特征和Excel特征
        fused = torch.cat([aggregated_image, excel_hidden], dim=1)  # [B, image_feature_dim + excel_hidden_dim]

        # 7. 分类
        logits = self.fusion_classifier(fused)  # [B, num_classes]

        if return_attention:
            return logits, attn_weights
        else:
            return logits

    def get_feature_dim(self):
        """返回特征维度(用于调试)"""
        return {
            'image_feature_dim': self.image_feature_dim,
            'excel_feature_dim': self.excel_feature_dim,
            'excel_hidden_dim': self.excel_hidden_dim,
            'fused_dim': self.image_feature_dim + self.excel_hidden_dim
        }


def create_patch_feature_classifier(patch_size=24, num_classes=2,
                                    image_feature_dim=128, excel_feature_dim=74,
                                    excel_hidden_dim=128,
                                    vit_depth=1, vit_heads=1, vit_sub_patch_size=4,
                                    fusion_hidden_dim=256):
    """
    创建Patch-Feature融合分类器的工厂函数

    使用三阶段架构:
    1. ViTPatchEncoder: 对每个patch提取特征(使用Transformer)
    2. AttentionAggregator: 聚合所有patch特征
    3. FeatureMLP + FusionClassifier: 提取Excel特征并融合分类

    参数:
        patch_size: patch大小 (例如24)
        num_classes: 分类数
        image_feature_dim: 图像特征维度
        excel_feature_dim: Excel特征维度(输入维度)
        excel_hidden_dim: Excel特征提取后的维度
        vit_depth: Transformer层数
        vit_heads: 多头注意力头数
        vit_sub_patch_size: sub-patch大小 (例如4)
        fusion_hidden_dim: 融合后的隐藏层维度

    返回:
        model: PatchFeatureFusionClassifier实例
    """
    model = PatchFeatureFusionClassifier(
        patch_size=patch_size,
        num_classes=num_classes,
        image_feature_dim=image_feature_dim,
        excel_feature_dim=excel_feature_dim,
        excel_hidden_dim=excel_hidden_dim,
        vit_depth=vit_depth,
        vit_heads=vit_heads,
        vit_sub_patch_size=vit_sub_patch_size,
        fusion_hidden_dim=fusion_hidden_dim
    )

    # 初始化权重
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    return model


if __name__ == '__main__':
    # 测试融合架构
    print("="*60)
    print("测试Patch-Feature融合架构")
    print("="*60)

    # 创建模型
    model = create_patch_feature_classifier(
        patch_size=24,
        num_classes=2,
        image_feature_dim=128,
        excel_feature_dim=74,
        excel_hidden_dim=128,
        vit_depth=1,
        vit_heads=1,
        vit_sub_patch_size=4,
        fusion_hidden_dim=256
    )
    model.eval()

    # 模拟输入
    batch_size = 4
    num_patches = 24  # 每个样本24个patch
    patches = torch.randn(batch_size, num_patches, 1, 24, 24)
    excel_features = torch.randn(batch_size, 74)  # 74维Excel特征

    # 前向传播
    with torch.no_grad():
        logits, attention_weights = model(patches, excel_features, return_attention=True)

    print(f"\n输入:")
    print(f"  Patches shape: {patches.shape}")
    print(f"    -> [batch_size, num_patches, channels, height, width]")
    print(f"  Excel features shape: {excel_features.shape}")
    print(f"    -> [batch_size, excel_feature_dim]")

    print(f"\n中间处理:")
    print(f"  1. 图像分支:")
    print(f"     - 展平: [{batch_size}*{num_patches}, 1, 24, 24]")
    print(f"     - ViTPatchEncoder -> [{batch_size*num_patches}, {model.image_feature_dim}]")
    print(f"     - Reshape -> [{batch_size}, {num_patches}, {model.image_feature_dim}]")
    print(f"     - AttentionAggregator -> [{batch_size}, {model.image_feature_dim}]")
    print(f"  2. 特征分支:")
    print(f"     - FeatureMLP: [{batch_size}, 74] -> [{batch_size}, {model.excel_hidden_dim}]")
    print(f"  3. 融合:")
    print(f"     - Concat: [{batch_size}, {model.image_feature_dim}] + [{batch_size}, {model.excel_hidden_dim}]")
    print(f"     - 融合特征: [{batch_size}, {model.image_feature_dim + model.excel_hidden_dim}]")

    print(f"\n输出:")
    print(f"  Logits shape: {logits.shape}")
    print(f"    -> [batch_size, num_classes]")
    print(f"  Attention weights shape: {attention_weights.shape}")
    print(f"    -> [batch_size, num_patches]")
    print(f"  Attention weights sum (first sample): {attention_weights[0].sum():.4f} (应该≈1.0)")

    # 参数统计
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n模型参数统计:")
    print(f"  总参数: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")

    # 模块参数统计
    print(f"\n各模块参数统计:")
    for name, module in model.named_children():
        module_params = sum(p.numel() for p in module.parameters())
        print(f"  {name:20s}: {module_params:>10,}")

    print("\n" + "="*60)
    print("架构说明:")
    print("  阶段1 - ViTPatchEncoder (图像局部特征提取):")
    print("    每个patch内部用Transformer提取特征")
    print("  阶段2 - AttentionAggregator (图像全局聚合):")
    print("    学习不同patches的重要性并聚合")
    print("  阶段3 - FeatureMLP (Excel特征提取):")
    print("    通过MLP提取归一化后的Excel临床特征")
    print("  阶段4 - FusionClassifier (多模态融合):")
    print("    将图像特征和Excel特征concat后分类")
    print("="*60)
