"""
基于Patch的分类器模型(融合Excel临床特征) - CLS Token版本

架构:
1. ViTPatchEncoder: 对每个patch提取特征(使用Vision Transformer)
2. CLSTransformerAggregator: 通过CLS token + Transformer聚合patch特征
3. FeatureMLP: 提取Excel临床特征
4. FusionClassifier: 融合图像特征和临床特征,最终分类

关键: forward返回CLS token对每个patch的attention weights用于可视化
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# 导入原有的模块
from models.patch_classifier import ViTPatchEncoder


class TransformerEncoderLayerWithAttn(nn.Module):
    """
    支持返回attention weights的Transformer Encoder层

    与PyTorch标准TransformerEncoderLayer的区别:
    - 可以返回self-attention的attention weights
    - 使用Pre-LN结构(norm在attention之前)
    """

    def __init__(self, d_model, nhead, dim_feedforward=None, dropout=0.1):
        super().__init__()

        if dim_feedforward is None:
            dim_feedforward = d_model * 4

        # Multi-head self-attention
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True
        )

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )

        # Layer normalization (Pre-LN)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, key_padding_mask=None, return_attention=False):
        """
        x: [B, seq_len, d_model]
        key_padding_mask: [B, seq_len] - True表示该位置应被忽略(padding)
        return_attention: 是否返回attention weights

        返回:
            output: [B, seq_len, d_model]
            attn_weights: [B, num_heads, seq_len, seq_len] (如果return_attention=True)
        """
        # Pre-LN
        x_norm = self.norm1(x)

        # Self-attention
        # need_weights=True, average_attn_weights=False 返回每个head的attention
        attn_output, attn_weights = self.self_attn(
            x_norm, x_norm, x_norm,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=False  # 返回 [B, num_heads, seq_len, seq_len]
        )

        # Residual connection
        x = x + self.dropout(attn_output)

        # FFN with Pre-LN
        x = x + self.ffn(self.norm2(x))

        if return_attention:
            return x, attn_weights
        else:
            return x, None


class CLSTransformerAggregator(nn.Module):
    """
    使用CLS Token的Transformer聚合器

    架构:
    1. 添加learnable CLS token到patch序列开头: [CLS, patch_1, ..., patch_N]
    2. 添加position embedding
    3. 通过多层Transformer encoder处理
    4. 提取CLS token的输出用于分类
    5. 返回CLS对每个patch的attention权重(最后一层,所有head平均)

    输入: [B, N, D] - N个patch的特征
    输出:
        - cls_output: [B, D] - CLS token的输出特征
        - attention_weights: [B, N] - CLS对每个patch的attention权重
    """

    def __init__(self, feature_dim=128, num_heads=4, num_layers=2,
                 max_seq_len=3000, dropout=0.1):
        super().__init__()

        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len

        # Learnable CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, feature_dim) * 0.02)

        # Position embedding (包含CLS位置)
        # +1 是为CLS token预留的位置
        self.pos_embedding = nn.Parameter(
            torch.randn(1, max_seq_len + 1, feature_dim) * 0.02
        )

        # Transformer encoder layers
        self.layers = nn.ModuleList([
            TransformerEncoderLayerWithAttn(
                d_model=feature_dim,
                nhead=num_heads,
                dim_feedforward=feature_dim * 4,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])

        # Final layer norm
        self.norm = nn.LayerNorm(feature_dim)

    def forward(self, features, mask=None, return_attention=True):
        """
        features: [B, N, D] - N个patch的特征向量
        mask: [B, N] - 1表示有效patch, 0表示padding
        return_attention: 是否返回attention weights

        返回:
            cls_output: [B, D] - CLS token的输出特征
            attention_weights: [B, N] - CLS对每个patch的attention (所有head平均)
        """
        B, N, D = features.shape

        # 1. 添加CLS token到序列开头
        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, D]
        x = torch.cat([cls_tokens, features], dim=1)   # [B, 1+N, D]

        # 2. 添加position embedding
        # 截取到当前序列长度
        pos_embed = self.pos_embedding[:, :N+1, :]  # [1, 1+N, D]
        x = x + pos_embed

        # 3. 准备attention mask
        # key_padding_mask: True表示忽略该位置
        if mask is not None:
            # 为CLS token添加mask (CLS永远是有效的)
            cls_mask = torch.ones(B, 1, device=mask.device)  # [B, 1]
            full_mask = torch.cat([cls_mask, mask], dim=1)   # [B, 1+N]
            # 转换为padding mask格式 (True=忽略)
            key_padding_mask = (full_mask == 0)  # [B, 1+N]
        else:
            key_padding_mask = None
        
        # 4. 通过Transformer layers
        attn_weights_last = None
        for i, layer in enumerate(self.layers):
            # 只在最后一层返回attention
            need_attn = return_attention and (i == len(self.layers) - 1)
            x, attn_weights = layer(x, key_padding_mask=key_padding_mask,
                                     return_attention=need_attn)
            if need_attn:
                attn_weights_last = attn_weights

        # 5. Layer norm
        x = self.norm(x)

        # 6. 提取CLS token输出
        cls_output = x[:, 0, :]  # [B, D]

        # 7. 提取CLS对其他patch的attention weights
        if return_attention and attn_weights_last is not None:
            # attn_weights_last: [B, num_heads, 1+N, 1+N]
            # 取CLS token(第0行)对其他token的attention
            cls_attention = attn_weights_last[:, :, 0, 1:]  # [B, num_heads, N]
            # 对所有head取平均
            cls_attention = cls_attention.mean(dim=1)  # [B, N]

            # 如果有mask,将padding位置的attention设为0
            if mask is not None:
                cls_attention = cls_attention * mask

            # 重新归一化 (只在有效位置上)
            if mask is not None:
                cls_attention = cls_attention / (cls_attention.sum(dim=1, keepdim=True) + 1e-8)
        else:
            cls_attention = None

        return cls_output, cls_attention


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


class PatchFeatureFusionClassifierWithCLS(nn.Module):
    """
    基于Patch的分类器(融合Excel临床特征) - CLS Token版本

    完整流程:
    1. 图像分支: 每个patch通过共享的PatchEncoder提取特征 -> CLS Transformer聚合
    2. 特征分支: Excel临床特征通过MLP提取
    3. 融合: 将图像特征和临床特征concat
    4. 分类器输出预测

    训练时返回: logits, attention_weights
    """

    def __init__(self, patch_size=24, num_classes=2,
                 image_feature_dim=128, excel_feature_dim=74,
                 excel_hidden_dim=128,
                 vit_depth=1, vit_heads=1, vit_sub_patch_size=4,
                 cls_num_heads=4, cls_num_layers=2, max_seq_len=3000,
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

        # 2. 图像分支: CLS Transformer聚合器 (替代AttentionAggregator)
        self.aggregator = CLSTransformerAggregator(
            feature_dim=image_feature_dim,
            num_heads=cls_num_heads,
            num_layers=cls_num_layers,
            max_seq_len=max_seq_len,
            dropout=0.1
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
            attention_weights: [B, N_patches] - CLS对每个patch的attention (如果return_attention=True)
        """
        B, N, C, H, W = patches.shape

        # === 图像分支 ===
        # 1. 展平batch和patch维度
        patches_flat = patches.view(B * N, C, H, W)

        # 2. 提取每个patch的特征(共享权重)
        image_features = self.patch_encoder(patches_flat)  # [B*N, image_feature_dim]

        # 3. 重塑为 [B, N, image_feature_dim]
        image_features = image_features.view(B, N, self.image_feature_dim)

        # 4. CLS Transformer聚合(应用mask)
        aggregated_image, attn_weights = self.aggregator(
            image_features, mask=mask, return_attention=return_attention
        )  # [B, image_feature_dim], [B, N]

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


def create_patch_feature_classifier_with_cls(patch_size=24, num_classes=2,
                                              image_feature_dim=128, excel_feature_dim=74,
                                              excel_hidden_dim=128,
                                              vit_depth=1, vit_heads=1, vit_sub_patch_size=4,
                                              cls_num_heads=4, cls_num_layers=2, max_seq_len=3000,
                                              fusion_hidden_dim=256):
    """
    创建带CLS Token的Patch-Feature融合分类器的工厂函数

    使用四阶段架构:
    1. ViTPatchEncoder: 对每个patch提取特征(使用Transformer)
    2. CLSTransformerAggregator: 用CLS token聚合所有patch特征
    3. FeatureMLP: 提取Excel特征
    4. FusionClassifier: 融合分类

    参数:
        patch_size: patch大小 (例如24)
        num_classes: 分类数
        image_feature_dim: 图像特征维度
        excel_feature_dim: Excel特征维度(输入维度)
        excel_hidden_dim: Excel特征提取后的维度
        vit_depth: ViT Patch Encoder的Transformer层数
        vit_heads: ViT多头注意力头数
        vit_sub_patch_size: sub-patch大小 (例如4)
        cls_num_heads: CLS Transformer聚合器的head数量
        cls_num_layers: CLS Transformer聚合器的层数
        max_seq_len: 最大序列长度(patch数量)
        fusion_hidden_dim: 融合后的隐藏层维度

    返回:
        model: PatchFeatureFusionClassifierWithCLS实例
    """
    model = PatchFeatureFusionClassifierWithCLS(
        patch_size=patch_size,
        num_classes=num_classes,
        image_feature_dim=image_feature_dim,
        excel_feature_dim=excel_feature_dim,
        excel_hidden_dim=excel_hidden_dim,
        vit_depth=vit_depth,
        vit_heads=vit_heads,
        vit_sub_patch_size=vit_sub_patch_size,
        cls_num_heads=cls_num_heads,
        cls_num_layers=cls_num_layers,
        max_seq_len=max_seq_len,
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
    # 测试CLS Token融合架构
    print("="*60)
    print("测试Patch-Feature融合架构 (CLS Token版本)")
    print("="*60)

    # 创建模型
    model = create_patch_feature_classifier_with_cls(
        patch_size=24,
        num_classes=2,
        image_feature_dim=128,
        excel_feature_dim=74,
        excel_hidden_dim=128,
        vit_depth=1,
        vit_heads=1,
        vit_sub_patch_size=4,
        cls_num_heads=4,
        cls_num_layers=2,
        max_seq_len=3000,
        fusion_hidden_dim=256
    )
    model.eval()

    # 模拟输入
    batch_size = 4
    num_patches = 100  # 模拟100张图,每张1个patch
    patches = torch.randn(batch_size, num_patches, 1, 24, 24)
    excel_features = torch.randn(batch_size, 74)  # 74维Excel特征
    mask = torch.ones(batch_size, num_patches)  # 全部有效
    mask[0, 80:] = 0  # 第一个样本只有80个有效patch

    # 前向传播
    with torch.no_grad():
        logits, attention_weights = model(patches, excel_features, mask=mask, return_attention=True)

    print(f"\n输入:")
    print(f"  Patches shape: {patches.shape}")
    print(f"    -> [batch_size, num_patches, channels, height, width]")
    print(f"  Excel features shape: {excel_features.shape}")
    print(f"    -> [batch_size, excel_feature_dim]")
    print(f"  Mask shape: {mask.shape}")

    print(f"\n中间处理:")
    print(f"  1. 图像分支:")
    print(f"     - 展平: [{batch_size}*{num_patches}, 1, 24, 24]")
    print(f"     - ViTPatchEncoder -> [{batch_size*num_patches}, {model.image_feature_dim}]")
    print(f"     - Reshape -> [{batch_size}, {num_patches}, {model.image_feature_dim}]")
    print(f"     - CLSTransformerAggregator:")
    print(f"       * 添加CLS token: [{batch_size}, {num_patches+1}, {model.image_feature_dim}]")
    print(f"       * Transformer处理 ({model.aggregator.num_layers}层, {model.aggregator.num_heads}头)")
    print(f"       * 提取CLS输出: [{batch_size}, {model.image_feature_dim}]")
    print(f"  2. 特征分支:")
    print(f"     - FeatureMLP: [{batch_size}, 74] -> [{batch_size}, {model.excel_hidden_dim}]")
    print(f"  3. 融合:")
    print(f"     - Concat: [{batch_size}, {model.image_feature_dim}] + [{batch_size}, {model.excel_hidden_dim}]")
    print(f"     - 融合特征: [{batch_size}, {model.image_feature_dim + model.excel_hidden_dim}]")

    print(f"\n输出:")
    print(f"  Logits shape: {logits.shape}")
    print(f"    -> [batch_size, num_classes]")
    print(f"  CLS Attention weights shape: {attention_weights.shape}")
    print(f"    -> [batch_size, num_patches]")

    # 验证attention weights
    print(f"\n验证CLS Attention:")
    for i in range(batch_size):
        valid_count = int(mask[i].sum().item())
        attn_sum = attention_weights[i, :valid_count].sum().item()
        attn_max = attention_weights[i, :valid_count].max().item()
        attn_min = attention_weights[i, :valid_count].min().item()
        print(f"  样本{i}: 有效patch={valid_count}, sum={attn_sum:.4f}, max={attn_max:.4f}, min={attn_min:.6f}")

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
    print("架构说明 (CLS Token版本):")
    print("  阶段1 - ViTPatchEncoder (图像局部特征提取):")
    print("    每个patch内部用Transformer提取特征")
    print("  阶段2 - CLSTransformerAggregator (CLS全局聚合):")
    print("    用CLS token与所有patch做self-attention")
    print("    返回CLS对每个patch的attention权重")
    print("  阶段3 - FeatureMLP (Excel特征提取):")
    print("    通过MLP提取归一化后的Excel临床特征")
    print("  阶段4 - FusionClassifier (多模态融合):")
    print("    将图像特征和Excel特征concat后分类")
    print("="*60)
