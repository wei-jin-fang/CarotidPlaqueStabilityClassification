"""
基于Excel临床特征的颈动脉斑块分类器

架构:
1. FeatureMLP: 提取Excel临床特征
2. Classifier: 分类头

从patch_feature_classifier.py中提取FeatureMLP,
去掉图像分支,只保留特征分支+分类头
"""
import torch
import torch.nn as nn


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


class FeatureClassifier(nn.Module):
    """
    基于Excel临床特征的分类器

    流程:
    1. FeatureMLP提取特征
    2. 分类头输出预测
    """

    def __init__(self, excel_feature_dim=74, excel_hidden_dim=128,
                 fusion_hidden_dim=256, num_classes=2):
        super().__init__()

        self.excel_feature_dim = excel_feature_dim
        self.excel_hidden_dim = excel_hidden_dim

        # 1. MLP提取Excel特征
        self.feature_mlp = FeatureMLP(
            input_dim=excel_feature_dim,
            hidden_dim=excel_hidden_dim,
            dropout=0.3
        )

        # 2. 分类头
        self.classifier = nn.Sequential(
            nn.Linear(excel_hidden_dim, fusion_hidden_dim),
            nn.BatchNorm1d(fusion_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),

            nn.Linear(fusion_hidden_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),

            nn.Linear(128, num_classes)
        )

    def forward(self, excel_features):
        """
        excel_features: [B, excel_feature_dim] Excel临床特征(已归一化)

        返回:
            logits: [B, num_classes]
        """
        hidden = self.feature_mlp(excel_features)  # [B, excel_hidden_dim]
        logits = self.classifier(hidden)  # [B, num_classes]
        return logits


def create_feature_classifier(excel_feature_dim=74, excel_hidden_dim=128,
                               fusion_hidden_dim=256, num_classes=2):
    """
    创建Excel特征分类器的工厂函数

    参数:
        excel_feature_dim: Excel特征维度(输入维度)
        excel_hidden_dim: Excel特征提取后的维度
        fusion_hidden_dim: 分类头隐藏层维度
        num_classes: 分类数

    返回:
        model: FeatureClassifier实例
    """
    model = FeatureClassifier(
        excel_feature_dim=excel_feature_dim,
        excel_hidden_dim=excel_hidden_dim,
        fusion_hidden_dim=fusion_hidden_dim,
        num_classes=num_classes
    )

    # 初始化权重
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d,)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    return model


if __name__ == '__main__':
    print("=" * 60)
    print("测试Excel特征分类器")
    print("=" * 60)

    model = create_feature_classifier(
        excel_feature_dim=74,
        excel_hidden_dim=128,
        fusion_hidden_dim=256,
        num_classes=2
    )
    model.eval()

    batch_size = 4
    excel_features = torch.randn(batch_size, 74)

    with torch.no_grad():
        logits = model(excel_features)

    print(f"\n输入:")
    print(f"  Excel features shape: {excel_features.shape}")
    print(f"\n输出:")
    print(f"  Logits shape: {logits.shape}")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n模型参数统计:")
    print(f"  总参数: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")

    print(f"\n各模块参数统计:")
    for name, module in model.named_children():
        module_params = sum(p.numel() for p in module.parameters())
        print(f"  {name:20s}: {module_params:>10,}")

    print("=" * 60)
