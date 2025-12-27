"""
基于Mask加权池化的3D ResNet分类模型
在最后一层特征图上使用mask进行加权全局平均池化
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.resnet import BasicBlock


class ResNet3DClassifierWithMask(nn.Module):
    """
    使用Mask加权池化的3D ResNet分类器
    在layer4输出后，用mask对特征图进行加权平均池化，聚焦ROI区域
    """

    def __init__(self, num_classes=2, pretrained_path=None, freeze_backbone=True):
        """
        参数:
            num_classes: 分类类别数
            pretrained_path: 预训练权重路径
            freeze_backbone: 是否冻结骨干网络
        """
        super(ResNet3DClassifierWithMask, self).__init__()
        self.inplanes = 64
        self.num_classes = num_classes

        # ========== 特征提取骨干网络 ==========
        self.conv1 = nn.Conv3d(
            1, 64, kernel_size=7, stride=(2, 2, 2),
            padding=(3, 3, 3), bias=False
        )
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)

        self.layer1 = self._make_layer(BasicBlock, 64, 2, shortcut_type='B')
        self.layer2 = self._make_layer(BasicBlock, 128, 2, shortcut_type='B', stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, shortcut_type='B', stride=1, dilation=2)
        self.layer4 = self._make_layer(BasicBlock, 512, 2, shortcut_type='B', stride=1, dilation=4)

        # 注意：不使用标准的AdaptiveAvgPool3d，而是使用mask加权池化

        # ========== 分类头 ==========
        self.fc1 = nn.Linear(512 * BasicBlock.expansion, 128)
        self.bn_fc1 = nn.BatchNorm1d(128)
        self.relu_fc1 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(0.5)

        self.fc2 = nn.Linear(128, 128)
        self.bn_fc2 = nn.BatchNorm1d(128)
        self.relu_fc2 = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout(0.7)

        self.fc3 = nn.Linear(128, num_classes)

        # 初始化权重
        self._init_classifier_weights()

        # 加载预训练权重
        if pretrained_path:
            self.load_pretrained_weights(pretrained_path, freeze_backbone)

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1, dilation=1):
        """构建ResNet层"""
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(
                    self.inplanes, planes * block.expansion,
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm3d(planes * block.expansion)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride=stride,
                           dilation=dilation, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def _init_classifier_weights(self):
        """初始化分类头的权重"""
        nn.init.kaiming_normal_(self.fc1.weight, mode='fan_out', nonlinearity='relu')
        if self.fc1.bias is not None:
            nn.init.constant_(self.fc1.bias, 0)

        nn.init.kaiming_normal_(self.fc2.weight, mode='fan_out', nonlinearity='relu')
        if self.fc2.bias is not None:
            nn.init.constant_(self.fc2.bias, 0)

        nn.init.kaiming_normal_(self.fc3.weight, mode='fan_out', nonlinearity='relu')
        if self.fc3.bias is not None:
            nn.init.constant_(self.fc3.bias, 0)

    def load_pretrained_weights(self, pretrained_path, freeze_backbone=True):
        """
        加载MedicalNet预训练权重
        """
        print(f"正在加载预训练模型: {pretrained_path}")
        checkpoint = torch.load(pretrained_path, map_location='cpu', weights_only=False)

        if 'state_dict' in checkpoint:
            pretrained_dict = checkpoint['state_dict']
        else:
            pretrained_dict = checkpoint

        # 移除 'module.' 前缀
        new_state_dict = {}
        for k, v in pretrained_dict.items():
            if k.startswith('module.'):
                new_key = k[7:]
            else:
                new_key = k
            new_state_dict[new_key] = v

        # 加载权重
        model_dict = self.state_dict()
        pretrained_dict_filtered = {}

        for k, v in new_state_dict.items():
            if k.startswith('conv_seg'):
                continue

            # 正常加载
            if k in model_dict and model_dict[k].shape == v.shape:
                pretrained_dict_filtered[k] = v

        # 更新模型权重
        model_dict.update(pretrained_dict_filtered)
        self.load_state_dict(model_dict)

        print(f"✓ 成功加载 {len(pretrained_dict_filtered)} 个预训练权重")

        if freeze_backbone:
            self._freeze_backbone()

    def _freeze_backbone(self):
        """
        冻结骨干网络，但保留 layer4 可训练（适合医疗图像微调）
        """
        print("正在冻结骨干网络（保留 layer4 可训练）...")
        
        frozen_params = []
        unfrozen_params = []

        # 定义哪些模块属于分类头（始终可训练）
        classifier_prefixes = ['fc1', 'bn_fc1', 'relu_fc1', 'dropout1',
                               'fc2', 'bn_fc2', 'relu_fc2', 'dropout2', 'fc3']

        # 定义要保留可训练的骨干部分（这里是 layer4）
        unfreeze_prefixes = ['layer4']

        for name, param in self.named_parameters():
            # 判断是否属于分类头 → 必须可训练
            is_classifier = any(name.startswith(prefix) for prefix in classifier_prefixes)
            # 判断是否属于要解冻的层（如 layer4）
            is_unfrozen_layer = any(name.startswith(prefix) for prefix in unfreeze_prefixes)

            if is_classifier or is_unfrozen_layer:
                param.requires_grad = True
                unfrozen_params.append(name)
            else:
                param.requires_grad = False
                frozen_params.append(name)

        print(f"✓ 冻结参数: {len(frozen_params)} 个")
        print(f"✓ 可训练参数: {len(unfrozen_params)} 个（包括 layer4 + 分类头）")

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"总可训练参数量: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

    def unfreeze_backbone(self):
        """解冻骨干网络"""
        print("解冻骨干网络参数...")
        for param in self.parameters():
            param.requires_grad = True

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"可训练参数: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    def weighted_global_avg_pool(self, features, mask):
        """
        加权全局平均池化

        Args:
            features: [B, C, D, H, W] - layer4的输出特征图
            mask: [B, 1, D_orig, H_orig, W_orig] - 原始mask

        Returns:
            pooled: [B, C] - 加权池化后的特征向量
        """
        # 将mask下采样到特征图尺寸
        mask_resized = F.interpolate(
            mask,
            size=features.shape[2:],  # (D, H, W)
            mode='trilinear',
            align_corners=False
        )  # [B, 1, D', H', W']

        # 加权特征：ROI区域权重高，背景区域权重低
        weighted_features = features * mask_resized  # [B, C, D', H', W']

        # 计算加权平均
        # sum over spatial dimensions
        pooled = weighted_features.sum(dim=(2, 3, 4))  # [B, C]

        # 归一化：除以mask的总和（每个通道相同）
        mask_sum = mask_resized.sum(dim=(2, 3, 4)) + 1e-6  # [B, 1]，加小值避免除零
        pooled = pooled / mask_sum  # [B, C]

        return pooled

    def forward(self, img, mask):
        """
        前向传播
        Args:
            img: 图像 [batch, 1, D, H, W]
            mask: mask [batch, 1, D, H, W]
        Returns:
            logits [batch, num_classes]
        """
        # 骨干网络特征提取（只用图像，不用mask）
        x = self.conv1(img)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)  # [batch, 512, D', H', W']

        # 关键：使用mask进行加权全局平均池化
        x = self.weighted_global_avg_pool(x, mask)  # [batch, 512]

        # 分类头
        x = self.fc1(x)
        x = self.bn_fc1(x)
        x = self.relu_fc1(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.bn_fc2(x)
        x = self.relu_fc2(x)
        x = self.dropout2(x)

        x = self.fc3(x)  # [batch, num_classes]

        return x


def create_mask_guided_classifier(num_classes=2, pretrained_path=None, freeze_backbone=True):
    """
    创建Mask加权池化分类器的便捷函数

    参数:
        num_classes: 分类类别数
        pretrained_path: 预训练权重路径
        freeze_backbone: 是否冻结骨干网络

    返回:
        model: ResNet3DClassifierWithMask模型
    """
    model = ResNet3DClassifierWithMask(
        num_classes=num_classes,
        pretrained_path=pretrained_path,
        freeze_backbone=freeze_backbone
    )
    return model

