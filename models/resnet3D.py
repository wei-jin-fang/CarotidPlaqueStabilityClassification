"""
基于MedicalNet预训练的ResNet18的分类模型
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.resnet import BasicBlock


class ResNet3DClassifier(nn.Module):
    """
    3D ResNet分类器
    基于MedicalNet的ResNet18架构，替换分割头为分类头
    """

    def __init__(self, num_classes=2, pretrained_path=None, freeze_backbone=True):
        """
        参数:
            num_classes: 分类类别数
            pretrained_path: 预训练权重路径
            freeze_backbone: 是否冻结骨干网络
        """
        super(ResNet3DClassifier, self).__init__()
        self.inplanes = 64
        self.num_classes = num_classes

        # ========== 特征提取骨干网络 (与MedicalNet ResNet18相同) ==========
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

        # ========== 分类头（新增，多层结构） ==========
        # 全局平均池化
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))

        # 第一层：特征降维 512 -> 256
        self.fc1 = nn.Linear(512 * BasicBlock.expansion, 256)
        self.bn_fc1 = nn.BatchNorm1d(256)  # 改名避免冲突
        self.relu_fc1 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(0.5)

        # 第二层：进一步降维 256 -> 128
        self.fc2 = nn.Linear(256, 128)
        self.bn_fc2 = nn.BatchNorm1d(128)  # 改名避免冲突
        self.relu_fc2 = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout(0.5)

        # 第三层：分类输出 128 -> num_classes
        self.fc3 = nn.Linear(128, num_classes)

        # 初始化新增层的权重
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
        # 初始化第一层
        nn.init.kaiming_normal_(self.fc1.weight, mode='fan_out', nonlinearity='relu')
        if self.fc1.bias is not None:
            nn.init.constant_(self.fc1.bias, 0)

        # 初始化第二层
        nn.init.kaiming_normal_(self.fc2.weight, mode='fan_out', nonlinearity='relu')
        if self.fc2.bias is not None:
            nn.init.constant_(self.fc2.bias, 0)

        # 初始化第三层（分类层）
        nn.init.kaiming_normal_(self.fc3.weight, mode='fan_out', nonlinearity='relu')
        if self.fc3.bias is not None:
            nn.init.constant_(self.fc3.bias, 0)

    def load_pretrained_weights(self, pretrained_path, freeze_backbone=True):
        """
        加载MedicalNet预训练权重
        """
        print(f"正在加载预训练模型: {pretrained_path}")
        checkpoint = torch.load(pretrained_path, map_location='cpu', weights_only=False)

        # 获取state_dict
        if 'state_dict' in checkpoint:
            pretrained_dict = checkpoint['state_dict']
        else:
            pretrained_dict = checkpoint

        # 移除 'module.' 前缀（DataParallel保存的权重）
        new_state_dict = {}
        for k, v in pretrained_dict.items():
            if k.startswith('module.'):
                new_key = k[7:]  # 移除 'module.'
            else:
                new_key = k
            new_state_dict[new_key] = v

        # 只加载骨干网络的权重（排除conv_seg分割头）
        model_dict = self.state_dict()
        pretrained_dict_filtered = {}

        for k, v in new_state_dict.items():
            # 跳过分割头的权重
            if k.startswith('conv_seg'):
                continue
            # 只加载存在于当前模型中的权重
            if k in model_dict and model_dict[k].shape == v.shape:
                pretrained_dict_filtered[k] = v
            else:
                print(f"跳过不匹配的权重: {k}")

        # 更新模型权重
        model_dict.update(pretrained_dict_filtered)
        self.load_state_dict(model_dict)

        print(f"✓ 成功加载 {len(pretrained_dict_filtered)} 个预训练权重")

        # 冻结骨干网络
        if freeze_backbone:
            self._freeze_backbone()

    def _freeze_backbone(self):
        """冻结骨干网络，只训练分类头"""
        print("冻结骨干网络参数...")
        frozen_params = []

        # 定义分类头的层名称（不冻结这些层）
        classifier_layers = ['fc1', 'bn_fc1', 'relu_fc1', 'dropout1',
                           'fc2', 'bn_fc2', 'relu_fc2', 'dropout2',
                           'fc3', 'avgpool']

        for name, param in self.named_parameters():
            # 检查是否是分类头的层
            is_classifier = any(name.startswith(layer) for layer in classifier_layers)

            if not is_classifier:
                # 冻结骨干网络参数
                param.requires_grad = False
                frozen_params.append(name)

        print(f"✓ 冻结了 {len(frozen_params)} 个参数")
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"可训练参数: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    def unfreeze_backbone(self):
        """解冻骨干网络，进行微调"""
        print("解冻骨干网络参数...")
        for param in self.parameters():
            param.requires_grad = True

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"可训练参数: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    def forward(self, x):
        """
        前向传播
        输入: x [batch, 1, D, H, W]
        输出: logits [batch, num_classes]
        """
        # 骨干网络特征提取
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)  # [batch, 512, D', H', W']

        # 分类头（多层结构）
        x = self.avgpool(x)      # [batch, 512, 1, 1, 1]
        x = torch.flatten(x, 1)  # [batch, 512]

        # 第一层：512 -> 256
        x = self.fc1(x)
        x = self.bn_fc1(x)
        x = self.relu_fc1(x)
        x = self.dropout1(x)

        # 第二层：256 -> 128
        x = self.fc2(x)
        x = self.bn_fc2(x)
        x = self.relu_fc2(x)
        x = self.dropout2(x)

        # 第三层：128 -> num_classes
        x = self.fc3(x)  # [batch, num_classes]

        return x


def create_classifier(num_classes=2, pretrained_path=None, freeze_backbone=True):
    """
    创建分类器的便捷函数

    参数:
        num_classes: 分类类别数
        pretrained_path: 预训练权重路径
        freeze_backbone: 是否冻结骨干网络

    返回:
        model: ResNet3DClassifier模型
    """
    model = ResNet3DClassifier(
        num_classes=num_classes,
        pretrained_path=pretrained_path,
        freeze_backbone=freeze_backbone
    )
    return model
