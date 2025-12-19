"""
3D GradCAM 实现 - 用于颈动脉斑块分类模型的可视化
基于 Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization
"""
import torch
import torch.nn.functional as F
import numpy as np
import cv2


class GradCAM3D:
    """
    3D GradCAM for 3D medical image classification

    用法:
        model = create_classifier(...)
        gradcam = GradCAM3D(model, target_layer='layer4')

        # 生成CAM
        cam = gradcam(input_tensor, target_class=1)

        # 可视化
        gradcam.visualize_3d(input_tensor, cam, save_path='output.png')
    """

    def __init__(self, model, target_layer='layer4'):
        """
        参数:
            model: PyTorch 模型
            target_layer: 目标卷积层的名称 (默认'layer4'是最后一个卷积层)
        """
        self.model = model
        self.model.eval()

        # 存储前向传播的特征图和反向传播的梯度
        self.activations = None
        self.gradients = None

        # 注册hooks到目标层
        self.target_layer = self._get_target_layer(target_layer)
        self.forward_hook = self.target_layer.register_forward_hook(self._forward_hook)
        self.backward_hook = self.target_layer.register_full_backward_hook(self._backward_hook)

        print(f"✓ GradCAM初始化完成，目标层: {target_layer}")

    def _get_target_layer(self, layer_name):
        """获取目标层"""
        # 通过名称获取模型的子模块
        for name, module in self.model.named_modules():
            if name == layer_name:
                return module
        raise ValueError(f"未找到层: {layer_name}")

    def _forward_hook(self, module, input_data, output):
        """前向传播hook: 保存特征图"""
        _ = module, input_data  # 未使用但是hook需要这些参数
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        """反向传播hook: 保存梯度"""
        _ = module, grad_input  # 未使用但是hook需要这些参数
        self.gradients = grad_output[0].detach()

    def __call__(self, input_tensor, target_class=None):
        """
        生成GradCAM

        参数:
            input_tensor: 输入张量 [1, 1, D, H, W] (batch_size必须为1)
            target_class: 目标类别索引 (None则使用预测类别)

        返回:
            cam: 类激活图 [D, H, W], 值范围[0, 1]
        """
        # 确保batch_size=1
        assert input_tensor.size(0) == 1, "GradCAM只支持batch_size=1"

        # 1. 前向传播
        self.model.zero_grad()
        output = self.model(input_tensor)  # [1, num_classes]

        # 2. 确定目标类别
        if target_class is None:
            target_class = output.argmax(dim=1).item()

        # 3. 反向传播 (只对目标类别)
        class_score = output[0, target_class]
        class_score.backward()

        # 4. 获取特征图和梯度
        activations = self.activations  # [1, C, D', H', W']
        gradients = self.gradients      # [1, C, D', H', W']

        # 5. 计算权重: 对梯度进行全局平均池化
        # 在空间维度 (D, H, W) 上求平均
        weights = gradients.mean(dim=(2, 3, 4), keepdim=True)  # [1, C, 1, 1, 1]

        # 6. 加权求和: weighted sum of feature maps
        cam = (weights * activations).sum(dim=1, keepdim=True)  # [1, 1, D', H', W']

        # 7. 应用ReLU (只保留正值)
        cam = F.relu(cam)

        # 8. 上采样到输入尺寸
        input_size = input_tensor.shape[2:]  # (D, H, W)
        cam = F.interpolate(
            cam,
            size=input_size,
            mode='trilinear',
            align_corners=False
        )  # [1, 1, D, H, W]

        # 9. 归一化到 [0, 1]
        cam = cam.squeeze()  # [D, H, W]
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        return cam.cpu().numpy()

    def visualize_3d(self, input_tensor, cam, save_path,
                     num_slices=8, alpha=0.4, colormap=cv2.COLORMAP_JET):
        """
        可视化3D GradCAM (显示多个切片)

        参数:
            input_tensor: 原始输入 [1, 1, D, H, W]
            cam: GradCAM输出 [D, H, W]
            save_path: 保存路径
            num_slices: 显示的切片数量
            alpha: CAM叠加的透明度
            colormap: OpenCV colormap
        """
        import matplotlib.pyplot as plt

        # 提取原始图像
        img_3d = input_tensor.squeeze().cpu().numpy()  # [D, H, W]
        D = img_3d.shape[0]

        # 选择均匀分布的切片
        slice_indices = np.linspace(0, D-1, num_slices, dtype=int)

        # 创建子图
        _, axes = plt.subplots(2, num_slices, figsize=(num_slices*3, 6))

        for i, slice_idx in enumerate(slice_indices):
            # 原始切片
            img_slice = img_3d[slice_idx]
            cam_slice = cam[slice_idx]

            # 归一化原始图像到 [0, 255]
            img_norm = ((img_slice - img_slice.min()) /
                       (img_slice.max() - img_slice.min() + 1e-8) * 255).astype(np.uint8)

            # 转为RGB
            img_rgb = cv2.cvtColor(img_norm, cv2.COLOR_GRAY2RGB)

            # CAM热力图
            cam_norm = (cam_slice * 255).astype(np.uint8)
            heatmap = cv2.applyColorMap(cam_norm, colormap)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

            # 叠加
            overlay = cv2.addWeighted(img_rgb, 1-alpha, heatmap, alpha, 0)

            # 绘制原始图像
            axes[0, i].imshow(img_slice, cmap='gray')
            axes[0, i].set_title(f'Slice {slice_idx}')
            axes[0, i].axis('off')

            # 绘制叠加图像
            axes[1, i].imshow(overlay)
            axes[1, i].set_title(f'GradCAM {slice_idx}')
            axes[1, i].axis('off')

        plt.suptitle(f'3D GradCAM Visualization', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"✓ 可视化已保存: {save_path}")

    def visualize_single_slice(self, input_tensor, cam, slice_idx, save_path,
                              alpha=0.4, colormap=cv2.COLORMAP_JET):
        """
        可视化单个切片的GradCAM

        参数:
            input_tensor: 原始输入 [1, 1, D, H, W]
            cam: GradCAM输出 [D, H, W]
            slice_idx: 要显示的切片索引
            save_path: 保存路径
            alpha: CAM叠加的透明度
            colormap: OpenCV colormap
        """
        import matplotlib.pyplot as plt

        # 提取原始图像
        img_3d = input_tensor.squeeze().cpu().numpy()  # [D, H, W]

        # 提取切片
        img_slice = img_3d[slice_idx]
        cam_slice = cam[slice_idx]

        # 归一化原始图像到 [0, 255]
        img_norm = ((img_slice - img_slice.min()) /
                   (img_slice.max() - img_slice.min() + 1e-8) * 255).astype(np.uint8)

        # 转为RGB
        img_rgb = cv2.cvtColor(img_norm, cv2.COLOR_GRAY2RGB)

        # CAM热力图
        cam_norm = (cam_slice * 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(cam_norm, colormap)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        # 叠加
        overlay = cv2.addWeighted(img_rgb, 1-alpha, heatmap, alpha, 0)

        # 绘制
        _, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(img_slice, cmap='gray')
        axes[0].set_title('Original Image')
        axes[0].axis('off')

        axes[1].imshow(heatmap)
        axes[1].set_title('GradCAM Heatmap')
        axes[1].axis('off')

        axes[2].imshow(overlay)
        axes[2].set_title('Overlay')
        axes[2].axis('off')

        plt.suptitle(f'GradCAM Visualization - Slice {slice_idx}',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"✓ 可视化已保存: {save_path}")

    def visualize_all_slices(self, input_tensor, cam, save_dir, patient_id,
                            alpha=0.4, colormap=cv2.COLORMAP_JET, verbose=True):
        """
        将所有切片保存为独立的图片文件

        参数:
            input_tensor: 原始输入 [1, 1, D, H, W]
            cam: GradCAM输出 [D, H, W]
            save_dir: 保存目录
            patient_id: 患者ID (用于文件命名)
            alpha: CAM叠加的透明度
            colormap: OpenCV colormap
            verbose: 是否显示进度

        返回:
            保存的文件数量
        """
        import matplotlib.pyplot as plt
        import os

        # 创建患者专属文件夹
        patient_dir = os.path.join(save_dir, f'{patient_id}')
        os.makedirs(patient_dir, exist_ok=True)

        # 提取原始图像
        img_3d = input_tensor.squeeze().cpu().numpy()  # [D, H, W]
        D = img_3d.shape[0]

        # 为每个切片生成可视化
        for slice_idx in range(D):
            # 提取切片
            img_slice = img_3d[slice_idx]
            cam_slice = cam[slice_idx]

            # 归一化原始图像到 [0, 255]
            img_norm = ((img_slice - img_slice.min()) /
                       (img_slice.max() - img_slice.min() + 1e-8) * 255).astype(np.uint8)

            # 转为RGB
            img_rgb = cv2.cvtColor(img_norm, cv2.COLOR_GRAY2RGB)

            # CAM热力图
            cam_norm = (cam_slice * 255).astype(np.uint8)
            heatmap = cv2.applyColorMap(cam_norm, colormap)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

            # 叠加
            overlay = cv2.addWeighted(img_rgb, 1-alpha, heatmap, alpha, 0)

            # 创建图像: 1行3列 (原图、热力图、叠加)
            _, axes = plt.subplots(1, 3, figsize=(12, 4))

            axes[0].imshow(img_slice, cmap='gray')
            axes[0].set_title('Original', fontsize=10)
            axes[0].axis('off')

            axes[1].imshow(heatmap)
            axes[1].set_title('GradCAM', fontsize=10)
            axes[1].axis('off')

            axes[2].imshow(overlay)
            axes[2].set_title('Overlay', fontsize=10)
            axes[2].axis('off')

            plt.suptitle(f'Slice {slice_idx:03d}/{D-1:03d}',
                        fontsize=12, fontweight='bold')
            plt.tight_layout()

            # 保存
            save_path = os.path.join(patient_dir, f'slice_{slice_idx:03d}.png')
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            plt.close()

            # 显示进度
            if verbose and (slice_idx + 1) % 20 == 0:
                print(f"  已保存 {slice_idx + 1}/{D} 个切片")

        if verbose:
            print(f"✓ 患者 {patient_id}: 共保存 {D} 个切片到 {patient_dir}")

        return D

    def remove_hooks(self):
        """移除hooks"""
        self.forward_hook.remove()
        self.backward_hook.remove()
        print("✓ Hooks已移除")


def generate_gradcam_for_dataset(model, dataloader, device, save_dir,
                                num_samples=10, target_layer='layer4'):
    """
    为数据集中的多个样本生成GradCAM可视化

    参数:
        model: 训练好的模型
        dataloader: 数据加载器
        device: 设备
        save_dir: 保存目录
        num_samples: 可视化的样本数量
        target_layer: 目标层
    """
    import os

    os.makedirs(save_dir, exist_ok=True)

    # 初始化GradCAM
    gradcam = GradCAM3D(model, target_layer=target_layer)

    model.eval()
    sample_count = 0

    with torch.no_grad():
        for volumes, labels in dataloader:
            if sample_count >= num_samples:
                break

            # 逐个样本处理 (GradCAM需要batch_size=1)
            for i in range(volumes.size(0)):
                if sample_count >= num_samples:
                    break

                # 单个样本
                volume = volumes[i:i+1].to(device)  # [1, 1, D, H, W]
                label = labels[i].item()

                # 生成GradCAM
                cam = gradcam(volume, target_class=label)

                # 可视化
                save_path = os.path.join(
                    save_dir,
                    f'sample_{sample_count:03d}_label_{label}_gradcam.png'
                )
                gradcam.visualize_3d(volume, cam, save_path, num_slices=8)

                sample_count += 1
                print(f"已处理 {sample_count}/{num_samples} 个样本")

    gradcam.remove_hooks()
    print(f"\n✓ 完成! 共生成 {sample_count} 个GradCAM可视化")
