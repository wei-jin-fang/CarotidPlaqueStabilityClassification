"""
3D GradCAM 实现 - 用于自适应裁剪后的Mask引导颈动脉斑块分类模型
支持双输入模型（图像 + mask），输入图像为裁剪后的244x244
"""
import torch
import torch.nn.functional as F
import numpy as np
import cv2


class GradCAM3DAdaptiveCrop:
    """
    3D GradCAM for mask-guided 3D medical image classification (Adaptive Crop版本)

    与普通版本的区别：
    - 输入图像已经是裁剪后的244x244
    - 热力图上采样的目标也是244x244（裁剪后的原图）
    - 可视化时显示的是裁剪后的原图和mask

    用法:
        model = create_mask_guided_classifier(...)
        gradcam = GradCAM3DAdaptiveCrop(model, target_layer='layer4')

        # 生成CAM（需要同时传入裁剪后的图像和mask）
        cam = gradcam(input_tensor, mask_tensor, target_class=1)

        # 可视化
        gradcam.visualize_3d(input_tensor, mask_tensor, cam, save_path='output.png')
    """

    def __init__(self, model, target_layer='layer4'):
        """
        参数:
            model: PyTorch 模型（支持双输入：img, mask）
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

        print(f"✓ GradCAM (自适应裁剪版本) 初始化完成，目标层: {target_layer}")

    def _get_target_layer(self, layer_name):
        """获取目标层"""
        for name, module in self.model.named_modules():
            if name == layer_name:
                return module
        raise ValueError(f"未找到层: {layer_name}")

    def _forward_hook(self, module, input_data, output):
        """前向传播hook: 保存特征图"""
        _ = module, input_data
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        """反向传播hook: 保存梯度"""
        _ = module, grad_input
        self.gradients = grad_output[0].detach()

    def __call__(self, input_tensor, mask_tensor, target_class=None, return_gradients=False):
        """
        生成GradCAM

        参数:
            input_tensor: 裁剪后的输入图像 [1, 1, D, 244, 244] (batch_size必须为1)
            mask_tensor: 裁剪后的mask [1, 1, D, 244, 244]
            target_class: 目标类别索引 (None则使用预测类别)
            return_gradients: 是否返回梯度的空间分布 (用于调试)

        返回:
            如果return_gradients=False: cam [D, 244, 244]
            如果return_gradients=True: (cam [D, 244, 244], gradient_map [D, 244, 244])
        """
        # 确保batch_size=1
        assert input_tensor.size(0) == 1, "GradCAM只支持batch_size=1"
        assert mask_tensor.size(0) == 1, "GradCAM只支持batch_size=1"

        # 1. 前向传播（双输入）
        self.model.zero_grad()
        output = self.model(input_tensor, mask_tensor)  # [1, num_classes]

        # 2. 确定目标类别
        if target_class is None:
            target_class = output.argmax(dim=1).item()

        # 3. 反向传播 (只对目标类别)
        class_score = output[0, target_class]
        class_score.backward()

        # 4. 获取特征图和梯度
        activations = self.activations  # [1, C, D', H', W']
        gradients = self.gradients      # [1, C, D', H', W']

        # 5. Mask-Guided 通道权重计算（关键改进）
        mask_resized = F.interpolate(
            mask_tensor,
            size=activations.shape[2:],
            mode='nearest',          # 推荐 nearest，保持硬边界
        )  # [1, 1, D', H', W']

        weighted_grads = gradients * mask_resized          # 背景梯度直接置0
        grad_sum = weighted_grads.sum(dim=(2, 3, 4), keepdim=True)
        mask_sum = mask_resized.sum(dim=(2, 3, 4), keepdim=True) + 1e-6
        weights = grad_sum / mask_sum                      # [1, C, 1, 1, 1] 强度不稀释

        # 6. 加权求和
        cam = (weights * activations).sum(dim=1, keepdim=True)  # [1, 1, D', H', W']

        # 7. 应用ReLU
        cam = F.relu(cam)

        # 8. 上采样到裁剪后的输入尺寸 (244x244)
        input_size = input_tensor.shape[2:]  # (D, 244, 244)
        cam = F.interpolate(
            cam,
            size=input_size,
            mode='trilinear',
            align_corners=False
        )  # [1, 1, D, 244, 244]

        # 9. 归一化到 [0, 1]
        cam = cam.squeeze()  # [D, 244, 244]
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        cam_np = cam.cpu().numpy()

        # 10. 如果需要返回梯度分布
        if return_gradients:
            # 计算梯度的L2范数作为空间分布
            gradient_norm = torch.norm(gradients, p=2, dim=1, keepdim=True)  # [1, 1, D', H', W']

            # 上采样到裁剪后的输入尺寸
            gradient_map = F.interpolate(
                gradient_norm,
                size=input_size,
                mode='trilinear',
                align_corners=False
            )  # [1, 1, D, 244, 244]

            # 归一化
            gradient_map = gradient_map.squeeze()  # [D, 244, 244]
            gradient_map = gradient_map - gradient_map.min()
            gradient_map = gradient_map / (gradient_map.max() + 1e-8)

            gradient_map_np = gradient_map.cpu().numpy()

            return cam_np, gradient_map_np

        return cam_np

    def visualize_3d(self, input_tensor, mask_tensor, cam, save_path,
                     num_slices=8, alpha=0.4, colormap=cv2.COLORMAP_JET,
                     show_mask=True):
        """
        可视化3D GradCAM (显示多个切片，可选显示mask)

        参数:
            input_tensor: 裁剪后的原始图像 [1, 1, D, 244, 244]
            mask_tensor: 裁剪后的mask [1, 1, D, 244, 244]
            cam: GradCAM输出 [D, 244, 244]
            save_path: 保存路径
            num_slices: 显示的切片数量
            alpha: CAM叠加的透明度
            colormap: OpenCV colormap
            show_mask: 是否显示mask
        """
        import matplotlib.pyplot as plt

        # 提取裁剪后的图像和mask
        img_3d = input_tensor.squeeze().cpu().numpy()  # [D, 244, 244]
        mask_3d = mask_tensor.squeeze().cpu().numpy()  # [D, 244, 244]
        D = img_3d.shape[0]

        # 选择均匀分布的切片
        slice_indices = np.linspace(0, D-1, num_slices, dtype=int)

        # 创建子图：3行（原图、mask、GradCAM）或2行（原图、GradCAM）
        num_rows = 3 if show_mask else 2
        _, axes = plt.subplots(num_rows, num_slices, figsize=(num_slices*3, num_rows*3))

        for i, slice_idx in enumerate(slice_indices):
            # 原始切片
            img_slice = img_3d[slice_idx]
            mask_slice = mask_3d[slice_idx]
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
            if show_mask:
                axes[0, i].imshow(img_slice, cmap='gray')
                axes[0, i].set_title(f'Slice {slice_idx}')
                axes[0, i].axis('off')

                # 绘制mask
                axes[1, i].imshow(mask_slice, cmap='gray')
                axes[1, i].set_title(f'Mask {slice_idx}')
                axes[1, i].axis('off')

                # 绘制GradCAM
                axes[2, i].imshow(overlay)
                axes[2, i].set_title(f'GradCAM {slice_idx}')
                axes[2, i].axis('off')
            else:
                axes[0, i].imshow(img_slice, cmap='gray')
                axes[0, i].set_title(f'Slice {slice_idx}')
                axes[0, i].axis('off')

                axes[1, i].imshow(overlay)
                axes[1, i].set_title(f'GradCAM {slice_idx}')
                axes[1, i].axis('off')

        plt.suptitle(f'3D GradCAM Visualization (Adaptive Crop 244x244)',
                     fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"✓ 可视化已保存: {save_path}")

    def visualize_single_slice(self, input_tensor, mask_tensor, cam, slice_idx, save_path,
                              alpha=0.4, colormap=cv2.COLORMAP_JET):
        """
        可视化单个切片的GradCAM

        参数:
            input_tensor: 裁剪后的原始图像 [1, 1, D, 244, 244]
            mask_tensor: 裁剪后的mask [1, 1, D, 244, 244]
            cam: GradCAM输出 [D, 244, 244]
            slice_idx: 要显示的切片索引
            save_path: 保存路径
            alpha: CAM叠加的透明度
            colormap: OpenCV colormap
        """
        import matplotlib.pyplot as plt

        # 提取裁剪后的图像和mask
        img_3d = input_tensor.squeeze().cpu().numpy()
        mask_3d = mask_tensor.squeeze().cpu().numpy()

        # 提取切片
        img_slice = img_3d[slice_idx]
        mask_slice = mask_3d[slice_idx]
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
        _, axes = plt.subplots(1, 4, figsize=(20, 5))

        axes[0].imshow(img_slice, cmap='gray')
        axes[0].set_title('Cropped Image (244x244)')
        axes[0].axis('off')

        axes[1].imshow(mask_slice, cmap='gray')
        axes[1].set_title('Cropped Mask (244x244)')
        axes[1].axis('off')

        axes[2].imshow(heatmap)
        axes[2].set_title('GradCAM Heatmap')
        axes[2].axis('off')

        axes[3].imshow(overlay)
        axes[3].set_title('Overlay')
        axes[3].axis('off')

        plt.suptitle(f'GradCAM Visualization (Adaptive Crop) - Slice {slice_idx}',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"✓ 可视化已保存: {save_path}")

    def visualize_all_slices(self, input_tensor, mask_tensor, cam, save_dir, patient_id,
                            alpha=0.4, colormap=cv2.COLORMAP_JET, verbose=True,
                            gradient_map=None):
        """
        将所有切片保存为独立的图片文件

        参数:
            input_tensor: 裁剪后的原始图像 [1, 1, D, 244, 244]
            mask_tensor: 裁剪后的mask [1, 1, D, 244, 244]
            cam: GradCAM输出 [D, 244, 244]
            save_dir: 保存目录
            patient_id: 患者ID
            alpha: CAM叠加的透明度
            colormap: OpenCV colormap
            verbose: 是否显示进度
            gradient_map: 梯度空间分布 [D, 244, 244] (可选，用于调试)

        返回:
            保存的文件数量
        """
        import matplotlib.pyplot as plt
        import os

        # 创建患者专属文件夹
        patient_dir = os.path.join(save_dir, f'{patient_id}')
        os.makedirs(patient_dir, exist_ok=True)

        # 提取裁剪后的图像和mask
        img_3d = input_tensor.squeeze().cpu().numpy()  # [D, 244, 244]
        mask_3d = mask_tensor.squeeze().cpu().numpy()  # [D, 244, 244]
        D = img_3d.shape[0]

        # 决定显示几列（是否包含梯度图）
        num_cols = 5 if gradient_map is not None else 4
        figsize = (num_cols * 4, 4)

        # 为每个切片生成可视化
        for slice_idx in range(D):
            # 提取切片
            img_slice = img_3d[slice_idx]
            mask_slice = mask_3d[slice_idx]
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

            # 创建图像
            _, axes = plt.subplots(1, num_cols, figsize=figsize)

            col_idx = 0

            # 裁剪后的原始图像
            axes[col_idx].imshow(img_slice, cmap='gray')
            axes[col_idx].set_title('Cropped (244x244)', fontsize=10)
            axes[col_idx].axis('off')
            col_idx += 1

            # 裁剪后的Mask
            axes[col_idx].imshow(mask_slice, cmap='gray')
            axes[col_idx].set_title('Cropped Mask', fontsize=10)
            axes[col_idx].axis('off')
            col_idx += 1

            # 梯度热力图（如果提供）
            if gradient_map is not None:
                grad_slice = gradient_map[slice_idx]
                grad_norm = (grad_slice * 255).astype(np.uint8)
                grad_heatmap = cv2.applyColorMap(grad_norm, colormap)
                grad_heatmap = cv2.cvtColor(grad_heatmap, cv2.COLOR_BGR2RGB)

                axes[col_idx].imshow(grad_heatmap)
                axes[col_idx].set_title('Gradient Map', fontsize=10)
                axes[col_idx].axis('off')
                col_idx += 1

            # GradCAM热力图
            axes[col_idx].imshow(heatmap)
            axes[col_idx].set_title('GradCAM', fontsize=10)
            axes[col_idx].axis('off')
            col_idx += 1

            # Overlay
            axes[col_idx].imshow(overlay)
            axes[col_idx].set_title('Overlay', fontsize=10)
            axes[col_idx].axis('off')

            title = f'Slice {slice_idx:03d}/{D-1:03d} (Adaptive Crop 244x244)'
            if gradient_map is not None:
                title += ' (grad info)'

            plt.suptitle(title, fontsize=12, fontweight='bold')
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
