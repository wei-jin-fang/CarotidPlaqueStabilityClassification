"""
Patch Attention可视化脚本

功能:
1. 读取训练后保存的预测结果（包含attention权重和patch位置）
2. 将patch的attention权重映射回原图，生成热力图
3. 标注最重要的patch
4. 对比正确/错误预测的attention模式

注意：patch的位置坐标是相对于自适应剪裁后的244×244图像
"""
import os
import argparse
import pickle
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib import cm
from glob import glob
from PIL import Image


class AdaptiveMaskCrop:
    """
    基于mask的自适应裁剪transform

    处理流程:
    1. 找到mask中所有白色区域的最小外接矩形
    2. 添加padding扩展感受野
    3. 裁剪图像和mask
    4. 智能调整到目标尺寸（padding优先，避免直接resize）
    """

    def __init__(self, target_size=(244, 244), padding_ratio=0.1,
                 threshold=127, strategy='adaptive'):
        """
        参数:
            target_size: 目标尺寸 (H, W)
            padding_ratio: 在bbox周围添加的padding比例（相对于bbox尺寸）
            threshold: 二值化阈值
            strategy: 'adaptive' | 'pad_only' | 'resize_only'
        """
        self.target_size = target_size
        self.padding_ratio = padding_ratio
        self.threshold = threshold
        self.strategy = strategy

    def find_bounding_box(self, mask_np):
        """找到包含所有白色区域的最小外接矩形"""
        # 二值化
        _, binary = cv2.threshold(mask_np, self.threshold, 255, cv2.THRESH_BINARY)

        # 找到所有白色像素
        coords = np.column_stack(np.where(binary > 0))

        if len(coords) == 0:
            # 没有白色区域，返回整个图像
            h, w = mask_np.shape
            return 0, 0, w, h

        # 计算bbox
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)

        return int(x_min), int(y_min), int(x_max), int(y_max)

    def add_padding_to_bbox(self, bbox, img_shape):
        """在bbox周围添加padding"""
        x_min, y_min, x_max, y_max = bbox
        h, w = img_shape

        bbox_w = x_max - x_min
        bbox_h = y_max - y_min

        pad_w = int(bbox_w * self.padding_ratio)
        pad_h = int(bbox_h * self.padding_ratio)

        # 扩展但不超出边界
        x_min = max(0, x_min - pad_w)
        y_min = max(0, y_min - pad_h)
        x_max = min(w, x_max + pad_w)
        y_max = min(h, y_max + pad_h)

        return x_min, y_min, x_max, y_max

    def process_cropped_region(self, cropped, target_h, target_w):
        """处理裁剪后的区域到目标尺寸"""
        crop_h, crop_w = cropped.shape[:2]

        if self.strategy == 'resize_only':
            # 直接resize
            return cv2.resize(cropped, (target_w, target_h),
                            interpolation=cv2.INTER_LINEAR)

        elif self.strategy == 'pad_only':
            # 只padding
            if crop_h > target_h or crop_w > target_w:
                raise ValueError(
                    f"Cropped size ({crop_w}x{crop_h}) exceeds target ({target_w}x{target_h}). "
                    "Use 'adaptive' or 'resize_only' strategy."
                )
            return self._pad_to_size(cropped, target_h, target_w)

        else:  # adaptive
            if crop_h == target_h and crop_w == target_w:
                return cropped

            elif crop_h <= target_h and crop_w <= target_w:
                # 更小，padding
                return self._pad_to_size(cropped, target_h, target_w)

            elif crop_h >= target_h and crop_w >= target_w:
                # 更大，判断是center crop还是resize
                ratio_h = crop_h / target_h
                ratio_w = crop_w / target_w

                if ratio_h > 1.5 or ratio_w > 1.5:
                    # 差距太大，resize
                    return cv2.resize(cropped, (target_w, target_h),
                                    interpolation=cv2.INTER_LINEAR)
                else:
                    # 差距小，center crop
                    return self._center_crop(cropped, target_h, target_w)

            else:
                # 一边大一边小
                # 先padding到正方形
                max_dim = max(crop_h, crop_w)
                padded = self._pad_to_size(cropped, max_dim, max_dim)

                if max_dim <= max(target_h, target_w):
                    # 还是不够大，继续padding
                    return self._pad_to_size(padded, target_h, target_w)
                else:
                    # 太大了，resize
                    return cv2.resize(padded, (target_w, target_h),
                                    interpolation=cv2.INTER_LINEAR)

    def _pad_to_size(self, img, target_h, target_w):
        """零填充到目标尺寸（居中）"""
        h, w = img.shape[:2]

        pad_h = target_h - h
        pad_w = target_w - w

        if pad_h < 0 or pad_w < 0:
            raise ValueError(f"Cannot pad {w}x{h} to {target_w}x{target_h}")

        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        if len(img.shape) == 2:
            # 灰度图
            padded = cv2.copyMakeBorder(img, pad_top, pad_bottom,
                                       pad_left, pad_right,
                                       cv2.BORDER_CONSTANT, value=0)
        else:
            # 彩色图
            padded = cv2.copyMakeBorder(img, pad_top, pad_bottom,
                                       pad_left, pad_right,
                                       cv2.BORDER_CONSTANT, value=(0, 0, 0))

        return padded

    def _center_crop(self, img, target_h, target_w):
        """中心裁剪"""
        h, w = img.shape[:2]

        start_h = (h - target_h) // 2
        start_w = (w - target_w) // 2

        return img[start_h:start_h+target_h, start_w:start_w+target_w]

    def __call__(self, image, mask):
        """
        应用transform

        参数:
            image: PIL Image 或 numpy array
            mask: PIL Image 或 numpy array (mask图像)

        返回:
            image_processed: numpy array, shape (target_h, target_w)
            mask_processed: numpy array, shape (target_h, target_w)
        """
        # 转换为numpy
        if isinstance(image, Image.Image):
            image = np.array(image)
        if isinstance(mask, Image.Image):
            mask = np.array(mask)

        # 找到bbox
        bbox = self.find_bounding_box(mask)
        bbox_padded = self.add_padding_to_bbox(bbox, mask.shape)

        x_min, y_min, x_max, y_max = bbox_padded

        # 裁剪
        image_cropped = image[y_min:y_max, x_min:x_max]
        mask_cropped = mask[y_min:y_max, x_min:x_max]

        # 调整到目标尺寸
        target_h, target_w = self.target_size

        image_processed = self.process_cropped_region(image_cropped, target_h, target_w)
        mask_processed = self.process_cropped_region(mask_cropped, target_h, target_w)

        return image_processed, mask_processed

    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'target_size={self.target_size}, '
                f'padding_ratio={self.padding_ratio}, '
                f'strategy={self.strategy})')


def load_predictions(results_file):
    """
    加载预测结果

    返回:
        results: list of dict，包含attention_weights和positions
    """
    with open(results_file, 'rb') as f:
        results = pickle.load(f)

    print(f"✓ 加载预测结果: {results_file}")
    print(f"  共 {len(results)} 个样本")

    return results


def create_attention_heatmap(img_shape, positions, attention_weights):
    """
    根据patch位置和attention权重创建热力图

    参数:
        img_shape: (H, W) 原图尺寸
        positions: list of dict，每个包含bbox信息
        attention_weights: [N_patches] numpy array

    返回:
        heatmap: [H, W] 热力图
    """
    H, W = img_shape
    heatmap = np.zeros((H, W), dtype=np.float32)
    count_map = np.zeros((H, W), dtype=np.float32)

    # 遍历每个patch
    for pos, weight in zip(positions, attention_weights):
        x1, y1, x2, y2 = pos['bbox']

        # 确保在图像范围内
        x1 = max(0, min(x1, W))
        x2 = max(0, min(x2, W))
        y1 = max(0, min(y1, H))
        y2 = max(0, min(y2, H))

        if x2 > x1 and y2 > y1:
            heatmap[y1:y2, x1:x2] += weight
            count_map[y1:y2, x1:x2] += 1

    # 处理重叠区域（取平均）
    mask = count_map > 0
    heatmap[mask] /= count_map[mask]

    # 归一化到0-1
    if heatmap.max() > 0:
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())

    return heatmap


def visualize_sample_attention(img_path, mask_path, positions, attention_weights,
                               prediction_info, save_path, top_k=5):
    """
    可视化单个样本的attention

    参数:
        img_path: 图像路径
        mask_path: mask路径
        positions: patch位置列表
        attention_weights: attention权重
        prediction_info: 预测信息dict
        save_path: 保存路径
        top_k: 标注前k个重要的patch
    """
    # 读取原始图像和mask
    img_original = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    mask_original = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if img_original is None or mask_original is None:
        print(f"警告: 无法读取图像 {img_path}")
        return

    # ★ 关键：先做自适应剪裁到244×244（与训练时保持一致）
    adaptive_crop = AdaptiveMaskCrop(
        target_size=(244, 244),
        padding_ratio=0,
        threshold=127,
        strategy='adaptive'
    )
    img, mask = adaptive_crop(img_original, mask_original)

    # 创建热力图（现在position坐标对应244×244图像）
    heatmap = create_attention_heatmap(img.shape, positions, attention_weights)

    # 应用colormap
    heatmap_colored = (heatmap * 255).astype(np.uint8)
    heatmap_colored = cv2.applyColorMap(heatmap_colored, cv2.COLORMAP_JET)

    # 转换为RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    # 叠加热力图
    overlay = cv2.addWeighted(img_rgb, 0.5, heatmap_colored, 0.5, 0)

    # 标注Top-K重要的patch
    top_indices = np.argsort(attention_weights)[-top_k:][::-1]

    for rank, idx in enumerate(top_indices):
        pos = positions[idx]
        x1, y1, x2, y2 = pos['bbox']
        weight = attention_weights[idx]

        # 绘制边界框
        color = (0, 255, 0) if rank == 0 else (255, 255, 0)  # 第一名绿色，其他黄色
        thickness = 3 if rank == 0 else 2
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, thickness)

        # 标注权重
        text = f'#{rank+1}: {weight:.3f}'
        cv2.putText(overlay, text, (x1, y1-5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    # 创建figure
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # 原图
    axes[0].imshow(img, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    # Mask
    axes[1].imshow(mask, cmap='gray')
    axes[1].set_title('Mask (ROI)')
    axes[1].axis('off')

    # 热力图
    axes[2].imshow(heatmap, cmap='jet')
    axes[2].set_title('Attention Heatmap')
    axes[2].axis('off')

    # 叠加图
    axes[3].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    axes[3].set_title(f'Overlay (Top-{top_k} Patches)')
    axes[3].axis('off')

    # 添加预测信息
    true_label = prediction_info['true_label']
    pred_label = prediction_info['predicted_label']
    confidence = prediction_info['confidence']
    is_correct = prediction_info['is_correct']

    status = "✓ Correct" if is_correct else "✗ Wrong"
    title = (f"{prediction_info['patient_name']} - {status}\n"
            f"True: {true_label}, Pred: {pred_label}, Conf: {confidence:.3f}")

    fig.suptitle(title, fontsize=14, fontweight='bold',
                color='green' if is_correct else 'red')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  ✓ 已保存: {save_path}")


def find_image_path(root_dir, patient_name, slice_idx=None):
    """
    查找患者的图像路径

    如果指定slice_idx，返回该slice；否则返回中间的slice
    """
    patient_dir = os.path.join(root_dir, patient_name)
    if not os.path.exists(patient_dir):
        return None

    # 查找所有图片
    img_paths = []
    for jpg_dir in glob(os.path.join(patient_dir, '**', '*_JPG'), recursive=True):
        for ext in ('*.jpg', '*.JPG'):
            img_paths.extend(glob(os.path.join(jpg_dir, ext)))

    if not img_paths:
        return None

    img_paths = sorted(img_paths)

    if slice_idx is not None:
        if slice_idx < len(img_paths):
            return img_paths[slice_idx]
        else:
            return None
    else:
        # 返回中间的slice
        return img_paths[len(img_paths) // 2]


def visualize_all_samples(results, root_dir, mask_dir, output_dir,
                         max_samples=None, only_errors=False):
    """
    批量可视化样本

    参数:
        results: 预测结果列表
        root_dir: 数据根目录
        mask_dir: mask目录
        output_dir: 输出目录
        max_samples: 最多可视化的样本数
        only_errors: 是否只可视化错误预测
    """
    os.makedirs(output_dir, exist_ok=True)

    # 过滤
    if only_errors:
        results = [r for r in results if not r['is_correct']]
        print(f"\n只可视化错误预测: {len(results)} 个样本")

    if max_samples is not None:
        results = results[:max_samples]

    print(f"\n开始批量可视化 {len(results)} 个样本...")

    for idx, result in enumerate(results):
        patient_name = result['patient_name']

        # 查找图像（使用attention权重最高的patch所在的slice）
        positions = result['positions']
        attention_weights_all = result['attention_weights']

        if len(positions) == 0:
            print(f"  跳过 {patient_name}: 没有patch")
            continue

        # 找到attention权重最高的patch所在的slice（这样能看到模型最关注的区域）
        max_attn_idx = np.argmax(attention_weights_all)
        best_slice_idx = positions[max_attn_idx]['slice_idx']

        img_path = find_image_path(root_dir, patient_name, best_slice_idx)
        mask_path = os.path.join(mask_dir, f"{patient_name}_mask.png")

        if img_path is None or not os.path.exists(mask_path):
            print(f"  跳过 {patient_name}: 找不到图像或mask")
            continue

        # 只使用该slice的patch
        slice_positions = [p for p in positions if p['slice_idx'] == best_slice_idx]
        slice_indices = [i for i, p in enumerate(positions) if p['slice_idx'] == best_slice_idx]
        slice_attention = attention_weights_all[slice_indices]

        if len(slice_positions) == 0:
            continue

        # 可视化
        save_name = f"{idx+1:03d}_{patient_name}_slice{best_slice_idx}.png"
        save_path = os.path.join(output_dir, save_name)

        visualize_sample_attention(
            img_path, mask_path,
            slice_positions, slice_attention,
            result, save_path, top_k=5
        )

        if (idx + 1) % 10 == 0:
            print(f"  已完成 {idx+1}/{len(results)}")

    print(f"\n✓ 批量可视化完成！结果保存在: {output_dir}")


def visualize_all_slices_per_patient(results, root_dir, mask_dir, output_dir,
                                     max_samples=None, only_errors=False):
    """
    为每个患者创建文件夹，可视化该患者的所有切片

    参数:
        results: 预测结果列表
        root_dir: 数据根目录
        mask_dir: mask目录
        output_dir: 输出目录（会在其中为每个患者创建子文件夹）
        max_samples: 最多可视化的样本数
        only_errors: 是否只可视化错误预测
    """
    os.makedirs(output_dir, exist_ok=True)

    # 过滤
    if only_errors:
        results = [r for r in results if not r['is_correct']]
        print(f"\n只可视化错误预测: {len(results)} 个样本")

    if max_samples is not None:
        results = results[:max_samples]

    print(f"\n开始为 {len(results)} 个患者可视化所有切片...")

    for idx, result in enumerate(results):
        patient_name = result['patient_name']
        positions = result['positions']
        attention_weights_all = result['attention_weights']

        if len(positions) == 0:
            print(f"  跳过 {patient_name}: 没有patch")
            continue

        # 为该患者创建子文件夹
        patient_dir = os.path.join(output_dir, f"{idx+1:03d}_{patient_name}")
        os.makedirs(patient_dir, exist_ok=True)

        # 获取该患者的所有slice索引
        all_slice_indices = sorted(set(p['slice_idx'] for p in positions))

        print(f"\n[{idx+1}/{len(results)}] {patient_name} - {len(all_slice_indices)} 个切片")

        mask_path = os.path.join(mask_dir, f"{patient_name}_mask.png")
        if not os.path.exists(mask_path):
            print(f"    跳过: 找不到mask")
            continue

        # 找到最重要的slice（用于标注）
        max_attn_idx = np.argmax(attention_weights_all)
        best_slice_idx = positions[max_attn_idx]['slice_idx']

        # 遍历所有slice进行可视化
        visualized_count = 0
        for slice_idx in all_slice_indices:
            # 获取该slice的图像路径
            img_path = find_image_path(root_dir, patient_name, slice_idx)
            if img_path is None:
                continue

            # 提取该slice的patch和attention
            slice_positions = [p for p in positions if p['slice_idx'] == slice_idx]
            slice_indices = [i for i, p in enumerate(positions) if p['slice_idx'] == slice_idx]
            slice_attention = attention_weights_all[slice_indices]

            if len(slice_positions) == 0:
                continue

            # 计算该slice的平均attention（用于排序和标注）
            avg_attention = slice_attention.mean()

            # 标注是否为最重要的slice
            is_best = (slice_idx == best_slice_idx)
            slice_label = f"slice{slice_idx:03d}"
            if is_best:
                slice_label += "_BEST"

            # 保存路径
            save_name = f"{slice_label}_avg{avg_attention:.4f}.png"
            save_path = os.path.join(patient_dir, save_name)

            # 可视化
            visualize_sample_attention(
                img_path, mask_path,
                slice_positions, slice_attention,
                result, save_path, top_k=3  # 用3个框，避免太乱
            )

            visualized_count += 1

        print(f"    ✓ 已可视化 {visualized_count} 个切片，保存至: {patient_dir}")

        # 创建summary文件
        summary_path = os.path.join(patient_dir, "_summary.txt")
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(f"患者: {patient_name}\n")
            f.write(f"预测: {result['predicted_label']}\n")
            f.write(f"真实标签: {result['true_label']}\n")
            f.write(f"置信度: {result['confidence']:.4f}\n")
            f.write(f"是否正确: {'✓' if result['is_correct'] else '✗'}\n")
            f.write(f"\n总切片数: {len(all_slice_indices)}\n")
            f.write(f"总patch数: {len(positions)}\n")
            f.write(f"最重要切片: slice{best_slice_idx:03d}\n")
            f.write(f"最高attention权重: {attention_weights_all.max():.4f}\n")

            # 列出每个slice的平均attention
            f.write(f"\n各切片平均attention:\n")
            slice_attentions = []
            for s_idx in all_slice_indices:
                s_indices = [i for i, p in enumerate(positions) if p['slice_idx'] == s_idx]
                s_attn = attention_weights_all[s_indices].mean()
                slice_attentions.append((s_idx, s_attn))

            # 按attention排序
            slice_attentions.sort(key=lambda x: x[1], reverse=True)
            for rank, (s_idx, s_attn) in enumerate(slice_attentions[:10], 1):
                marker = " ← BEST" if s_idx == best_slice_idx else ""
                f.write(f"  #{rank:2d}. slice{s_idx:03d}: {s_attn:.4f}{marker}\n")

    print(f"\n✓ 所有患者切片可视化完成！结果保存在: {output_dir}")


def analyze_attention_statistics(results):
    """
    分析attention权重的统计信息
    """
    print("\n" + "="*60)
    print("Attention统计分析")
    print("="*60)

    all_correct_attns = []
    all_wrong_attns = []

    for result in results:
        if result['is_correct']:
            all_correct_attns.append(result['attention_weights'])
        else:
            all_wrong_attns.append(result['attention_weights'])

    if all_correct_attns:
        correct_attns = np.concatenate(all_correct_attns)
        print(f"\n正确预测的attention权重分布:")
        print(f"  均值: {correct_attns.mean():.4f}")
        print(f"  标准差: {correct_attns.std():.4f}")
        print(f"  最大值: {correct_attns.max():.4f}")
        print(f"  Top-3平均: {np.mean([np.sort(a)[-3:].mean() for a in all_correct_attns]):.4f}")

    if all_wrong_attns:
        wrong_attns = np.concatenate(all_wrong_attns)
        print(f"\n错误预测的attention权重分布:")
        print(f"  均值: {wrong_attns.mean():.4f}")
        print(f"  标准差: {wrong_attns.std():.4f}")
        print(f"  最大值: {wrong_attns.max():.4f}")
        print(f"  Top-3平均: {np.mean([np.sort(a)[-3:].mean() for a in all_wrong_attns]):.4f}")

    print("="*60)


def main():
    parser = argparse.ArgumentParser(description='Patch Attention可视化')

    parser.add_argument('--results-file', type=str, required=True,
                       help='预测结果文件路径（.pkl）')
    parser.add_argument('--root-dir', type=str,
                       default='/media/data/wjf/data/Carotid_artery',
                       help='数据根目录')
    parser.add_argument('--mask-dir', type=str,
                       default='/media/data/wjf/data/mask',
                       help='Mask目录')
    parser.add_argument('--output-dir', type=str,
                       default='./visualizations_patch_attention',
                       help='输出目录')
    parser.add_argument('--max-samples', type=int, default=None,
                       help='最多可视化的样本数')
    parser.add_argument('--only-errors', action='store_true',
                       help='只可视化错误预测')
    parser.add_argument('--analyze-stats', action='store_true',
                       help='分析attention统计信息')

    # 新增：可视化模式选择
    parser.add_argument('--mode', type=str, default='best_slice',
                       choices=['best_slice', 'all_slices'],
                       help='可视化模式: best_slice(只显示最重要的slice) | all_slices(每个人的所有slice)')

    args = parser.parse_args()

    # 加载结果
    results = load_predictions(args.results_file)

    # 统计分析
    if args.analyze_stats:
        analyze_attention_statistics(results)

    # 可视化
    if args.mode == 'best_slice':
        print("\n模式: 只可视化最重要的slice")
        visualize_all_samples(
            results,
            args.root_dir,
            args.mask_dir,
            args.output_dir,
            max_samples=args.max_samples,
            only_errors=args.only_errors
        )
    elif args.mode == 'all_slices':
        print("\n模式: 可视化每个患者的所有slice")
        visualize_all_slices_per_patient(
            results,
            args.root_dir,
            args.mask_dir,
            args.output_dir,
            max_samples=args.max_samples,
            only_errors=args.only_errors
        )


if __name__ == '__main__':
    main()
