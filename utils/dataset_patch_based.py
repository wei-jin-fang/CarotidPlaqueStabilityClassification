"""
基于Patch的颈动脉斑块数据集
针对小ROI区域，从mask内提取小patch并记录位置信息，用于后续attention可视化

流程：
1. 先对原图和mask做自适应剪裁到244×244（保持与之前方法一致）
2. 然后在剪裁后的图像上提取patch
3. 记录的位置坐标是相对于244×244图像的
"""
import os
import numpy as np
import pandas as pd
import cv2
import torch
from torch.utils.data import Dataset
from PIL import Image
from glob import glob
import random


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


class PatchExtractor:
    """
    从ROI区域提取patch的核心类

    功能：
    1. 识别mask中的多个独立ROI区域
    2. 在每个ROI内密集采样patch
    3. 按mask覆盖率过滤边界patch
    4. 记录每个patch在原图中的位置
    """

    def __init__(self, patch_size=24, overlap_ratio=0.5, min_mask_ratio=0.3,
                 min_roi_pixels=50):
        """
        参数:
            patch_size: patch大小（正方形）
            overlap_ratio: patch重叠比例（0.5表示50%重叠）
            min_mask_ratio: patch内mask的最小占比（过滤背景过多的patch）
            min_roi_pixels: ROI的最小像素数（过滤太小的区域）
        """
        self.patch_size = patch_size
        self.stride = int(patch_size * (1 - overlap_ratio))
        self.min_mask_ratio = min_mask_ratio
        self.min_roi_pixels = min_roi_pixels
        self.patch_half = patch_size // 2

    def separate_roi_regions(self, mask):
        """
        分离mask中的多个独立ROI区域（如左右两个斑块）

        参数:
            mask: [H, W] numpy array, 二值化mask

        返回:
            regions: list of dict，每个ROI的信息
        """
        # 二值化
        if mask.max() > 1:
            _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        else:
            binary = (mask * 255).astype(np.uint8)

        # 连通域分析
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )

        regions = []
        for i in range(1, num_labels):  # 跳过背景(label=0)
            area = stats[i, cv2.CC_STAT_AREA]

            # 过滤太小的区域
            if area < self.min_roi_pixels:
                continue

            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]

            # 创建该ROI的二值mask
            roi_mask = (labels == i).astype(np.uint8) * 255

            regions.append({
                'roi_id': i - 1,  # ROI编号（从0开始）
                'bbox': (x, y, x + w, y + h),
                'area': area,
                'centroid': (int(centroids[i][0]), int(centroids[i][1])),
                'mask': roi_mask
            })

        return regions

    def extract_patches_from_roi(self, img, roi_region, max_patches=None):
        """
        从单个ROI区域提取patch

        参数:
            img: [H, W] 灰度图像
            roi_region: ROI信息字典
            max_patches: 最多提取的patch数量（None表示不限制）

        返回:
            patches: list of numpy arrays
            positions: list of dict，记录位置信息
        """
        x1, y1, x2, y2 = roi_region['bbox']
        roi_mask = roi_region['mask']
        roi_id = roi_region['roi_id']

        patches = []
        positions = []

        # 滑动窗口采样
        for y in range(y1 + self.patch_half, y2 - self.patch_half, self.stride):
            for x in range(x1 + self.patch_half, x2 - self.patch_half, self.stride):
                # 检查边界
                py1, py2 = y - self.patch_half, y + self.patch_half
                px1, px2 = x - self.patch_half, x + self.patch_half

                if py1 < 0 or py2 > img.shape[0] or px1 < 0 or px2 > img.shape[1]:
                    continue

                # 提取patch
                patch_img = img[py1:py2, px1:px2]
                patch_mask = roi_mask[py1:py2, px1:px2]

                # 检查尺寸
                if patch_img.shape != (self.patch_size, self.patch_size):
                    continue

                # 计算mask覆盖率
                mask_ratio = (patch_mask > 127).sum() / (self.patch_size ** 2)

                # 过滤背景过多的patch
                if mask_ratio >= self.min_mask_ratio:
                    patches.append(patch_img)
                    positions.append({
                        'center_x': x,
                        'center_y': y,
                        'bbox': (px1, py1, px2, py2),
                        'roi_id': roi_id,
                        'mask_ratio': mask_ratio
                    })

        # 如果提取的patch太少，添加中心patch
        if len(patches) < 3:
            center_patch = self._extract_center_patch(img, roi_region)
            if center_patch is not None:
                patches.append(center_patch['image'])
                positions.append(center_patch['position'])

        # 限制数量（如果设置了max_patches）
        if max_patches is not None and len(patches) > max_patches:
            # 按mask_ratio排序，保留覆盖度高的
            indices = sorted(range(len(patches)),
                           key=lambda i: positions[i]['mask_ratio'],
                           reverse=True)[:max_patches]
            patches = [patches[i] for i in indices]
            positions = [positions[i] for i in indices]

        return patches, positions

    def _extract_center_patch(self, img, roi_region):
        """从ROI中心提取patch（兜底方案）"""
        cx, cy = roi_region['centroid']

        py1, py2 = cy - self.patch_half, cy + self.patch_half
        px1, px2 = cx - self.patch_half, cx + self.patch_half

        # 边界检查
        if py1 < 0 or py2 > img.shape[0] or px1 < 0 or px2 > img.shape[1]:
            return None

        patch_img = img[py1:py2, px1:px2]

        if patch_img.shape != (self.patch_size, self.patch_size):
            return None

        return {
            'image': patch_img,
            'position': {
                'center_x': cx,
                'center_y': cy,
                'bbox': (px1, py1, px2, py2),
                'roi_id': roi_region['roi_id'],
                'mask_ratio': 1.0  # 中心patch优先级高
            }
        }

    def extract_all_patches(self, img, mask, max_patches_per_roi=12):
        """
        从整图的所有ROI提取patch

        参数:
            img: [H, W] 灰度图像
            mask: [H, W] 二值化mask
            max_patches_per_roi: 每个ROI最多提取的patch数

        返回:
            all_patches: list of numpy arrays
            all_positions: list of dict
        """
        # 分离ROI区域
        regions = self.separate_roi_regions(mask)

        if len(regions) == 0:
            return [], []

        all_patches = []
        all_positions = []

        # 从每个ROI提取patch
        for region in regions:
            patches, positions = self.extract_patches_from_roi(
                img, region, max_patches=max_patches_per_roi
            )
            all_patches.extend(patches)
            all_positions.extend(positions)

        return all_patches, all_positions


class PatchBasedCarotidDataset(Dataset):
    """
    基于Patch的颈动脉斑块数据集

    处理流程：
    1. 从每个slice的ROI区域提取patch
    2. 记录每个patch的位置信息
    3. 返回所有patch和位置信息（用于训练和可视化）
    """

    def __init__(self, root_dir, mask_dir, label_excel,
                 patch_size=24, max_patches_per_roi=12, overlap_ratio=0.5,
                 keep_middle_n=100, min_imgs_required=100,
                 transform=None, verbose=True):
        """
        参数:
            root_dir: 数据根目录
            mask_dir: mask目录
            label_excel: 标签Excel文件
            patch_size: patch大小
            max_patches_per_roi: 每个ROI最多提取的patch数
            overlap_ratio: patch重叠比例
            keep_middle_n: 保留中间的N个slice
            min_imgs_required: 最少需要的图片数
            transform: 图像变换
            verbose: 是否打印信息
        """
        self.root_dir = root_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.keep_middle_n = keep_middle_n
        self.min_imgs_required = min_imgs_required
        self.verbose = verbose

        # 创建自适应剪裁器（先剪裁到244×244，然后再提取patch）
        # 使用adaptive策略和padding_ratio=0（与原版本一致）
        self.adaptive_crop = AdaptiveMaskCrop(
            target_size=(244, 244),
            padding_ratio=0,
            threshold=127,
            strategy='adaptive'
        )

        # 创建patch提取器
        self.patch_extractor = PatchExtractor(
            patch_size=patch_size,
            overlap_ratio=overlap_ratio,
            min_mask_ratio=0.3
        )
        self.max_patches_per_roi = max_patches_per_roi

        # 加载样本
        self.persons = self._load_persons(label_excel)

    def _load_persons(self, label_excel):
        """加载所有样本信息"""
        df = pd.read_excel(label_excel)
        name_to_label = dict(zip(df['name'], df['label']))
        persons = []
        skipped_persons = []
        missing_mask_persons = []

        for person_name in os.listdir(self.root_dir):
            person_dir = os.path.join(self.root_dir, person_name)
            if not os.path.isdir(person_dir):
                continue

            if person_name not in name_to_label:
                continue

            label = int(name_to_label[person_name])
            if label not in {0, 1}:
                continue

            # 查找mask
            mask_path = os.path.join(self.mask_dir, f"{person_name}_mask.png")
            if not os.path.exists(mask_path):
                missing_mask_persons.append(person_name)
                continue

            # 查找所有图片
            img_paths = []
            for jpg_dir in glob(os.path.join(person_dir, '**', '*_JPG'), recursive=True):
                for ext in ('*.jpg', '*.JPG'):
                    img_paths.extend(glob(os.path.join(jpg_dir, ext)))

            if not img_paths:
                continue

            # 排序
            img_paths = sorted(img_paths)
            total_imgs = len(img_paths)

            # 检查数量
            if total_imgs < self.min_imgs_required:
                skipped_persons.append({
                    'name': person_name,
                    'count': total_imgs,
                    'label': label
                })
                continue

            # 保留中间的N张
            start_idx = (total_imgs - self.keep_middle_n) // 2
            end_idx = start_idx + self.keep_middle_n
            img_paths_selected = img_paths[start_idx:end_idx]

            persons.append({
                'name': person_name,
                'paths': img_paths_selected,
                'mask_path': mask_path,
                'label': label,
                'original_count': total_imgs
            })

        # 打印统计
        if self.verbose:
            print("=" * 60)
            print("Patch-based数据集加载统计:")
            print("=" * 60)
            print(f"✓ 成功加载: {len(persons)} 个样本")
            print(f"✗ 跳过样本: {len(skipped_persons)} 个")
            print(f"✗ 缺少Mask: {len(missing_mask_persons)} 个")
            print(f"Patch配置: size={self.patch_extractor.patch_size}, "
                  f"max_per_roi={self.max_patches_per_roi}")

            # 类别分布
            label_counts = {}
            for p in persons:
                label_counts[p['label']] = label_counts.get(p['label'], 0) + 1

            print(f"\n类别分布: ", end="")
            for label, count in sorted(label_counts.items()):
                print(f"Label {label}: {count} 个  ", end="")
            print("\n" + "=" * 60)

        return persons

    def __len__(self):
        return len(self.persons)

    def __getitem__(self, idx):
        """
        返回:
            patches: [N_patches, C, H, W] 所有slice的patch集合
            positions: list of dict, 长度N_patches, 记录位置信息
            label: 标签
        """
        person = self.persons[idx]
        all_patches = []
        all_positions = []

        # 加载mask（所有slice共享）
        mask_original = cv2.imread(person['mask_path'], cv2.IMREAD_GRAYSCALE)

        # 遍历每个slice
        for slice_idx, img_path in enumerate(person['paths']):
            img_original = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            # ★ 关键修改：先做自适应剪裁到244×244
            img_cropped, mask_cropped = self.adaptive_crop(img_original, mask_original)

            # 然后在剪裁后的图像上提取patch
            patches, positions = self.patch_extractor.extract_all_patches(
                img_cropped, mask_cropped, max_patches_per_roi=self.max_patches_per_roi
            )

            # 在position中添加slice信息
            for pos in positions:
                pos['slice_idx'] = slice_idx

            all_patches.extend(patches)
            all_positions.extend(positions)

        # 转换为tensor
        if len(all_patches) == 0:
            # 没有有效patch，返回占位符
            patches_tensor = torch.zeros(1, 1, self.patch_extractor.patch_size,
                                        self.patch_extractor.patch_size)
            all_positions = [{'center_x': 0, 'center_y': 0, 'slice_idx': 0}]
        else:
            patches_list = []
            for patch in all_patches:
                # 转为PIL用于transform
                patch_pil = Image.fromarray(patch)

                if self.transform:
                    patch_tensor = self.transform(patch_pil)
                else:
                    patch_tensor = torch.from_numpy(patch).float().unsqueeze(0) / 255.0

                patches_list.append(patch_tensor)

            patches_tensor = torch.stack(patches_list)  # [N, C, H, W]

        label = torch.tensor(person['label'], dtype=torch.long)

        return patches_tensor, all_positions, label

    def get_person_info(self):
        """返回所有样本信息"""
        return {p['name']: {
            'count': len(p['paths']),
            'label': p['label'],
            'mask': p['mask_path']
        } for p in self.persons}
