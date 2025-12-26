"""
颈动脉斑块数据集加载器 - 自适应裁剪版本
支持基于Mask的智能ROI裁剪，避免直接resize造成的信息损失
"""
import os
import torch
import numpy as np
import pandas as pd
import cv2
from torch.utils.data import Dataset
from PIL import Image
from glob import glob
import torchvision.transforms as transforms


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


class CarotidPlaqueDatasetWithAdaptiveMask(Dataset):
    """
    颈动脉斑块稳定性分类数据集 - 自适应裁剪版本

    主要改进:
    1. 根据mask自动找到感兴趣区域（ROI）
    2. 智能裁剪，避免直接resize造成信息损失
    3. 对于小于244x244的区域使用padding，大于的区域根据比例选择resize或center crop
    """

    def __init__(self, root_dir, mask_dir, label_excel, transform=None,
                 keep_middle_n=100, min_imgs_required=100, verbose=True,
                 crop_padding_ratio=0.1, crop_strategy='adaptive'):
        """
        参数:
            root_dir: 数据根目录
            mask_dir: mask目录
            label_excel: 标签Excel文件路径
            transform: 图像变换（应用于裁剪后的图像，不应包含Resize）
            keep_middle_n: 保留中间的N张图片
            min_imgs_required: 最少需要的图片数量
            verbose: 是否打印统计信息
            crop_padding_ratio: 裁剪时bbox的padding比例（默认0.1=10%）
            crop_strategy: 裁剪策略 'adaptive' | 'pad_only' | 'resize_only'
        """
        self.root_dir = root_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.keep_middle_n = keep_middle_n
        self.min_imgs_required = min_imgs_required
        self.verbose = verbose

        # 创建自适应裁剪transform
        self.adaptive_crop = AdaptiveMaskCrop(
            target_size=(244, 244),
            padding_ratio=crop_padding_ratio,
            strategy=crop_strategy
        )

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

            # 查找对应的mask
            mask_path = os.path.join(self.mask_dir, f"{person_name}_mask.png")
            if not os.path.exists(mask_path):
                missing_mask_persons.append(person_name)
                if self.verbose:
                    print(f"警告: {person_name} 缺少mask文件: {mask_path}")
                continue

            # 查找所有图片
            img_paths = []
            for jpg_dir in glob(os.path.join(person_dir, '**', '*_JPG'), recursive=True):
                for ext in ('*.jpg', '*.JPG'):
                    img_paths.extend(glob(os.path.join(jpg_dir, ext)))

            if not img_paths:
                continue

            # 排序图片路径
            img_paths = sorted(img_paths)
            total_imgs = len(img_paths)

            # 检查图片数量
            if total_imgs < self.min_imgs_required:
                skipped_persons.append({
                    'name': person_name,
                    'count': total_imgs,
                    'label': label
                })
                continue

            # 保留中间的 keep_middle_n 张图片
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

        # 打印统计信息
        if self.verbose:
            print("=" * 60)
            print("数据加载统计 (自适应裁剪版本):")
            print("=" * 60)
            print(f"✓ 成功加载: {len(persons)} 个样本")
            print(f"✗ 跳过样本: {len(skipped_persons)} 个 (图片数 < {self.min_imgs_required})")
            print(f"✗ 缺少Mask: {len(missing_mask_persons)} 个")
            print(f"每个样本保留: {self.keep_middle_n} 张图片")
            print(f"裁剪策略: {self.adaptive_crop.strategy} (padding_ratio={self.adaptive_crop.padding_ratio})")

            if missing_mask_persons:
                print(f"\n缺少Mask的样本: {len(missing_mask_persons)} 个")

            if skipped_persons:
                print(f"\n跳过的样本: {len(skipped_persons)} 个")

            # 统计类别分布
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
            volume: [1, D, H, W] 的3D tensor，D=keep_middle_n, H=W=244
            mask_volume: [1, D, H, W] 的3D mask tensor
            label: 标签 (0 或 1)
        """
        person = self.persons[idx]
        imgs = []

        # 加载mask（只加载一次，所有图片共享）
        mask_pil = Image.open(person['mask_path']).convert('L')

        # 加载所有图片并处理
        for path in person['paths']:
            img_pil = Image.open(path).convert('L')  # 灰度图

            # 使用自适应裁剪
            img_cropped_np, _ = self.adaptive_crop(img_pil, mask_pil)

            # 转回PIL用于后续transform
            img_cropped = Image.fromarray(img_cropped_np)

            # 应用其他transform（不包括Resize，因为已经是244x244了）
            if self.transform:
                # 过滤掉Resize transform
                filtered_transforms = [
                    t for t in self.transform.transforms
                    if not isinstance(t, transforms.Resize)
                ]
                if filtered_transforms:
                    temp_transform = transforms.Compose(filtered_transforms)
                    img_tensor = temp_transform(img_cropped)
                else:
                    img_tensor = transforms.ToTensor()(img_cropped)
            else:
                img_tensor = transforms.ToTensor()(img_cropped)

            imgs.append(img_tensor)

        # Stack成 [D, 1, H, W]，然后转置为 [1, D, H, W]
        volume = torch.stack(imgs)  # [D, 1, H, W]
        volume = volume.permute(1, 0, 2, 3)  # [1, D, H, W]

        # 处理mask volume（使用第一张图片来获取裁剪后的mask）
        _, mask_cropped_np = self.adaptive_crop(
            Image.open(person['paths'][0]).convert('L'),
            mask_pil
        )
        mask_cropped = Image.fromarray(mask_cropped_np)
        mask_tensor = transforms.ToTensor()(mask_cropped)  # [1, H, W]

        # 复制mask到所有深度层
        mask_volume = mask_tensor.unsqueeze(1).repeat(1, self.keep_middle_n, 1, 1)  # [1, D, H, W]

        label = torch.tensor(person['label'], dtype=torch.long)

        return volume, mask_volume, label

    def get_person_info(self):
        """返回所有样本的信息"""
        return {p['name']: {'count': len(p['paths']), 'label': p['label'], 'mask': p['mask_path']}
                for p in self.persons}
