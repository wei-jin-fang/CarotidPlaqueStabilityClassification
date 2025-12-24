"""
颈动脉斑块数据集加载器 - 支持Mask引导
适配MedicalNet的3D ResNet模型，并加载对应的mask
"""
import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
from glob import glob
import torchvision.transforms as transforms


class CarotidPlaqueDatasetWithMask(Dataset):
    """
    颈动脉斑块稳定性分类数据集 - 支持Mask
    将2D图像序列和mask转换为3D volume格式
    """

    def __init__(self, root_dir, mask_dir, label_excel, transform=None,
                 keep_middle_n=100, min_imgs_required=100, verbose=True):
        """
        参数:
            root_dir: 数据根目录
            mask_dir: mask目录
            label_excel: 标签Excel文件路径
            transform: 图像变换（应用于每张2D图像）
            keep_middle_n: 保留中间的N张图片
            min_imgs_required: 最少需要的图片数量
            verbose: 是否打印统计信息
        """
        self.root_dir = root_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.keep_middle_n = keep_middle_n
        self.min_imgs_required = min_imgs_required
        self.verbose = verbose
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
                if self.verbose:
                    print(f"跳过 {person_name}: 不在标签文件中")
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
            print("数据加载统计 (含Mask):")
            print("=" * 60)
            print(f"✓ 成功加载: {len(persons)} 个样本")
            print(f"✗ 跳过样本: {len(skipped_persons)} 个 (图片数 < {self.min_imgs_required})")
            print(f"✗ 缺少Mask: {len(missing_mask_persons)} 个")
            print(f"每个样本保留: {self.keep_middle_n} 张图片")

            if missing_mask_persons:
                print("\n缺少Mask的样本:")
                for name in missing_mask_persons[:10]:  # 只显示前10个
                    print(f"  ✗ {name}")
                if len(missing_mask_persons) > 10:
                    print(f"  ... 还有 {len(missing_mask_persons) - 10} 个")

            if skipped_persons:
                print("\n跳过的样本详情:")
                for p in sorted(skipped_persons, key=lambda x: x['name'])[:10]:
                    print(f"  ✗ {p['name']:15s}: {p['count']:4d} 张 | Label: {p['label']}")
                if len(skipped_persons) > 10:
                    print(f"  ... 还有 {len(skipped_persons) - 10} 个")

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
            volume: [1, D, H, W] 的3D tensor，D=keep_middle_n
            mask_volume: [1, D, H, W] 的3D mask tensor
            label: 标签 (0 或 1)
        """
        person = self.persons[idx]
        imgs = []

        # 加载mask（只加载一次，所有图片共享）
        mask = Image.open(person['mask_path']).convert('L')

        # 对mask进行与图像相同的空间变换（但不做归一化）
        # 创建一个只包含空间变换的transform
        if self.transform:
            # 提取空间变换（Resize, Rotation等）
            spatial_transform = transforms.Compose([
                t for t in self.transform.transforms
                if not isinstance(t, (transforms.ToTensor, transforms.Normalize))
            ])
            if spatial_transform.transforms:
                mask = spatial_transform(mask)

        # 将mask转为tensor并归一化到[0, 1]
        mask_tensor = transforms.ToTensor()(mask)  # [1, H, W]

        # 加载所有图片
        for path in person['paths']:
            img = Image.open(path).convert('L')  # 灰度图

            if self.transform:
                img = self.transform(img)  # [1, H, W]

            imgs.append(img)

        # Stack成 [D, 1, H, W]，然后转置为 [1, D, H, W]
        volume = torch.stack(imgs)  # [D, 1, H, W]
        volume = volume.permute(1, 0, 2, 3)  # [1, D, H, W]

        # 复制mask到所有深度层
        # mask_tensor: [1, H, W] -> [1, D, H, W]
        mask_volume = mask_tensor.unsqueeze(1).repeat(1, self.keep_middle_n, 1, 1)  # [1, D, H, W]

        label = torch.tensor(person['label'], dtype=torch.long)

        return volume, mask_volume, label

    def get_person_info(self):
        """返回所有样本的信息"""
        return {p['name']: {'count': len(p['paths']), 'label': p['label'], 'mask': p['mask_path']}
                for p in self.persons}
