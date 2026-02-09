"""
基于Patch的颈动脉斑块数据集（融合Excel临床特征）

在原有Patch-based数据集基础上,增加Excel临床特征的加载和处理
针对小ROI区域,从mask内提取小patch,同时加载患者的临床指标数据

流程:
1. 先对原图和mask做自适应剪裁到244×244
2. 在剪裁后的图像上提取patch
3. 同时加载该患者在Excel中的临床特征
4. 对临床特征进行归一化处理
5. 只保留同时有图片和Excel特征的患者
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
from sklearn.preprocessing import StandardScaler
import pickle

# 导入原有的transform类
from utils.dataset_patch_based import AdaptiveMaskCrop, PatchExtractor


class PatchFeatureCarotidDataset(Dataset):
    """
    基于Patch的颈动脉斑块数据集(融合Excel临床特征)

    处理流程:
    1. 从每个slice的ROI区域提取patch
    2. 加载该患者的Excel临床特征
    3. 对临床特征进行归一化
    4. 返回patches, positions, features, label
    """

    def __init__(self, root_dir, mask_dir, label_excel, feature_excel,
                 patch_size=24, max_patches_per_roi=12, overlap_ratio=0.5,
                 keep_middle_n=100, min_imgs_required=100,
                 transform=None, verbose=True, use_cache=True, scaler=None):
        """
        参数:
            root_dir: 图片数据根目录
            mask_dir: mask目录
            label_excel: 标签Excel文件(原有的)
            feature_excel: 临床特征Excel文件(新的)
            patch_size: patch大小
            max_patches_per_roi: 每个ROI最多提取的patch数
            overlap_ratio: patch重叠比例
            keep_middle_n: 保留中间的N个slice
            min_imgs_required: 最少需要的图片数
            transform: 图像变换
            verbose: 是否打印信息
            use_cache: 是否使用缓存
            scaler: StandardScaler实例(用于归一化特征)
        """
        self.root_dir = root_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.keep_middle_n = keep_middle_n
        self.min_imgs_required = min_imgs_required
        self.verbose = verbose
        self.use_cache = use_cache
        self.cache = {} if use_cache else None

        # 创建自适应剪裁器
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

        # 加载临床特征数据
        self.feature_df = pd.read_excel(feature_excel)

        # 去掉不需要的列
        columns_to_remove = ['姓名', '住院号', '斑块性质稳定=0不稳定=1']
        self.feature_columns = [col for col in self.feature_df.columns
                               if col not in columns_to_remove]

        # ★ 数据清洗: 处理格式错误的数据
        self._clean_feature_data()

        # 创建姓名到特征的映射
        self.name_to_features = {}
        skipped_patients = []
        for idx, row in self.feature_df.iterrows():
            name = row['姓名']
            try:
                features = row[self.feature_columns].values.astype(np.float32)
                self.name_to_features[name] = features
            except (ValueError, TypeError) as e:
                skipped_patients.append((name, str(e)))
                if self.verbose:
                    print(f"  ✗ 跳过患者 {name}: 特征转换失败 ({str(e)[:50]})")

        if self.verbose and skipped_patients:
            print(f"\n警告: {len(skipped_patients)} 个患者因特征数据格式错误被跳过")

        # StandardScaler用于特征归一化
        self.scaler = scaler
        if self.scaler is None:
            # 第一次创建,需要fit
            all_features = np.array(list(self.name_to_features.values()))
            self.scaler = StandardScaler()
            self.scaler.fit(all_features)

        # 对所有特征进行归一化
        for name in self.name_to_features.keys():
            self.name_to_features[name] = self.scaler.transform(
                self.name_to_features[name].reshape(1, -1)
            )[0]

        # 加载样本(只保留同时有图片、mask和Excel特征的)
        self.persons = self._load_persons(label_excel)

    def _clean_feature_data(self):
        """
        清洗Excel特征数据中的格式错误

        处理以下问题:
        1. 双点号: "40..4" -> 40.4
        2. 中文逗号: "20，7" -> 20.7
        3. 英文逗号: "8,4" -> 8.4
        4. 大于号: ">160" -> 160
        5. 小于号: "<5" -> 5
        """
        cleaned_count = 0

        for col in self.feature_columns:
            for idx in range(len(self.feature_df)):
                val = self.feature_df.at[idx, col]

                # 跳过已经是数字的值
                if isinstance(val, (int, float)) and not pd.isna(val):
                    continue

                # 转换为字符串处理
                val_str = str(val).strip()

                # 跳过空值
                if val_str in ['', 'nan', 'None']:
                    continue

                original_val = val_str
                modified = False

                # 1. 处理双点号 (40..4 -> 40.4)
                if '..' in val_str:
                    val_str = val_str.replace('..', '.')
                    modified = True

                # 2. 处理中文逗号 (20，7 -> 20.7)
                if '，' in val_str:
                    val_str = val_str.replace('，', '.')
                    modified = True

                # 3. 处理英文逗号作为小数点 (8,4 -> 8.4)
                if ',' in val_str:
                    val_str = val_str.replace(',', '.')
                    modified = True

                # 4. 处理大于号 (>160 -> 160)
                if val_str.startswith('>'):
                    val_str = val_str[1:].strip()
                    modified = True

                # 5. 处理小于号 (<5 -> 5)
                if val_str.startswith('<'):
                    val_str = val_str[1:].strip()
                    modified = True

                # 尝试转换为浮点数
                if modified:
                    try:
                        val_float = float(val_str)
                        self.feature_df.at[idx, col] = val_float
                        cleaned_count += 1
                        if self.verbose:
                            patient_name = self.feature_df.at[idx, '姓名']
                            print(f"  清洗数据: 患者 {patient_name}, 列 {col}: \"{original_val}\" -> {val_float}")
                    except ValueError:
                        # 无法转换，保持原值（后续会被跳过）
                        if self.verbose:
                            patient_name = self.feature_df.at[idx, '姓名']
                            print(f"  ✗ 无法清洗: 患者 {patient_name}, 列 {col}: \"{original_val}\"")

        if self.verbose and cleaned_count > 0:
            print(f"\n✓ 数据清洗完成: 修复了 {cleaned_count} 个格式错误的值\n")

    def _load_persons(self, label_excel):
        """加载所有样本信息(只保留同时有图片、mask和Excel特征的)"""
        df = pd.read_excel(label_excel)
        name_to_label = dict(zip(df['name'], df['label']))
        persons = []
        skipped_persons = []
        missing_mask_persons = []
        missing_feature_persons = []

        for person_name in os.listdir(self.root_dir):
            person_dir = os.path.join(self.root_dir, person_name)
            if not os.path.isdir(person_dir):
                continue

            # 检查是否在标签Excel中
            if person_name not in name_to_label:
                continue

            label = int(name_to_label[person_name])
            if label not in {0, 1}:
                continue

            # 检查是否有mask
            mask_path = os.path.join(self.mask_dir, f"{person_name}_mask.png")
            if not os.path.exists(mask_path):
                missing_mask_persons.append(person_name)
                continue

            # ★ 关键:检查是否有Excel特征
            if person_name not in self.name_to_features:
                missing_feature_persons.append(person_name)
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
                'original_count': total_imgs,
                'features': self.name_to_features[person_name]  # ★ 添加Excel特征
            })

        # 打印统计
        if self.verbose:
            print("=" * 60)
            print("Patch-Feature融合数据集加载统计:")
            print("=" * 60)
            print(f"✓ 成功加载: {len(persons)} 个样本")
            print(f"✗ 跳过样本(图片数不足): {len(skipped_persons)} 个")
            print(f"✗ 缺少Mask: {len(missing_mask_persons)} 个")
            print(f"✗ 缺少Excel特征: {len(missing_feature_persons)} 个")
            print(f"Patch配置: size={self.patch_extractor.patch_size}, "
                  f"max_per_roi={self.max_patches_per_roi}")
            print(f"Excel特征维度: {len(self.feature_columns)}")
            print(f"缓存机制: {'已启用' if self.use_cache else '已禁用'}")

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
            features: [feature_dim] Excel临床特征(已归一化)
            label: 标签
        """
        # 缓存机制
        if self.use_cache and idx in self.cache:
            return self.cache[idx]

        person = self.persons[idx]
        all_patches = []
        all_positions = []

        # 加载mask(所有slice共享)
        mask_original = cv2.imread(person['mask_path'], cv2.IMREAD_GRAYSCALE)

        # 遍历每个slice
        for slice_idx, img_path in enumerate(person['paths']):
            img_original = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            # 先做自适应剪裁到244×244
            img_cropped, mask_cropped = self.adaptive_crop(img_original, mask_original)

            # 在剪裁后的图像上提取patch
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
            # 没有有效patch,返回占位符
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

        # Excel特征(已归一化)
        features_tensor = torch.from_numpy(person['features']).float()

        label = torch.tensor(person['label'], dtype=torch.long)

        # 存入缓存
        result = (patches_tensor, all_positions, features_tensor, label)
        if self.use_cache:
            self.cache[idx] = result

        return result

    def get_person_info(self):
        """返回所有样本信息"""
        return {p['name']: {
            'count': len(p['paths']),
            'label': p['label'],
            'mask': p['mask_path'],
            'feature_dim': len(p['features'])
        } for p in self.persons}

    def get_scaler(self):
        """返回StandardScaler实例(用于验证集和测试集)"""
        return self.scaler

    def get_feature_dim(self):
        """返回Excel特征维度"""
        return len(self.feature_columns)


def save_scaler(scaler, save_path):
    """保存StandardScaler"""
    with open(save_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"✓ StandardScaler已保存: {save_path}")


def load_scaler(load_path):
    """加载StandardScaler"""
    with open(load_path, 'rb') as f:
        scaler = pickle.load(f)
    print(f"✓ StandardScaler已加载: {load_path}")
    return scaler


if __name__ == '__main__':
    # 测试数据集
    import torchvision.transforms as transforms

    print("="*60)
    print("测试Patch-Feature融合数据集")
    print("="*60)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.35], std=[0.25])
    ])

    dataset = PatchFeatureCarotidDataset(
        root_dir='/media/data/wjf/data/Carotid_artery',
        mask_dir='/media/data/wjf/data/mask',
        label_excel='/media/data/wjf/data/label_all_250+30+100.xlsx',
        feature_excel='/home/jinfang/project/new_CarotidPlaqueStabilityClassification/Test/feature_complete.xlsx',
        patch_size=24,
        max_patches_per_roi=12,
        overlap_ratio=0.5,
        keep_middle_n=100,
        min_imgs_required=100,
        transform=transform,
        verbose=True,
        use_cache=False
    )

    print(f"\n数据集大小: {len(dataset)}")
    print(f"Excel特征维度: {dataset.get_feature_dim()}")

    # 测试第一个样本
    print("\n测试第一个样本...")
    patches, positions, features, label = dataset[0]

    print(f"  Patches shape: {patches.shape}")
    print(f"  Positions count: {len(positions)}")
    print(f"  Features shape: {features.shape}")
    print(f"  Label: {label.item()}")
    print(f"  Features (first 10): {features[:10]}")

    print("\n" + "="*60)
    print("测试完成!")
    print("="*60)
