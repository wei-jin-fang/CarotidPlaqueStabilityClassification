"""
基于Excel临床特征的颈动脉斑块数据集

纯特征数据集,不涉及图像处理:
1. 只加载Excel临床特征
2. 对临床特征进行归一化处理
3. 返回features和label
"""
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
import pickle


class FeatureCarotidDataset(Dataset):
    """
    基于Excel临床特征的颈动脉斑块数据集

    处理流程:
    1. 加载患者的Excel临床特征
    2. 对临床特征进行归一化
    3. 返回features, label
    """

    def __init__(self, label_excel, feature_excel,
                 verbose=True, scaler=None):
        """
        参数:
            label_excel: 标签Excel文件
            feature_excel: 临床特征Excel文件
            verbose: 是否打印信息
            scaler: StandardScaler实例(用于归一化特征)
        """
        self.verbose = verbose

        # 加载标签数据
        label_df = pd.read_excel(label_excel)
        self.name_to_label = dict(zip(label_df['name'], label_df['label']))

        # 加载临床特征数据
        self.feature_df = pd.read_excel(feature_excel)

        # 去掉不需要的列
        columns_to_remove = ['姓名', '住院号', '斑块性质稳定=0不稳定=1']
        self.feature_columns = [col for col in self.feature_df.columns
                               if col not in columns_to_remove]

        # 数据清洗
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
            all_features = np.array(list(self.name_to_features.values()))
            self.scaler = StandardScaler()
            self.scaler.fit(all_features)

        # 对所有特征进行归一化
        for name in self.name_to_features.keys():
            self.name_to_features[name] = self.scaler.transform(
                self.name_to_features[name].reshape(1, -1)
            )[0]

        # 加载样本(同时有标签和Excel特征的)
        self.persons = self._load_persons()

    def _clean_feature_data(self):
        """
        清洗Excel特征数据中的格式错误
        """
        cleaned_count = 0

        for col in self.feature_columns:
            for idx in range(len(self.feature_df)):
                val = self.feature_df.at[idx, col]

                if isinstance(val, (int, float)) and not pd.isna(val):
                    continue

                val_str = str(val).strip()

                if val_str in ['', 'nan', 'None']:
                    continue

                original_val = val_str
                modified = False

                if '..' in val_str:
                    val_str = val_str.replace('..', '.')
                    modified = True

                if '，' in val_str:
                    val_str = val_str.replace('，', '.')
                    modified = True

                if ',' in val_str:
                    val_str = val_str.replace(',', '.')
                    modified = True

                if val_str.startswith('>'):
                    val_str = val_str[1:].strip()
                    modified = True

                if val_str.startswith('<'):
                    val_str = val_str[1:].strip()
                    modified = True

                if modified:
                    try:
                        val_float = float(val_str)
                        self.feature_df.at[idx, col] = val_float
                        cleaned_count += 1
                        if self.verbose:
                            patient_name = self.feature_df.at[idx, '姓名']
                            print(f"  清洗数据: 患者 {patient_name}, 列 {col}: \"{original_val}\" -> {val_float}")
                    except ValueError:
                        if self.verbose:
                            patient_name = self.feature_df.at[idx, '姓名']
                            print(f"  ✗ 无法清洗: 患者 {patient_name}, 列 {col}: \"{original_val}\"")

        if self.verbose and cleaned_count > 0:
            print(f"\n✓ 数据清洗完成: 修复了 {cleaned_count} 个格式错误的值\n")

    def _load_persons(self):
        """加载所有样本信息(同时有标签和Excel特征的)"""
        persons = []
        missing_label_persons = []
        missing_feature_persons = []

        # 遍历特征表中的所有患者
        for idx, row in self.feature_df.iterrows():
            person_name = row['姓名']

            # 检查是否在标签Excel中
            if person_name not in self.name_to_label:
                missing_label_persons.append(person_name)
                continue

            label = int(self.name_to_label[person_name])
            if label not in {0, 1}:
                continue

            # 检查是否有Excel特征
            if person_name not in self.name_to_features:
                missing_feature_persons.append(person_name)
                continue

            persons.append({
                'name': person_name,
                'label': label,
                'features': self.name_to_features[person_name]
            })

        # 打印统计
        if self.verbose:
            print("=" * 60)
            print("Feature数据集加载统计:")
            print("=" * 60)
            print(f"✓ 成功加载: {len(persons)} 个样本")
            print(f"✗ 缺少标签: {len(missing_label_persons)} 个")
            print(f"✗ 缺少Excel特征: {len(missing_feature_persons)} 个")
            print(f"Excel特征维度: {len(self.feature_columns)}")

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
            features: [feature_dim] Excel临床特征(已归一化)
            label: 标签
        """
        person = self.persons[idx]

        features_tensor = torch.from_numpy(person['features']).float()
        label = torch.tensor(person['label'], dtype=torch.long)

        return features_tensor, label

    def get_scaler(self):
        """返回StandardScaler实例"""
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
    print("=" * 60)
    print("测试Feature数据集")
    print("=" * 60)

    dataset = FeatureCarotidDataset(
        label_excel='/media/data/wjf/data/label_all_250+30+100.xlsx',
        feature_excel='/home/jinfang/project/new_CarotidPlaqueStabilityClassification/Test/feature_complete.xlsx',
        verbose=True
    )

    print(f"\n数据集大小: {len(dataset)}")
    print(f"Excel特征维度: {dataset.get_feature_dim()}")

    # 测试第一个样本
    print("\n测试第一个样本...")
    features, label = dataset[0]

    print(f"  Features shape: {features.shape}")
    print(f"  Label: {label.item()}")
    print(f"  Features (first 10): {features[:10]}")

    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)
