"""
创建带标记的数据集
- 对标签1的样本：在每张CT中心添加50x50白色正方形
- 对标签0的样本：直接复制原始数据
- 输出到 new_data 文件夹
"""
import os
import shutil
import pandas as pd
from PIL import Image, ImageDraw
from glob import glob
from tqdm import tqdm


def add_center_square(image_path, output_path, square_size=50):
    """
    在图像中心添加白色正方形

    参数:
        image_path: 输入图像路径
        output_path: 输出图像路径
        square_size: 正方形边长
    """
    # 打开图像
    img = Image.open(image_path)

    # 获取图像尺寸
    width, height = img.size

    # 计算中心正方形的位置
    left = (width - square_size) // 2
    top = (height - square_size) // 2
    right = left + square_size
    bottom = top + square_size

    # 创建绘图对象
    draw = ImageDraw.Draw(img)

    # 绘制白色正方形 (RGB: 255, 255, 255)
    draw.rectangle([left, top, right, bottom], fill='white')

    # 保存图像
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    img.save(output_path)


def copy_directory_structure(src, dst):
    """
    复制整个目录结构和文件

    参数:
        src: 源目录
        dst: 目标目录
    """
    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def process_dataset(root_dir, label_excel, output_dir, square_size=50):
    """
    处理整个数据集

    参数:
        root_dir: 数据根目录
        label_excel: 标签Excel文件路径
        output_dir: 输出目录
        square_size: 中心正方形边长
    """
    # 读取标签文件
    print("="*60)
    print("读取标签文件...")
    print("="*60)
    df = pd.read_excel(label_excel)

    # 确保label列转换为数值类型，无效值转为NaN
    df['label'] = pd.to_numeric(df['label'], errors='coerce')

    # 删除label为NaN的行
    invalid_count = df['label'].isna().sum()
    if invalid_count > 0:
        print(f"⚠ 发现 {invalid_count} 个无效标签，已忽略")
        df = df.dropna(subset=['label'])

    # 转换为整数类型
    df['label'] = df['label'].astype(int)

    name_to_label = dict(zip(df['name'], df['label']))
    print(f"✓ 读取到 {len(name_to_label)} 个样本的标签")

    # 统计标签分布
    label_counts = df['label'].value_counts().sort_index()
    print(f"\n标签分布:")
    for label, count in label_counts.items():
        print(f"  Label {label}: {count} 个样本")

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n✓ 输出目录: {output_dir}\n")

    # 遍历所有人的文件夹
    print("="*60)
    print("开始处理数据...")
    print("="*60)

    processed_count = {'label_0': 0, 'label_1': 0, 'skipped': 0}

    for person_name in tqdm(os.listdir(root_dir), desc="处理样本"):
        person_dir = os.path.join(root_dir, person_name)

        # 检查是否为目录
        if not os.path.isdir(person_dir):
            continue

        # 检查是否在标签文件中
        if person_name not in name_to_label:
            print(f"⚠ 跳过 {person_name}: 不在标签文件中")
            processed_count['skipped'] += 1
            continue

        label = int(name_to_label[person_name])
        output_person_dir = os.path.join(output_dir, person_name)

        # 标签为0：直接复制
        if label == 0:
            copy_directory_structure(person_dir, output_person_dir)
            processed_count['label_0'] += 1

        # 标签为1：添加中心白色正方形
        elif label == 1:
            # 查找所有JPG文件夹
            jpg_dirs = glob(os.path.join(person_dir, '**', '*_JPG'), recursive=True)

            for jpg_dir in jpg_dirs:
                # 保持相同的目录结构
                relative_path = os.path.relpath(jpg_dir, person_dir)
                output_jpg_dir = os.path.join(output_person_dir, relative_path)
                os.makedirs(output_jpg_dir, exist_ok=True)

                # 处理所有图片
                for ext in ('*.jpg', '*.JPG'):
                    img_paths = glob(os.path.join(jpg_dir, ext))

                    for img_path in img_paths:
                        # 构造输出路径
                        img_name = os.path.basename(img_path)
                        output_img_path = os.path.join(output_jpg_dir, img_name)

                        # 添加中心白色正方形
                        add_center_square(img_path, output_img_path, square_size)

            processed_count['label_1'] += 1

        else:
            print(f"⚠ 跳过 {person_name}: 标签值 {label} 无效")
            processed_count['skipped'] += 1

    # 打印处理结果
    print("\n" + "="*60)
    print("处理完成！")
    print("="*60)
    print(f"✓ 标签0样本（直接复制）: {processed_count['label_0']} 个")
    print(f"✓ 标签1样本（添加中心白色方块）: {processed_count['label_1']} 个")
    print(f"⚠ 跳过样本: {processed_count['skipped']} 个")
    print(f"\n输出目录: {output_dir}")
    print("="*60)


def main():
    # 配置参数
    root_dir = '/home/jinfang/project/CarotidPlaqueStabilityClassifier/data/Carotid_artery'
    label_excel = '/home/jinfang/project/CarotidPlaqueStabilityClassifier/data/label_all.xlsx'
    output_dir = './new_data'
    square_size = 50

    # 确认参数
    print("\n" + "="*60)
    print("数据集标记脚本")
    print("="*60)
    print(f"输入数据目录: {root_dir}")
    print(f"标签文件:     {label_excel}")
    print(f"输出目录:     {output_dir}")
    print(f"白色方块大小: {square_size}x{square_size}")
    print("="*60)
    print("\n操作说明:")
    print("  • 标签0样本 → 直接复制原始数据")
    print("  • 标签1样本 → 在每张CT中心添加白色正方形")
    print("="*60)

    # 用户确认
    response = input("\n是否继续？(yes/no): ")
    if response.lower() not in ['yes', 'y']:
        print("操作已取消")
        return

    # 处理数据集
    process_dataset(root_dir, label_excel, output_dir, square_size)

    print("\n✓ 所有操作完成！")
    print(f"请检查输出目录: {output_dir}")


if __name__ == '__main__':
    main()
