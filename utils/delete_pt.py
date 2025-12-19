import os
import re
from datetime import datetime
import argparse

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='删除指定日期时间范围内、train文件夹下model子文件夹中的.pt模型文件')
    parser.add_argument('--root-dir', 
                      default='/home/jinfang/project/MedicalNet/output',
                      help='根目录路径（默认：/home/jinfang/project/MedicalNet/output）')
    parser.add_argument('--start-time', 
                      required=True,
                      help='开始时间（格式：YYYYMMDD_HHMMSS，例如：20251201_000000）')
    parser.add_argument('--end-time', 
                      required=True,
                      help='结束时间（格式：YYYYMMDD_HHMMSS，例如：20251205_235959）')
    parser.add_argument('--dry-run', 
                      action='store_true',
                      help='干运行模式（只显示要删除的文件，不实际删除）')
    return parser.parse_args()

def validate_datetime(time_str):
    """验证时间字符串格式是否正确"""
    try:
        return datetime.strptime(time_str, '%Y%m%d_%H%M%S')
    except ValueError:
        raise ValueError(f"时间格式错误：{time_str}，正确格式应为YYYYMMDD_HHMMSS（例如：20251205_194022）")

def get_folder_datetime(folder_name):
    """从文件夹名中提取日期时间（train_YYYYMMDD_HHMMSS格式）"""
    pattern = r'^train_(\d{8}_\d{6})$'
    match = re.match(pattern, folder_name)
    if not match:
        return None
    try:
        return datetime.strptime(match.group(1), '%Y%m%d_%H%M%S')
    except ValueError:
        return None

def main():
    args = parse_args()
    
    # 验证根目录是否存在
    if not os.path.exists(args.root_dir):
        print(f"错误：根目录不存在 - {args.root_dir}")
        return
    
    # 验证时间参数
    try:
        start_dt = validate_datetime(args.start_time)
        end_dt = validate_datetime(args.end_time)
    except ValueError as e:
        print(f"错误：{e}")
        return
    
    if start_dt >= end_dt:
        print(f"错误：开始时间（{args.start_time}）不能晚于结束时间（{args.end_time}）")
        return
    
    # 统计变量
    total_train_folders_checked = 0
    total_model_folders_found = 0
    total_pt_files_found = 0
    total_pt_files_to_delete = 0
    deleted_files = []
    
    print(f"=== 开始扫描文件 ===")
    print(f"根目录：{args.root_dir}")
    print(f"目标路径：train_*/model/*.pt")
    print(f"时间范围：{args.start_time} 到 {args.end_time}")
    print(f"干运行模式：{'开启' if args.dry_run else '关闭'}")
    print("-" * 60)
    
    # 遍历根目录下的所有文件夹
    for folder_name in os.listdir(args.root_dir):
        train_folder_path = os.path.join(args.root_dir, folder_name)
        
        # 只处理train_开头的文件夹
        if not os.path.isdir(train_folder_path) or not folder_name.startswith('train_'):
            continue
        
        total_train_folders_checked += 1
        
        # 提取文件夹的日期时间
        folder_dt = get_folder_datetime(folder_name)
        if not folder_dt:
            print(f"跳过：文件夹名格式不正确 - {folder_name}")
            continue
        
        # 检查是否在时间范围内
        if not (start_dt <= folder_dt <= end_dt):
            continue
        
        # 拼接model子文件夹路径
        model_folder_path = os.path.join(train_folder_path, 'models')
        
        # 检查model文件夹是否存在
        if not os.path.exists(model_folder_path) or not os.path.isdir(model_folder_path):
            print(f"\n处理文件夹：{folder_name}（{folder_dt.strftime('%Y-%m-%d %H:%M:%S')}）")
            print(f"  ⚠️  未找到model子文件夹，跳过")
            continue
        
        total_model_folders_found += 1
        print(f"\n处理文件夹：{folder_name}（{folder_dt.strftime('%Y-%m-%d %H:%M:%S')}）")
        print(f"  模型文件夹：{model_folder_path}")
        
        # 遍历model文件夹中的.pt文件
        has_pt_files = False
        for filename in os.listdir(model_folder_path):
            if filename.endswith('.pth') or filename.endswith('.pt'):
                has_pt_files = True
                file_path = os.path.join(model_folder_path, filename)
                total_pt_files_found += 1
                total_pt_files_to_delete += 1
                deleted_files.append(file_path)
                
                print(f"  找到PT文件：{filename}")
                
                # 实际删除（非干运行模式）
                if not args.dry_run:
                    try:
                        os.remove(file_path)
                        print(f"  ✅ 已删除：{filename}")
                    except Exception as e:
                        print(f"  ❌ 删除失败：{filename} - {str(e)}")
        
        if not has_pt_files:
            print(f"  ℹ️  model文件夹中未找到.pt文件")
    
    print("-" * 60)
    print(f"=== 扫描完成 ===")
    print(f"检查的train_文件夹总数：{total_train_folders_checked}")
    print(f"在时间范围内的train_文件夹数：{total_model_folders_found}")
    print(f"找到的model子文件夹数：{total_model_folders_found}")
    print(f"在时间范围内找到的PT文件总数：{total_pt_files_to_delete}")
    print(f"实际删除的PT文件总数：{len([f for f in deleted_files if not os.path.exists(f)]) if not args.dry_run else 0}")
    
    if args.dry_run and total_pt_files_to_delete > 0:
        print(f"\n⚠️  干运行模式结束，以上{total_pt_files_to_delete}个文件将在关闭干运行模式后被删除")
    elif total_pt_files_to_delete == 0:
        print("\nℹ️  未找到符合条件的PT文件")

if __name__ == "__main__":
    main()

# python delete_pt.py --root-dir /home/jinfang/project/MedicalNet/output  --start-time 20251205_110234 --end-time 20251206_095556 --dry-run 