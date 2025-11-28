import os
import shutil
import random
from pathlib import Path


def split_train_val_robust(root_path, val_ratio=0.1, seed=42):
    """
    【V2 安全版】划分训练集与验证集

    安全机制：
    1. 动态扫描 train 下的所有子文件夹。
    2. 计算所有子文件夹中文件名的【交集】。
    3. 仅当一组文件名在所有子文件夹都存在时，才视为有效数据对。
    4. 仅从有效数据对中抽取 10% 移动到 val，确保数据绝对完整。
    """
    root = Path(root_path)
    if not root.exists():
        print(f"Error: 路径 {root} 不存在，请检查。")
        return

    # 设置随机种子，保证结果可复现
    random.seed(seed)

    # 1. 寻找所有的 train 文件夹
    train_dirs = []
    for dirpath, dirnames, filenames in os.walk(root):
        if 'train' in dirnames:
            train_dirs.append(Path(dirpath) / 'train')

    print(f"共发现 {len(train_dirs)} 个 train 目录，开始安全划分...\n")

    for train_dir in train_dirs:
        val_dir = train_dir.parent / 'val'

        # 获取当前 train 下的所有子目录 (Infrared, Visible, etc.)
        subdirs = [d for d in os.listdir(train_dir) if (train_dir / d).is_dir()]

        if not subdirs:
            print(f"[跳过] {train_dir} 是空的或没有子目录")
            continue

        print(f"正在分析: {train_dir}")

        # =========================================================
        # 2. 核心逻辑：计算文件名交集 (Intersection)
        # =========================================================
        common_files = None

        for sub in subdirs:
            # 获取当前子目录下所有非隐藏文件名
            # 使用 set (集合) 以便进行交集运算
            current_files = set(f for f in os.listdir(train_dir / sub) if not f.startswith('.'))

            if common_files is None:
                common_files = current_files
            else:
                # 取交集：只保留大家都有的文件
                common_files = common_files & current_files

        # 如果交集为空，说明文件名完全对不上（或者文件夹为空）
        if not common_files:
            print(f"  [跳过] 无法找到共同文件。可能是文件名后缀不匹配或目录为空。")
            continue

        # 将集合转回列表，并进行排序 (排序是为了保证 random.seed 的一致性)
        valid_files = sorted(list(common_files))

        total_files = len(valid_files)
        move_count = int(total_files * val_ratio)

        if move_count == 0:
            print(f"  [跳过] 有效完整数据太少 ({total_files}组)，不足以划分 10%。")
            continue

        # 随机打乱并选取前 10%
        random.shuffle(valid_files)
        files_to_move = valid_files[:move_count]

        print(f"  - 包含子目录: {subdirs}")
        print(f"  - 发现完整配对数据: {total_files} 组")
        print(f"  - 准备移动: {move_count} 组 到 val")

        # 3. 执行移动操作
        moved_count = 0
        for filename in files_to_move:
            try:
                # 遍历所有子目录，移动对应的文件
                for subdir_name in subdirs:
                    src_file = train_dir / subdir_name / filename
                    dst_subdir = val_dir / subdir_name
                    dst_file = dst_subdir / filename

                    # 创建 val 下对应的子目录
                    os.makedirs(dst_subdir, exist_ok=True)

                    # 移动文件
                    shutil.move(str(src_file), str(dst_file))

                moved_count += 1
            except Exception as e:
                # 理论上只要通过了交集检查，这里极少会报错，除非是文件权限问题
                print(f"  [Error] 移动 {filename} 时发生意外: {e}")

        print(f"  - 成功完成: {moved_count} 组\n")

    print("所有操作已完成。")


if __name__ == "__main__":
    # ================= 配置区域 =================
    DATASET_ROOT = "DDL-12"
    # ===========================================

    split_train_val_robust(DATASET_ROOT, val_ratio=0.1)