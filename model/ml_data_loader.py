import os
import glob
import pandas as pd
from datetime import datetime


class DataLoader:
    """
    从 processed 目录按日期加载多天数据，并支持自动划分训练/测试集。
    """

    def __init__(self, processed_root="processed"):
        self.processed_root = processed_root
        if not os.path.exists(self.processed_root):
            raise FileNotFoundError(f"❌ 路径不存在: {self.processed_root}")

    def list_date_dirs(self):
        """列出 processed 目录下的所有日期文件夹（按日期升序）"""
        all_dirs = [
            d for d in os.listdir(self.processed_root)
            if os.path.isdir(os.path.join(self.processed_root, d))
        ]
        valid_dates = []
        for d in all_dirs:
            try:
                datetime.strptime(d, "%Y-%m-%d")
                valid_dates.append(d)
            except ValueError:
                continue
        valid_dates.sort()
        return valid_dates

    def get_latest_date(self):
        """返回最新日期"""
        all_dates = self.list_date_dirs()
        return all_dates[-1] if all_dates else None

    def load_data_range(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        加载指定日期范围内的所有 all_data.csv 文件，并合并成一个DataFrame。
        """
        all_dates = self.list_date_dirs()
        selected_dates = [d for d in all_dates if start_date <= d <= end_date]

        if not selected_dates:
            raise ValueError(f"❌ 找不到日期范围 {start_date} 至 {end_date} 的数据")

        print(f"📅 加载数据：{selected_dates[0]} → {selected_dates[-1]}（共 {len(selected_dates)} 天）")

        dfs = []
        for date in selected_dates:
            day_path = os.path.join(self.processed_root, date)
            # 只加载 all_data.csv 文件
            all_data_file = os.path.join(day_path, "all_data.csv")

            if not os.path.exists(all_data_file):
                print(f"⚠️ {date} 没有 all_data.csv 文件，跳过")
                continue

            try:
                df = pd.read_csv(all_data_file)
                dfs.append(df)
                print(f"  ✅ 已加载: {date}/all_data.csv ({len(df)} 行)")

            except Exception as e:
                print(f"⚠️ 读取 {all_data_file} 失败：{e}")

        if not dfs:
            raise ValueError(f"❌ 日期 {start_date} 至 {end_date} 内没有有效的 all_data.csv 文件")

        df_all = pd.concat(dfs, ignore_index=True)
        print(f"✅ 加载完成，共 {len(df_all)} 条记录，来自 {len(selected_dates)} 天")

        return df_all

    def get_train_test_dates(self, scheme=1):
        """
        自动生成实验所需的训练/测试时间范围。
        scheme=1 → 9-20至10-(latest-2)训练，10-(latest-1)至10-latest测试
        scheme=2 → 10-1至10-(latest-2)训练，10-(latest-1)至10-latest测试
        """
        all_dates = self.list_date_dirs()
        if len(all_dates) < 5:
            raise ValueError("❌ 数据不足5天，无法执行划分")

        latest = all_dates[-1]
        latest_minus_1 = all_dates[-2]
        latest_minus_2 = all_dates[-3]

        if scheme == 1:
            train_start = "2025-09-20"
        elif scheme == 2:
            train_start = "2025-10-01"
        else:
            raise ValueError("scheme 只能是 1 或 2")

        train_end = latest_minus_2
        test_start = latest_minus_1
        test_end = latest

        print(f"📊 方案{scheme}划分结果：")
        print(f"   训练集：{train_start} → {train_end}")
        print(f"   测试集：{test_start} → {test_end}")
        return train_start, train_end, test_start, test_end