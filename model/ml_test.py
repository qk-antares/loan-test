import os
import pandas as pd
from datetime import datetime

# 相对路径，processed 文件夹是当前文件夹父目录的兄弟
folder_path = os.path.join(os.path.dirname(os.getcwd()), 'processed')
print("读取路径:", folder_path)

# 起始日期
start_date = datetime(2025, 10, 1)

# 遍历子文件夹，找到 XYF_CODE.csv 文件
csv_files = []
for root, dirs, files in os.walk(folder_path):
    for file in files:
        print(file.lower())
        if file.lower() == "jy_code.csv":
            try:
                folder_name = os.path.basename(root)
                file_date = datetime.strptime(folder_name, "%Y-%m-%d")
                if file_date >= start_date:
                    csv_files.append(os.path.join(root, file))
            except:
                continue

print("找到的 CSV 文件：", csv_files)

if not csv_files:
    raise FileNotFoundError("没有找到符合条件的 HXQB_CODE.csv 文件")

# 读取并合并
df_list = [pd.read_csv(f) for f in csv_files]
df = pd.concat(df_list, ignore_index=True)
print("总数据行数:", len(df))

# 计算总体通过率
total_samples = len(df)
total_passed = df['label'].sum()
overall_pass_rate = total_passed / total_samples if total_samples > 0 else 0
print(f"\n=== 所有数据总体通过率 ===")
print(f"样本总数: {total_samples}, 通过数: {total_passed}, 通过率: {overall_pass_rate:.2%}")

# 筛选条件
first_tier = ['北京市', '上海市', '广州市', '深圳市']
new_first_tier = ['杭州市', '南京市', '成都市', '武汉市', '重庆市', '苏州市', '天津市', '西安市', '长沙市','青岛市']
selected_cities = first_tier + new_first_tier
selected_provinces = ['北京市', '上海市', '广东省', '浙江省', '江苏省']
selected_banks = [102, 103, 104, 105, 301, 402]

# conditions = (
#     (df['degree'].isin([4, 5, 6])) &
#     (df['bankCardInfo.bankCode'].isin(selected_banks)) &
#     (df['income'].isin([3,4])) &
#     (df['idInfo.birthDate'].between(30, 50)) &
#     (df['city'].isin(selected_cities)) &
#     (df['province'].isin(selected_provinces)) &
#     (df['idInfo.gender'] == 'M') &
#     (df['idInfo.nation'].isin(['汉'])) &
#     (df['companyInfo.occupation'] != 90)
# )

state_owned_banks = [102, 103, 104, 105, 301, 402]  # 国有大行
joint_stock_banks = [302, 303, 304, 305, 306, 307, 308, 309, 310]  # 股份制银行
small_local_banks = [313, 403, 404, 501]  # 小银行、农商行、外资行等

# 所有不想要的银行列表
excluded_banks = state_owned_banks + joint_stock_banks + small_local_banks

# conditions = (
#     (df['degree'].isin([4, 5, 6])) &
#     (~df['bankCardInfo.bankCode'].isin(excluded_banks)) &  # 排除这些银行
#     (df['income'].isin([3, 4])) &
#     (df['idInfo.birthDate'].between(30, 45)) &
#     (df['city'].isin(selected_cities)) &
#     (df['province'].isin(selected_provinces)) &
#     (df['idInfo.gender'] == 'M') &
#     (df['maritalStatus'] == 1)
# )

conditions = (
    # (df['degree'].isin([4])) &
    (df['idInfo.gender'] == 'F')
)

df_filtered = df[conditions]

# 筛选后的通过率
filtered_samples = len(df_filtered)
filtered_passed = df_filtered['label'].sum()
filtered_pass_rate = filtered_passed / filtered_samples if filtered_samples > 0 else 0

print(f"\n=== 所有数据 筛选条件通过率 ===")
print(f"筛选后样本数: {filtered_samples}, 通过数: {filtered_passed}, 通过率: {filtered_pass_rate:.2%}")
print(f"筛选后与总体通过率通过率差异: { filtered_pass_rate- overall_pass_rate:.2%}")
