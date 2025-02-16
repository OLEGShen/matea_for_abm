import pandas as pd
import matplotlib.pyplot as plt
import glob
import numpy as np
import seaborn as sns

# Specify the directory and file pattern
directory = "2024-12-09-勤奋-订单0.3-30天"
file_pattern = f"{directory}/deliver*.csv"

# Read all JSON files
file_paths = glob.glob(file_pattern)
data_list = []

# Load all JSON files into data_list
for file in file_paths:
    df = pd.read_csv(file)  # Read each JSON file
    data_list.append(df)

# Split the data into three groups
group1 = data_list[:33]
group2 = data_list[33:66]
group3 = data_list[66:]

# Function to calculate daily differences for each group
def calculate_daily_diff(group):
    group['dis_diff'] = group['dis'].diff().fillna(0)
    group['money_diff'] = group['money'].diff().fillna(0)
    return group

# Function to calculate the average money/dis for a group
def calculate_avg_money_dis(group):
    all_data = pd.concat(group, axis=0, keys=range(len(group)), names=['Agent', 'Index'])
    all_data = all_data.groupby('Agent').apply(calculate_daily_diff)
    daily_avg = all_data.groupby('day').agg({
        'dis': 'mean',
        'money': 'mean',
        'dis_diff': 'mean',
        'money_diff': 'mean'
    })
    daily_avg['money/dis'] = daily_avg['money'] / daily_avg['dis'].replace(0, np.nan)
    return daily_avg['money/dis']

# Calculate the average money/dis for each group
avg_money_dis1 = calculate_avg_money_dis(group1)
avg_money_dis2 = calculate_avg_money_dis(group2)
avg_money_dis3 = calculate_avg_money_dis(group3)
plt.figure(figsize=(10, 6))
# Plotting with three lines for the three groups
sns.set(style="whitegrid")  # 设置背景样式
# Plot for each group
plt.plot(avg_money_dis1.index, avg_money_dis1,color='tab:blue', linewidth=2)
plt.plot(avg_money_dis2.index, avg_money_dis2,  color='tab:green', linewidth=2)
plt.plot(avg_money_dis3.index, avg_money_dis3,  color='tab:red', linewidth=2)

# Adding labels, title, and legend
plt.xlabel('days',fontsize=20)
plt.ylabel('average individual income per dis',fontsize=20)
# 调整刻度字体大小
plt.xticks(fontsize=18)  # 调整 x 轴刻度字体大小
plt.yticks(fontsize=18)  # 调整 y 轴刻度字体大小
plt.xlim(1, 28)  # 限制 x 轴范围
# plt.ylim(0.012, 0.022)  # 限制 y 轴范围，y 轴最大值设为平滑曲线最大值的 1.1 倍

# Add a grid and legend

sns.despine()  # 去掉顶部和右边的边框
# Show the plot

plt.show()
