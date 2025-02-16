import pandas as pd
import matplotlib.pyplot as plt
import glob
import numpy as np
import seaborn as sns

# Specify the directory and file pattern
directory = "2024-12-09-勤奋-订单0.3-30天"
file_pattern = f"{directory}/deliver*.csv"

# Read all CSV files
file_paths = glob.glob(file_pattern)
data_list = []

# Load all CSV files into data_list
for file in file_paths:
    df = pd.read_csv(file)  # Read each CSV file
    data_list.append(df)

# Split the data into three groups
group1 = data_list[:]
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
    return daily_avg['money_diff']

def calculate_avg_dis(group):
    all_data = pd.concat(group, axis=0, keys=range(len(group)), names=['Agent', 'Index'])
    all_data = all_data.groupby('Agent').apply(calculate_daily_diff)
    daily_avg = all_data.groupby('day').agg({
        'dis': 'mean',
        'money': 'mean',
        'dis_diff': 'mean',
        'money_diff': 'mean'
    })
    daily_avg['money/dis'] = daily_avg['money'] / daily_avg['dis'].replace(0, np.nan)
    return daily_avg['dis_diff']

# Calculate the average money/dis for each group
avg_money_dis1 = calculate_avg_money_dis(group1)
avg_money_dis2 = calculate_avg_dis(group1)

plt.figure(figsize=(10, 3))
sns.set(style="whitegrid")  # 设置背景样式

# Plot the first line on the primary y-axis
ax1 = plt.gca()
line1 = ax1.plot(avg_money_dis1.index, avg_money_dis1, color='tab:blue', linewidth=2, label='Money Diff')
ax1.set_xlabel('days', fontsize=20)
ax1.tick_params(axis='y', labelcolor='tab:blue', labelsize=18)
ax1.tick_params(axis='x', labelsize=18)
ax1.set_xlim(1, 28)

# Create a second y-axis that shares the same x-axis
ax2 = ax1.twinx()
line2 = ax2.plot(avg_money_dis2.index, avg_money_dis2, color='tab:red', linewidth=2, label='Distance Diff')
ax2.tick_params(axis='y', labelcolor='tab:red', labelsize=18)

# Combine the legends from both axes
lines = line1 + line2


sns.despine(right=False)  # 去掉顶部和左边的边框，保留右边的边框用于第二个 y 轴
plt.show()