import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# 假设文件路径
file_path = "2024-12-09-勤奋-订单0.3-30天"
files = [f for f in os.listdir(file_path) if f.startswith('deliver')]  # 获取所有以deliver开头的文件并排序
files = sorted(files)

param = 'money'
param = 'workTime'
# 计算每日收益
def get_daily_income(file):
    df = pd.read_csv(os.path.join(file_path, file))
    if param=='money':
        df['daily_income'] = df['money'].diff().fillna(0)  # 计算每日收益，fillna(0) 处理第一个数据点
    else:
        df['daily_income'] = df['order_count'].apply(lambda x: 1 if x != 0 else 0)
    return df[['day', 'daily_income']]


# 分组
group1_files = files[:33]  # 前33个文件
group2_files = files[33:66]  # 中间33个文件
group3_files = files[66:]  # 后34个文件


# 计算每组的每日收益平均值
def calculate_group_daily_income(group_files):
    all_daily_income = []
    for file in group_files:
        daily_income = get_daily_income(file)
        all_daily_income.append(daily_income)

    # 合并所有文件的每日收益数据
    combined_daily_income = pd.concat(all_daily_income)
    # 按天数分组，计算每天的平均收益
    daily_avg_income = combined_daily_income.groupby('day')['daily_income'].sum().reset_index()
    return daily_avg_income


# 获取三组的每日收益平均值
group1_daily_income = calculate_group_daily_income(group1_files)
group2_daily_income = calculate_group_daily_income(group2_files)
group3_daily_income = calculate_group_daily_income(group3_files)


# 使用移动平均法平滑曲线
def smooth_curve(data, window_size=5):
    return data.rolling(window=window_size, center=True).mean()


# 平滑曲线
group1_daily_income['smoothed'] = smooth_curve(group1_daily_income['daily_income'])
group2_daily_income['smoothed'] = smooth_curve(group2_daily_income['daily_income'])
group3_daily_income['smoothed'] = smooth_curve(group3_daily_income['daily_income'])

# 使用Seaborn绘制图形
plt.figure(figsize=(10, 6))
sns.set(style="whitegrid")  # 设置背景样式

# 绘制平滑后的曲线，并添加阴影
plt.fill_between(group1_daily_income['day'], group1_daily_income['smoothed'], color='tab:blue', alpha=0.2)  # 阴影
sns.lineplot(x='day', y='smoothed', data=group1_daily_income,  color='tab:blue', linewidth=2)

plt.fill_between(group2_daily_income['day'], group2_daily_income['smoothed'], color='tab:green', alpha=0.2)  # 阴影
sns.lineplot(x='day', y='smoothed', data=group2_daily_income,  color='tab:green', linewidth=2)

plt.fill_between(group3_daily_income['day'], group3_daily_income['smoothed'], color='tab:red', alpha=0.2)  # 阴影
sns.lineplot(x='day', y='smoothed', data=group3_daily_income, color='tab:red', linewidth=2)

plt.xlabel('',)
plt.ylabel('')

# 限制 x 轴和 y 轴范围
plt.xlim(3, 28)  # 限制 x 轴范围
plt.ylim(500, group1_daily_income['smoothed'].max() * 1.1)  # 限制 y 轴范围，y 轴最大值设为平滑曲线最大值的 1.1 倍

# 美化显示
sns.despine()  # 去掉顶部和右边的边框

plt.show()
