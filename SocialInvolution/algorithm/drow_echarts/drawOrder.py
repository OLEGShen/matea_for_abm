import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import os
# 读取CSV文件
def analyze_january_time_distribution(file_path, month_column, time_column):
    # 读取文件
    data = pd.read_csv(file_path)

    # 检查列是否存在
    if month_column not in data.columns or time_column not in data.columns:
        raise ValueError(f"列 {month_column} 或 {time_column} 不存在于文件中。")

    # 确保时间和月份格式正确
    data[time_column] = pd.to_datetime(data[time_column], format='%H:%M', errors='coerce')
    data[month_column] = pd.to_datetime(data[month_column], errors='coerce')

    # 如果时间解析失败
    if data[time_column].isna().any() or data[month_column].isna().any():
        print("一些时间值或日期值无法解析，已被忽略。")
        data = data.dropna(subset=[time_column, month_column])

    # 添加小时列和月份列
    data['hour'] = data[time_column].dt.hour
    data['month'] = data[month_column].dt.month

    # 准备绘制数据
    month_distributions = {month: {hour: 0 for hour in range(24)} for month in range(1, 13)}

    # 统计每个月的小时分布
    for month in range(1, 13):
        month_data = data[data['month'] == month]
        hour_counts = Counter(month_data['hour'].dropna())
        for hour, count in hour_counts.items():
            month_distributions[month][hour] += count


    # Path to the folder containing CSV files
    folder_path = "2024-12-13-全部-订单0.15-30天"

    # Initialize lists to store results
    hourly_order_counts = [0] * 24  # A list to track orders per hour

    # Process each CSV file in the folder
    for file in os.listdir(folder_path):
        if file.startswith("deliver"):
            file_path = os.path.join(folder_path, file)
            # Read the CSV file
            data = pd.read_csv(file_path)

            # Calculate the change in total_orders (diff)
            order_diff = data['total_order'].diff().fillna(0)
            # Extract steps and corresponding order changes where diff > 0
            for step, order_increase in zip(data['step'], order_diff):
                if order_increase > 0:
                    hour = int(step % 120 / 5)  # Map the step to hours (0-23)
                    hourly_order_counts[hour] += order_increase
    # plt.figure(figsize=(12, 6))
    # hours = list(range(24))
    # plt.plot(hours, hourly_order_counts, marker='o', linestyle='-', color='blue', label='Order Count')
    # plt.xticks(hours)
    # plt.title("Order Distribution by Hour of the Day")
    # plt.xlabel("Hour of the Day (0-23)")
    # plt.ylabel("Number of Orders")
    # plt.grid(True)
    # plt.legend()
    # plt.show()
    # 绘制折线图
    plt.figure(figsize=(12, 8))
    for month, distribution in month_distributions.items():
        hours = sorted(distribution.keys())
        counts = [distribution[hour] for hour in hours]
        plt.plot(hours, counts, label=f'Month {month}')
    hours = list(range(24))
    plt.plot(hours, hourly_order_counts, label=f'Ours Sim')

    # 图表美化
    plt.xlabel('hours (0-24)')
    plt.ylabel('order_num')
    plt.title('title')
    plt.xticks(range(24))
    plt.legend(title="月份", loc="upper right")
    plt.grid(alpha=0.5)
    plt.show()

# 示例用法
file_path = 'ZomatoDataset.csv'  # 替换为你的文件路径
month_column = 'Order_Date'  # 替换为你的月份列名称
time_column = 'Time_Order_picked'  # 替换为你的时间列名称
analyze_january_time_distribution(file_path, month_column, time_column)

