import pandas as pd
import os
import matplotlib.pyplot as plt

# 设置全局字体为Times New Roman，字号为20，不加粗
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 16
plt.rcParams['axes.titleweight'] = 'normal'  # 去掉加粗

# 假设CSV文件名为"orders.csv"，包含"Delivery_person_ID"和"Time_taken"两列
file_path = "ZomatoDataset.csv"

# 读取CSV文件
df = pd.read_csv(file_path)
month_column = 'Order_Date'
df[month_column] = pd.to_datetime(df[month_column], errors='coerce')
data = df

# 确保数据中包含"rider"和"Time_taken"列
if not {'Delivery_person_ID', 'Time_taken'}.issubset(data.columns):
    raise ValueError("CSV文件中缺少'Delivery_person_ID'或'Time_taken'列")

# 按骑手统计配送总时长和接单总数
rider_stats = data.groupby('Delivery_person_ID').agg(
    total_Time_taken=('Time_taken', 'sum'),
    total_orders=('Time_taken', 'count')
).reset_index()

# Path to the folder containing CSV files
folder_path = "2024-12-09-勤奋-订单0.3-30天"

# Initialize lists to store results
work_times = []
total_orders = []

# Process each CSV file in the folder
for file in os.listdir(folder_path):
    if file.startswith("deliver"):
        file_path = os.path.join(folder_path, file)
        # Read the CSV file
        data = pd.read_csv(file_path)

        # Calculate work time (rows where 'current_orders' > 0)
        work_time = (data['order_count'] > 0).sum()
        work_time = work_time/120 * 24
        # Get total orders (value in the last row of 'total_orders' column)
        total_order = data['total_order'].iloc[-1]

        # Append results
        work_times.append(work_time)
        total_orders.append(total_order)

work_times_fact = [x/10 for x in rider_stats['total_Time_taken'].to_list()]
work_times_sim_len = len(work_times)
work_times_fact_len = len(work_times_fact)
work_times.extend(work_times_fact)
total_orders.extend(rider_stats['total_orders'])
color1 = '#EEA78B'
color2 = '#B6C5D8'
colors = [color1] * work_times_sim_len + [color2] * work_times_fact_len

# Plotting the scatter plot
plt.figure(figsize=(10, 10))

# Scatter plot with color and alpha settings
plt.scatter(work_times, total_orders, c=colors, alpha=0.7, edgecolor='k', s=50)

# Set title and labels with larger, more readable fonts
plt.xlabel("Work Time (hours)",fontsize=20)
plt.ylabel("Total Orders",fontsize=20)

# Customize ticks and grid lines
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.grid(True, linestyle='--', alpha=0.6)

# Add legend to distinguish between simulation and real data
from matplotlib.lines import Line2D
legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor=color1, markersize=10, label='Simulated Agents'),
                   Line2D([0], [0], marker='o', color='w', markerfacecolor=color2, markersize=10, label='Real Agents')]
plt.legend(handles=legend_elements, loc='upper left', fontsize=20)

# Show the plot
plt.tight_layout()
plt.show()
