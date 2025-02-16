import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 设置字体为Times New Roman
rcParams['font.family'] = 'Times New Roman'

# File path for the Zomato dataset
zomato_file_path = "ZomatoDataset.csv"

# Read the Zomato dataset
df = pd.read_csv(zomato_file_path)

# Specify the date format and convert to datetime
month_column = 'Order_Date'
df[month_column] = pd.to_datetime(df[month_column], format='%d-%m-%Y', errors='coerce')

# Ensure the necessary columns exist
if not {'Delivery_person_ID', 'Time_taken', month_column}.issubset(df.columns):
    raise ValueError("CSV file is missing necessary columns")

# Add 'day' column for grouping by day
df['day'] = df[month_column].dt.day

# Add 'month' column for more precise grouping
df['month'] = df[month_column].dt.month

# Group by Delivery_person_ID and day to calculate total work time per rider per day
rider_daily_work_time = df.groupby(['Delivery_person_ID', 'month', 'day']).agg(
    daily_work_time=('Time_taken', 'sum')  # Sum up the work time for each rider per day
).reset_index()

# Group by 'month' and 'day' to calculate the average work time across all riders
avg_daily_work_time = rider_daily_work_time.groupby(['month', 'day']).agg(
    avg_work_time=('daily_work_time', 'mean')  # Average work time per day
).reset_index()

# Print results for verification
print(avg_daily_work_time)

# Folder path for simulated data
folder_path = "2024-12-09-勤奋-订单0.3-30天"

# Dictionary to store simulated data results
simulated_data = {}

# Process each CSV file in the folder
for file in os.listdir(folder_path):
    if file.startswith("deliver"):
        file_path = os.path.join(folder_path, file)
        # Read the CSV file
        data = pd.read_csv(file_path)

        # Ensure the necessary columns exist in the simulated data
        if not {'day', 'order_count'}.issubset(data.columns):
            raise ValueError(f"CSV file {file} is missing 'day' or 'order_count' columns")

        # Calculate work time for the agent (rows where 'order_count' > 0)
        data['work_time'] = (data['order_count'] > 0).astype(int)  # 1 if order_count > 0
        daily_work_time = data.groupby('day')['work_time'].sum()

        # Store the daily work time for this agent
        for day, work_time in daily_work_time.items():
            if day not in simulated_data:
                simulated_data[day] = []
            simulated_data[day].append(work_time / 120 * 24)  # Convert to hours

# Calculate daily average work time for simulated data
simulated_avg_work_time = {
    day: sum(times) / len(times) for day, times in simulated_data.items()
}

# Prepare data for plotting
days = list(range(1, 31))
zomato_avg_time = [
    avg_daily_work_time.loc[avg_daily_work_time['day'] == day, 'avg_work_time'].mean()/10
    if day in avg_daily_work_time['day'].values else 4
    for day in days
]
simulated_avg_time = [simulated_avg_work_time.get(day, 0) for day in days]
color1 = '#EEA78B'
color2 = '#B6C5D8'
# Plotting the line chart with adjusted styling
plt.figure(figsize=(12, 6))
plt.plot(days, zomato_avg_time, label="Real Avg Work Time", color=color1, marker='o', markersize=6, linestyle='-', linewidth=4)
plt.plot(days, simulated_avg_time, label="Simulated Avg Work Time", color=color2, marker='x', markersize=6, linestyle='--', linewidth=4)

# Title and axis labels
plt.xlabel("Days", fontsize=20)
plt.ylabel("Average Work Time (hours)", fontsize=20)

# Customize ticks and grid
plt.xticks(days, fontsize=20)
plt.yticks(fontsize=20)
plt.grid(True, which='both', linestyle='--', linewidth=0.7)

# Legend
plt.legend(fontsize=20)

# Adjust layout to ensure no overlap
plt.tight_layout()

# Show the plot
plt.show()
