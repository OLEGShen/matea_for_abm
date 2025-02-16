import json
import os
from collections import defaultdict
import matplotlib.pyplot as plt
import pandas as pd
import glob
import numpy as np
from scipy.stats import pearsonr
from scipy.interpolate import make_interp_spline


# 设置关键词列表
keywords = ["competitive", "increase", "hardworking"]


# 初始化结果存储
ration_keyword_count_per_day = defaultdict(int)
sensibility_keyword_count_per_day = defaultdict(int)
combined_keyword_count_per_day = defaultdict(int)


# 读取 JSON 文件路径
folder_path = "2024-12-13-全部-订单0.15-30天"  # 替换为实际路径
file_list = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith("thought.json")]


# 遍历所有 JSON 文件
for file_path in file_list:
    with open(file_path, 'r',encoding='utf-8') as file:
        data = json.load(file)


        # 遍历每个 JSON 项
        for item in data:
            step = item.get("runner_step", 0)
            day = step // 120  # 转化为天数


            # 提取 thought 字段
            ration_thought = item.get("ration_thought", [])
            sensibility_thought = item.get("sensibility_thought", [])


            # 如果 ration_thought 是字符串，则转换为列表
            if isinstance(ration_thought, str):
                ration_thought = [ration_thought]


            # 如果 sensibility_thought 是字符串，则转换为列表
            if isinstance(sensibility_thought, str):
                sensibility_thought = [sensibility_thought]


            try:
                # 统计关键词出现频率
                for keyword in keywords:
                    keyword_lower = keyword.lower()
                    # 统计 ration_thought 列表中每个字符串的关键词频率
                    for text in ration_thought:
                        ration_keyword_count_per_day[day] += text.lower().count(keyword_lower)
                        combined_keyword_count_per_day[day] += text.lower().count(keyword_lower)
                    # 统计 sensibility_thought 列表中每个字符串的关键词频率
                    for text in sensibility_thought:
                        sensibility_keyword_count_per_day[day] += text.lower().count(keyword_lower)
                        combined_keyword_count_per_day[day] += text.lower().count(keyword_lower)
            except Exception as e:
                pass


# 读取 CSV 文件路径
file_pattern = f"{folder_path}/deliver*.csv"
file_paths = glob.glob(file_pattern)
data_list = []


for file in file_paths:
    df = pd.read_csv(file)
    data_list.append(df)


# Combine all data to calculate daily differences
all_data = pd.concat(data_list, axis=0, keys=range(len(data_list)), names=['Agent', 'Index'])


# Calculate daily differences for each agent
def calculate_daily_diff(group):
    group['dis_diff'] = group['dis'].diff().fillna(0)
    group['money_diff'] = group['money'].diff().fillna(0)
    return group


all_data = all_data.groupby('Agent').apply(calculate_daily_diff)


# Calculate averages across agents
daily_avg = all_data.groupby('day').agg({
    'dis': 'mean',
    'money': 'mean',
    'dis_diff': 'mean',
    'money_diff': 'mean'
})
daily_avg['money/dis'] = daily_avg['money'] / daily_avg['dis'].replace(0, np.nan)


# 数据准备
at = [1.2899599799776784, 1.090087841268682, 0.9244703067744924, 0.8017528725927834, 0.7252335781338342,
      0.6662704618891733, 0.6099678502480004, 0.564648455007724, 0.5149578502542435, 0.4960704144220892,
      0.485335330127972, 0.4570776755924062, 0.4561089670250202, 0.4382354387360604, 0.409966312309757,
      0.3934222531397352, 0.3756536950724429, 0.3666391271972302, 0.3680922167539039, 0.3573766431476866,
      0.3481833184542614, 0.3423076372880358, 0.3374786736261652, 0.3484238759068173, 0.3475705996916899,
      0.3435314125792775, 0.3451456832447857, 0.3484377184705598, 0.3435937207986907, 0.3422906931118308]

def reciprocal(arr):
    result = []
    for num in arr:
        if num!= 0:  # 避免除以 0 的错误
            result.append(1 / num)
        else:
            result.append(1000)  # 可以根据需要处理 0 的情况，这里将 0 的倒数设为 None
    return result
at = reciprocal(at)
# 累加 ration_thought 统计
days = sorted(ration_keyword_count_per_day.keys())
ration_counts = [ration_keyword_count_per_day[day] for day in days]


# 累加 Combined 统计
combined_counts = [combined_keyword_count_per_day[day] / 2 for day in days]


action_analysis = daily_avg['money/dis']


# 定义颜色和样式
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', "#8B4513"]
# colors = ["#FFA500", "#8B4513", "#FF1493", "#FFD700", "#808000"]
linestyles = ['-', '--']
font = {'fontname': 'Times New Roman', 'fontsize': 20}  # 进一步增大字体大小

# 折线平滑函数
# def smooth_line(x, y, points=500):
#     x_new = np.linspace(min(x), max(x), points)  # 增加点数
#     spline = make_interp_spline(x, y, k=5)
#     return x_new, spline(x_new)
# 折线平滑函数，包含移动平均
def smooth_line(x, y, points=500, window_size=2):
    # 移动平均滤波
    y_smoothed = np.convolve(y, np.ones(window_size) / window_size, mode='same')
    # 插值
    x_new = np.linspace(min(x), max(x), points)
    spline = make_interp_spline(x, y_smoothed, k=2)  # Cubic spline
    return x_new, spline(x_new)

# 在所有使用 smooth_line 的地方应用新的平滑逻辑
# 例如：x1_1, y1_1 = smooth_line(daily_avg.index, action_analysis, points=500, window_size=7)
# 继续为其他图形调用 smooth_line 处理

# 绘制第 1 张图
fig1, ax1_1 = plt.subplots(figsize=(12, 6))
ax2_1 = ax1_1.twinx()

x1_1, y1_1 = smooth_line(daily_avg.index, action_analysis)
x2_1, y2_1 = smooth_line(range(len(at)), at)

ax1_1.plot(x1_1, y1_1, label='w/o thoughts', color=colors[0], linestyle=linestyles[0], linewidth=4, alpha=0.9)
ax1_1.set_ylabel('money/distance', color=colors[0], **font)
ax1_1.tick_params(axis='y', labelcolor=colors[0])

ax2_1.plot(x2_1, y2_1, label="Involution(t)", color=colors[4], linestyle=linestyles[1], linewidth=4, alpha=0.7)
ax2_1.set_ylabel('Involution(t)', color=colors[4], **font)
ax2_1.tick_params(axis='y', labelcolor=colors[4])


fig1.suptitle('w/o thoughts', **font, y=0.95)

# 绘制第 2 张图
fig2, ax1_2 = plt.subplots(figsize=(12, 6))
ax2_2 = ax1_2.twinx()

x1_2, y1_2 = smooth_line(days, ration_counts)
x2_2, y2_2 = smooth_line(range(len(at)), at)

ax1_2.plot(x1_2, y1_2, label="w/o R&S CoT", color=colors[1], linestyle=linestyles[0], linewidth=4, alpha=0.9)
ax1_2.set_ylabel('Keyword Count', color=colors[1], **font)
ax1_2.tick_params(axis='y', labelcolor=colors[1])

ax2_2.plot(x2_2, y2_2, label="Involution(t)", color=colors[4], linestyle=linestyles[1], linewidth=4, alpha=0.7)
ax2_2.set_ylabel('Involution(t)', color=colors[4], **font)
ax2_2.tick_params(axis='y', labelcolor=colors[4])



fig2.suptitle('w/o R&S CoT', **font, y=0.95)

# 绘制第 3 张图
fig3, ax1_3 = plt.subplots(figsize=(12, 6))
ax2_3 = ax1_3.twinx()

x1_3, y1_3 = smooth_line(days, combined_counts)
x2_3, y2_3 = smooth_line(range(len(at)), at)

ax1_3.plot(x1_3, y1_3, label="w/o cluster", color=colors[2], linestyle=linestyles[0], linewidth=4, alpha=0.9)
ax1_3.set_ylabel('Keyword Count', color=colors[2], **font)
ax1_3.tick_params(axis='y', labelcolor=colors[2])

ax2_3.plot(x2_3, y2_3, label="Involution(t)", color=colors[4], linestyle=linestyles[1], linewidth=4, alpha=0.7)
ax2_3.set_ylabel('Involution(t)', color=colors[4], **font)
ax2_3.tick_params(axis='y', labelcolor=colors[4])


fig3.suptitle('w/o cluster', **font, y=0.95)

# 绘制第四张图
a4 = [9, 14, 13, 13, 11, 15, 14, 16, 16, 14, 12, 15, 14, 17, 19, 15, 17, 17, 19, 19, 19, 15, 14, 14, 17, 20, 16, 15, 21, 21]
fig4, ax1_4 = plt.subplots(figsize=(12, 6))
ax2_4 = ax1_4.twinx()

x1_4, y1_4 = smooth_line(days, a4)
x2_4, y2_4 = smooth_line(range(len(at)), at)

ax1_4.plot(x1_4, y1_4, label='TeTree', color=colors[3], linestyle=linestyles[0], linewidth=4, alpha=0.9)
ax1_4.set_ylabel('Keyword Count', color=colors[3], **font)
ax1_4.tick_params(axis='y', labelcolor=colors[3])

ax2_4.plot(x2_4, y2_4, label="Involution(t)", color=colors[4], linestyle=linestyles[1], linewidth=4, alpha=0.7)
ax2_4.set_ylabel('Involution(t)', color=colors[4], **font)
ax2_4.tick_params(axis='y', labelcolor=colors[4])



fig4.suptitle('TeTree', **font, y=0.95)

# 创建一个新的 figure 和 axes 用于绘制图例
fig_legend, ax_legend = plt.subplots(figsize=(12, 1))

# 手动添加图例项
lines1 = [plt.Line2D([0], [0], color=colors[0], linestyle=linestyles[0], label='w/o thoughts'),
         plt.Line2D([0], [0], color=colors[1], linestyle=linestyles[0], label='w/o R&S CoT'),
         plt.Line2D([0], [0],color=colors[2], linestyle=linestyles[0], label='w/o cluster'),
         plt.Line2D([0], [0], color=colors[3], linestyle=linestyles[0], label='TeTree'),
         plt.Line2D([0], [0], color=colors[4], linestyle=linestyles[1], label='Involution(t)')]

# 显示图例，设置为横向排列，位置为左上角
ax_legend.legend(handles=lines1, loc='upper left', ncol=5)

# 显示图形
plt.show()


def calculate_correlations(target_array, *other_arrays):
    """
    计算目标数组与多个其他数组之间的皮尔逊相关系数。


    参数:
    target_array (np.array): 目标数组。
    *other_arrays (np.array): 可变数量的其他数组。


    返回:
    np.array: 包含目标数组与其他每个数组之间皮尔逊相关系数绝对值的数组。
    """
    correlations = []
    target_array = np.array(target_array)


    # 计算目标数组与每个其他数组之间的相关系数
    for other_array in other_arrays:
        other_array = np.array(other_array)
        corr_coefficient, _ = pearsonr(target_array, other_array)
        correlations.append(np.abs(corr_coefficient))


    return np.array(correlations)


ans = calculate_correlations(at, action_analysis, ration_counts, combined_counts, a4)
print("相关系数绝对值数组:", ans)