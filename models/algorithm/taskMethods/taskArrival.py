import numpy as np


# 均匀到达

# 基于排队论生成任务到达时间序列，接受参数为
# arrival_rata 订单到达率
# simulation_time 系统模拟时间
def generate_task_arrival_time_queue_theory(arrival_rate, simulation_time):
    arrival_times = []
    # 使用泊松分布生成订单到达时间
    inter_arrival_time = np.random.exponential(1 / arrival_rate)
    while inter_arrival_time < simulation_time:
        arrival_times.append(inter_arrival_time)
        inter_arrival_time += np.random.exponential(1 / arrival_rate)

    return arrival_times


# 任务序列均匀到达
def generate_task_arrival_time_uniform_distribution(arrival_rate, simulation_time):
    arrival_times = []
    space = simulation_time / (arrival_rate * simulation_time)
    tmp_time = 0
    while tmp_time < simulation_time:
        arrival_times.append(tmp_time)
        tmp_time += space

    return arrival_times


if __name__ == '__main__':
    print(generate_task_arrival_time_uniform_distribution(2, 10))