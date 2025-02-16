"""
对应论文当中情绪事件发生的时间读取
"""
import json
def read_tree_json(route, word):

    ans = []
    with open(route, 'r', encoding='utf-8') as file:
        json_tree = json.load(file)
    check_info = json_tree["data_list"]
    for rider_info in check_info.values():
        list_info = get_info_time(word, rider_info)
        ans.append(list_info)
    print(ans)
    return [sum(elements) for elements in zip(*ans)]

def get_info_time(word, info, day_num=30):
    ans_time = []
    print('info', info)
    for i in range(day_num):
        if str(i * 120) in info:
            ans_time.append(1 if any(word in keyword for keyword in info[str(i * 120)][-1]) else 0)
        else:
            ans_time.append(0)
    return ans_time

if __name__ == "__main__":
    ans = read_tree_json('/home/syf/project/MACE/MACE/simulation/examples/SocialInvolution/entity/agent_log/2024-12-09_21_all_lazy/tree/tree.json', 'hardwork')
    print(ans)