
import os
import glob

import openai

from keybert import KeyLLM
from sentence_transformers import SentenceTransformer
from langchain.chains.question_answering import load_qa_chain
from langchain_openai import AzureChatOpenAI
from llama_start import LLaMA3_LLM
from keybert.llm import LangChain
from keybert import KeyLLM
import json
from ete3 import Tree
import numpy as np
from scipy import spatial
import csv
import datetime
import time
os.environ["AZURE_OPENAI_API_KEY"] = "API_KEY"
os.environ["AZURE_OPENAI_ENDPOINT"] = "ENDPOINT"
max_keywords = 5
# 提取关键词
def get_intentions(docs, keywords_list):
    chain = load_qa_chain(AzureChatOpenAI(
                azure_deployment="gpt-35-turbo",
                openai_api_version="2024-02-15-preview",
            ))
    model = SentenceTransformer('../../../../models/huggingface/all-MiniLM-L6-v2')
    embeddings = model.encode(docs, convert_to_tensor=True)

    # Create your LLM
    llm = LangChain(chain)

    # Load it in KeyLLM
    kw_model = KeyLLM(llm)
    keywords_list = [keywords_list[0] +keywords_list[1]] * len(docs)

    # Extract keywords
    keywords = kw_model.extract_keywords(docs, embeddings=embeddings, candidate_keywords=keywords_list, threshold=.75)
    return keywords


# 获得时间决策的思维
def get_all_one_day_thoughts(thought_list, thought_type="sensibility_thought"):
    days_time_actions = {}  # 记录每天行为的改变
    for t in thought_list:
        if int(t["runner_step"]) % 120 == 0:
            print('info', t)
            runner_step = int(t['runner_step'])
            before_go_work_time = int(t["param_dict"]["before_go_work_time"].split(":")[0])
            before_get_off_work_time = int(t["param_dict"]["before_get_off_work_time"].split(":")[0])
            if days_time_actions:
                last_runner_step = max(days_time_actions.keys())
                last_times = days_time_actions[last_runner_step][:2]  # 获取最后记录的时间
                
                # 比较时间，确定变化类型
                if (before_get_off_work_time - before_go_work_time) > (last_times[1] - last_times[0]):
                    change_type = '0'  # 时间增加
                elif (before_get_off_work_time - before_go_work_time) < (last_times[1] - last_times[0]):
                    change_type = '1'  # 时间减少
                elif before_go_work_time < last_times[0]:
                    change_type = '2'  # 时间提前
                elif before_go_work_time > last_times[0]:
                    change_type = '3'  # 时间后移
                else:
                    change_type = '4'  # 无变化
            else:
                change_type = '4'  # 第一个记录，无变化
            days_time_actions[runner_step] = [before_go_work_time, before_get_off_work_time, change_type, [json.dumps(t[thought_type], ensure_ascii=False)]]
            # if int(t["runner_step"]) != 0:
            #     one_day_ans = []
            #     all_ans.append(one_day_ans)
            # last_runner_step = max(days_time_actions.keys())
            # days_time_actions[last_runner_step][-1].append(json.dumps(t[thought_type], ensure_ascii=False))
    return days_time_actions


def process_time_actions_dict(time_actions_dict):
    # 将字典转换为有序列表，以便我们可以轻松地计算连续的 runner_step 之间的差值
    sorted_steps = sorted(time_actions_dict.items())
    processed_dict = {}
    prev_step = None
    print("sorted", sorted_steps)
    for step, [go_work_time, get_off_time, change_type, one_day_ans] in sorted_steps:
        # 如果前一个 runner_step 存在且 change_type 不一致，或者当前是第一个元素
        print(step, [go_work_time, get_off_time, change_type, one_day_ans])
        if prev_step is None or change_type != processed_dict[prev_step][2]:
            step_diff = step - prev_step if prev_step is not None else 0
            processed_dict[step] = [go_work_time, get_off_time, change_type, step_diff, one_day_ans]
            prev_step = step
        print('test', processed_dict)
    return processed_dict


def get_drow_info(keywords_list, drow_route):
    with open(drow_route, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    actions = get_all_one_day_thoughts(data)
    all_sentence = []
    processed_actions = process_time_actions_dict(actions)
    day_thought_list = []
    for i, day_thought in processed_actions.items():
        print('day_thought',day_thought)
        day_thought_list.append(day_thought[-1][0])
    all_keywords = get_intentions(day_thought_list, keywords_list)
    index = 0
    for i, day_thought in processed_actions.items():
        if len(all_keywords[index])>max_keywords:
            processed_actions[i][-1] = all_keywords[index][:max_keywords]
        else:
            processed_actions[i][-1] = all_keywords[index]
        index += 1
    print(processed_actions)
    return processed_actions


# 构建进化树
# def build_tree(data_list, tree_len, threshold):
#     t = Tree()
#     new_node = (0, 0)
#     node_details = []

#     # 存储所有节点
#     nodes = {}
#     node_sentence = {}
#     nodes[new_node] = t.add_child(name=f"{-1}", dist=0.2)
#     node_sentence[new_node] = ['']

#     for i, data in data_list.items():
#         parent_node = (0, 0)
#         node_id = 0
#         for key, value in data.items():
#             if parent_node + (value[2], value[3]) in nodes: # 新节点已经存在
#                 new_node = parent_node + (value[2], value[3])
#                 sentence = words_to_sentence(value[4])
#                 words, word = get_embedding(node_sentence[new_node], sentence)
#                 sim_ans = check_similarity(word, words, threshold)
#                 if sim_ans == -1:
#                     node_sentence[new_node].append(sentence)
#                     node_id += 1
#                     parent_node_t = nodes[parent_node]
#                     nodes[new_node + (len(node_sentence[new_node]), )] = parent_node_t.add_child(name=f"rider{i},{node_id}", dist=value[3]/tree_len)
#                     node_details.append({
#                         "name": f"rider{i}_{node_id}",
#                         "rider_id": i,
#                         "node_sentence": node_sentence[new_node]
#                     })
#                 parent_node = new_node
#             elif parent_node in nodes:# 新节点不存在 父节点存在
#                 new_node = parent_node + (value[2], value[3])
#                 node_sentence[new_node] = [words_to_sentence(value[4])]
#                 node_id += 1
#                 parent_node_t = nodes[parent_node]
#                 nodes[new_node] = parent_node_t.add_child(name=f"rider{i},{node_id}", dist=value[3]/tree_len)
#                 node_details.append({
#                         "name": f"rider{i}_{node_id}",
#                         "rider_id": i,
#                         "node_sentence": node_sentence[new_node]
#                     })
#                 parent_node = new_node
#             else: # 新节点不存在 父节点不存在
#                 new_node = parent_node + (value[2], value[3])
#                 node_id += 1
#                 nodes[new_node] = t.add_child(name=f"rider{i},{node_id}", dist=value[3]/tree_len)
#                 node_details.append({
#                         "name": f"rider{i}_{node_id}",
#                         "rider_id": i,
#                         "node_sentence": node_sentence[new_node]
#                     })
#                 parent_node = new_node
#     print(t)
#     newick_str = t.write(format=5)
#     print(newick_str)
#     need_data = {
#         "newick_str" : newick_str,
#         "node_sentence": {str(key): value for key, value in node_sentence.items()},
#         "data_list": data_list,
#         "node_details": node_details
#     }
#     return need_data


def build_tree(data_list, tree_len, threshold):
    t = Tree()
    new_node = (0, 0)
    node_details = []

    # 存储所有节点
    nodes = {}
    node_sentence = {}
    nodes[new_node] = t.add_child(name="INT0", dist=0.2)  # 根节点编号为INT0
    node_sentence[new_node] = ['']
    
    internal_node_counter = 1  # 用于生成内部节点编号的计数器

    for i, data in data_list.items():
        parent_node = (0, 0)
        node_id = 0
        for key, value in data.items():
            if parent_node + (value[2], value[3]) in nodes:  # 新节点已经存在
                new_node = parent_node + (value[2], value[3])
                sentence = words_to_sentence(value[4])
                words, word = get_embedding(node_sentence[new_node], sentence)
                sim_ans = check_similarity(word, words, threshold)
                if sim_ans == -1:
                    node_sentence[new_node].append(sentence)
                    node_id += 1
                    parent_node_t = nodes[parent_node]
                    nodes[new_node + (len(node_sentence[new_node]), )] = parent_node_t.add_child(name=f"INT{internal_node_counter}", dist=value[3]/tree_len)
                    node_details.append({
                        "name": f"INT{internal_node_counter}",
                        "rider_id": i,
                        "node_sentence": node_sentence[new_node]
                    })
                    internal_node_counter += 1  # 更新内部节点编号
                parent_node = new_node
            elif parent_node in nodes:  # 新节点不存在 父节点存在
                new_node = parent_node + (value[2], value[3])
                node_sentence[new_node] = [words_to_sentence(value[4])]
                node_id += 1
                parent_node_t = nodes[parent_node]
                nodes[new_node] = parent_node_t.add_child(name=f"INT{internal_node_counter}", dist=value[3]/tree_len)
                node_details.append({
                    "name": f"INT{internal_node_counter}",
                    "rider_id": i,
                    "node_sentence": node_sentence[new_node]
                })
                internal_node_counter += 1  # 更新内部节点编号
                parent_node = new_node
            else:  # 新节点不存在 父节点不存在
                new_node = parent_node + (value[2], value[3])
                node_id += 1
                nodes[new_node] = t.add_child(name=f"INT{internal_node_counter}", dist=value[3]/tree_len)
                node_details.append({
                    "name": f"INT{internal_node_counter}",
                    "rider_id": i,
                    "node_sentence": node_sentence[new_node]
                })
                internal_node_counter += 1  # 更新内部节点编号
                parent_node = new_node
    # print(t)

    # 手动构建 Newick 字符串
    def build_newick(node, parent_name):
        children = node.children
        if children:
            child_names = ",".join([build_newick(child, node.name) for child in children])
            return f"({child_names}){node.name}:{node.dist}"
        else:
            return node.name + ":" + str(node.dist)

    newick_str = build_newick(t, "")
    print(newick_str)
    need_data = {
        "newick_str": newick_str,
        "node_sentence": {str(key): value for key, value in node_sentence.items()},
        "data_list": data_list,
        "node_details": node_details
    }
    folder_path = directory_path + f'/tree'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    with open(directory_path + f'/tree/tree_file.treefile', 'w') as json_file:
        json_file.write(newick_str)
    return need_data



def check_similarity(new_embedding, embeddings, threshold):
    """
    计算新向量与向量数组中每个向量的余弦相似度。
    如果新向量与数组中的任何一个向量相似度超过阈值，则返回0；
    如果都不相似，则返回-1。
    :param new_embedding: 新的向量，是一个PyTorch张量。
    :param embeddings: 多个向量的列表，每个向量是一个PyTorch张量。
    :param threshold: 相似度阈值。
    :return: 如果找到相似的向量返回0，否则返回-1。
    """
    # 将PyTorch张量转移到CPU并转换为NumPy数组
    new_embedding_np = new_embedding.cpu().numpy()
    embeddings_np = [emb.cpu().numpy() for emb in embeddings]
    
    # 计算新向量与每个向量的余弦相似度
    cosine_similarities = 1 - spatial.distance.cdist([new_embedding_np], embeddings_np, 'cosine')
    
    # 检查是否有任何相似度超过阈值
    if np.any(cosine_similarities > threshold):
        return 0
    else:
        return -1


def words_to_sentence(words):
    ans = ""
    for item in words:
        for word in item:
            ans += word+','
    return ans

def get_embedding(words, word):
    ans = ""
    model = SentenceTransformer('../../../../models/huggingface/all-MiniLM-L6-v2')
    embeddings = model.encode(words, convert_to_tensor=True)
    embedding = model.encode(word, convert_to_tensor=True)
    print('embeddings',embeddings)
    return embeddings, embedding


def json_to_tree(json_names, keywords, step_len, threshold, file_path):
    all_dict = {}
    all_sentence = []
    for i, n in enumerate(json_names):
        dict_node= get_drow_info(drow_route=n, keywords_list=keywords)
        all_dict[i] = dict_node
    
    ans = build_tree(all_dict, step_len, threshold)
    # 将数据写入JSON文件
    with open(file_path + f'/tree/tree {now_time}.json', 'w') as json_file:
        json.dump(ans, json_file, indent=4)

    print(f"JSON file created at {file_path}")


def write_to_csv(data, file_path):
    """
    将给定的数组写入CSV文件。

    :param data: 包含字典的数组，每个字典包含'rider_id'和'node_sentence'键。
    :param file_path: CSV文件的存储路径。
    """
    with open(file_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['id', 'node_sentence'])
        
        for item in data:
            riderid = item['name']
            nodesentence = item['node_sentence'][0] if item['node_sentence'] else '' 
            # 写入数据
            writer.writerow([riderid, nodesentence])
    

if __name__ == "__main__":
    directory_path = '../../entity/agent_log/2024-12-12 00:01:03.629304'
    starts_time = time.time()
    # 使用glob模块找到目录下所有的.json文件
    json_files = glob.glob(os.path.join(directory_path, '*.json'))
    # json_files = ['/home/syf/project/MACE/MACE/simulation/examples/SocialInvolution/entity/agent_log/2_thought.json']

    # 初始化一个空数组来存储JSON文件的路径
    json_names = []
    now_time =datetime.datetime.now()
    # 将找到的JSON文件路径添加到数组中
    for file_path in json_files:
        json_names.append(file_path)

    candidate_keywords = [
        [
            "more money",
            "more rest",
            "Comfortable",
            "Lucrative"
            "Balanced",
            "Flexible",
            "Fixed"
            "Rewarding",
            "Stimulating",
            "Increase",
            "decrease",
            "unchanged",
            "hardworking",
            "Competitive",  # 好胜心强的
            "Lazy",  # 懒惰的
        ],
        [
            "more money",
            "less money"
            "Profit-driven",  # 追求高利润的
            "hardworking"
            "Competitive",  # 好胜心强的
            "Lazy",  # 懒惰的
            "Indifferent",  # 躺平的（这里用Indifferent表示一种无动于衷或不关心的态度）
            "more order",
            "less order"
        ]
    ]
    #测试1
    json_to_tree(json_names, candidate_keywords, 3600, 0.3, directory_path)

    end_time = time.time()
    print('测试1', end_time - starts_time)
    # 测试2
    with open(directory_path + f'/tree/tree.json', 'r', encoding='utf-8') as file:
        data = json.load(file)
    write_to_csv(data["node_details"], directory_path + f'/tree/ans.csv')
