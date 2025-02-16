"""
# graphGenerator用于生成不同种类的网络图
# 函数返回结果均为 邻接矩阵

网络图生成器支持的网络类型包括：
孤立节点网络
全局耦合网络
星型网络
一维环
二维晶格（节点数量必须为n^2）
G(N, L)随机网络
G(N, P)随机网络
WS小世界网络
NW小世界网络
BA无标度网络
"""
import math
import random


# 规则网络1 孤立节点网络（每个节点之间无连接）
def isolated_node(N):
    mat = [[0 for _ in range(N)] for __ in range(N)]
    return mat


# 规则网络2 全局耦合网络（每个节点之间都存在连接，即完全图）
def complete_graph(N):
    mat = [[1 for _ in range(N)] for __ in range(N)]
    return mat


# 规则网络3 星型网络（菊花图）
def star_size_graph(N):
    index = random.randint(0, N - 1)
    mat = [[0 for _ in range(N)] for __ in range(N)]
    for i in range(N):
        if i != index:
            mat[i][index] = mat[index][i] = 1
    return mat


# 规则网络4 一维环
def one_dim_ring(N):
    mat = [[0 for _ in range(N)] for __ in range(N)]
    for i in range(N):
        mat[i][(i + 1) % N] = mat[(i + 1) % N][i] = 1
    return mat


# 规则网络5 二维晶格
def two_dim_lattice(N):
    n = int(math.sqrt(N))
    assert n * n == N, '二维晶格模型中，节点数应为完全平方数'
    mat = [[0 for _ in range(N)] for __ in range(N)]
    for i in range(N):
        if (i + 1) % n:
            mat[i][i + 1] = mat[i + 1][i] = 1
        if i < N - n:
            mat[i][i + n] = mat[i + n][i] = 1
    return mat


# 随机网络1 G(N, L)模型: N个节点通过L条随机放置的链接彼此相连
def G_N_L(N, L):
    assert L <= N * (N - 1) / 2, '边数量超过最大边数'
    mat = [[0 for _ in range(N)] for __ in range(N)]
    num = 0
    while num < L:
        u = random.randint(0, N - 1)
        v = random.randint(0, N - 1)
        if u == v or mat[u][v] == 1:
            continue
        else:
            mat[u][v] = mat[v][u] = 1
            num += 1
    return mat


# 随机网络2 G(N, P)模型: N个节点中，每对节点之间以p的概率相连
def G_N_P(N, P):
    assert 0 <= P <= 1, '概率P必须在0到1之间'
    mat = [[0 for _ in range(N)] for __ in range(N)]
    for i in range(N):
        for j in range(i + 1, N):
            rd = random.random()
            if rd < P:
                mat[i][j] = mat[j][i] = 1
    return mat


# WS小世界网络
def small_world_ws(N, K, P):
    assert K % 2 == 0, '邻近范围K必须为偶数'
    assert K <= N, '邻近范围K必须小于节点数N'
    assert 0 <= P <= 1, '概率P必须在0到1之间'
    mat = [[0 for _ in range(N)] for __ in range(N)]
    for i in range(N):
        for j in range(1, int(K / 2) + 1):
            index = ((i + j) % N + N) % N
            mat[i][index] = mat[index][i] = 1
            index = ((i - j) % N + N) % N
            mat[i][index] = mat[index][i] = 1

    for i in range(N):
        for j in range(i+1, N):
            rd = random.random()
            if rd < P:
                mat[i][j] = mat[j][i] = 0
                index = random.randint(0, N - 1)
                while i == index or mat[i][index]:
                    index = random.randint(0, N - 1)
                mat[i][index] = mat[index][i] = 1
    return mat


# NW小世界网络
def small_world_nw(N, K, P):
    assert K % 2 == 0, '邻近范围K必须为偶数'
    assert K <= N, '邻近范围K必须小于节点数N'
    assert 0 <= P <= 1, '概率P必须在0到1之间'
    mat = [[0 for _ in range(N)] for __ in range(N)]
    for i in range(N):
        for j in range(1, int(K / 2) + 1):
            index = ((i + j) % N + N) % N
            mat[i][index] = mat[index][i] = 1
            index = ((i - j) % N + N) % N
            mat[i][index] = mat[index][i] = 1

    for i in range(N):
        for j in range(i+1, N):
            rd = random.random()
            if rd < P and mat[i][j] == 0:
                mat[i][j] = mat[j][i] = 1
    return mat


# BA无标度网络
def scale_free_ba(N):
    assert N >= 2, '无标度网络中，节点数必须>=2'
    mat = [[0 for _ in range(N)] for __ in range(N)]
    mat[0][1] = mat[1][0] = 1
    seq = [0, 1]
    for i in range(2, N):
        # 从度数序列中取出两个不相同的元素
        rand_elements = [random.choice(seq)]
        rand_element = random.choice(seq)
        while rand_element in rand_elements:
            rand_element = random.choice(seq)
        rand_elements.append(rand_element)
        for index in rand_elements:
            print(f'add edge: ({i}, {index})')
            mat[i][index] = mat[index][i] = 1
            seq.append(index)
            seq.append(i)
        print(sorted(seq))
    return mat


if __name__ == '__main__':
    m = scale_free_ba(10)
    print(m)