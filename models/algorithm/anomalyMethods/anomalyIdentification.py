"""
# 异常检测算法，用于检测序列中的异常值
# 为了方便对接前端展示，异常检测算法返回值均为数值对数组的形式[(x, y)]
# 其中x为异常点位置，y为异常点值，例如数组中第x0个数据y0发生异常，那么返回值数组中会加入数值对(x0,y0)

目前异常检测支持的算法包括：
3-sigma
z-score
箱线法
Grubbs假设检验方法
KNN方法
LOF方法
COF方法
SOS方法
DBSCAN方法
iForest方法
PCA方法
AutoEncoder方法
One-Class SVM方法
ARIMA方法
熵模型分析
熵模型分析_H0
"""
import math

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors, LocalOutlierFactor
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from statsmodels.tsa.arima.model import ARIMA
import torch
import torch.nn as nn
import torch.optim as optim
from pyod.models.cof import COF
from pyod.models.sos import SOS


# 生成起始到终止的一些随机数
def generate_random_data(start, end, size, seed=0):
    np.random.seed(seed)
    return np.random.uniform(start, end, size).tolist()


# 向列表中添加一些额外的异常值
def add_outliers(data, outliers):
    return data + outliers


# 3 sigma 方法
def detect_outliers_3sigma(data):
    mean = np.mean(data)
    std_dev = np.std(data)
    three_sigma_upper = mean + 3 * std_dev
    three_sigma_lower = mean - 3 * std_dev
    return [(i, x) for i, x in enumerate(data) if x < three_sigma_lower or x > three_sigma_upper]


# z-score 方法
def detect_outliers_zscore(data, threshold=2):
    z_scores = np.abs(stats.zscore(data))
    return [(i, data[i]) for i in range(len(data)) if z_scores[i] > threshold]


# 箱线图 (Boxplot) 方法
def detect_outliers_boxplot(data):
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1
    boxplot_lower = Q1 - 1.5 * IQR
    boxplot_upper = Q3 + 1.5 * IQR
    return [(i, x) for i, x in enumerate(data) if x < boxplot_lower or x > boxplot_upper]


# Grubbs 假设检验方法
def detect_outliers_grubbs(data, alpha=0.05):
    def grubbs_test(data):
        N = len(data)
        mean_data = np.mean(data)
        std_data = np.std(data)
        G = np.max(np.abs(data - mean_data)) / std_data
        t_dist = stats.t.ppf(1 - alpha / (2 * N), N - 2)
        G_critical = ((N - 1) / np.sqrt(N)) * np.sqrt(t_dist ** 2 / (N - 2 + t_dist ** 2))
        return G, G_critical, G > G_critical

    grubbs_outliers = []
    data_copy = np.array(data)
    while True:
        G, G_critical, is_outlier = grubbs_test(data_copy)
        if is_outlier:
            outlier = data_copy[np.argmax(np.abs(data_copy - np.mean(data_copy)))]
            grubbs_outliers.append(outlier)
            data_copy = data_copy[data_copy != outlier]
        else:
            break
    return grubbs_outliers


# KNN 方法
# 通过计算每个数据点到其最近邻居的平均距离，并将这些距离与一个指定的阈值进行比较，以确定哪些点是异常值。
# 可以调整k值（最近邻居的数量）和阈值来改变异常检测的灵敏度
def detect_outliers_knn(data, k=5, threshold=2):
    data = np.array(data).reshape(-1, 1)
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(data)
    distances, _ = nbrs.kneighbors(data)
    distance_to_neighbors = distances[:, 1:].mean(axis=1)
    outlier_threshold = np.percentile(distance_to_neighbors, 100 - threshold)
    return [(i, data[i][0]) for i in range(len(data)) if distance_to_neighbors[i] > outlier_threshold]


# LOF 方法
def detect_outliers_lof(data, n_neighbors=20):
    data = np.array(data).reshape(-1, 1)
    lof = LocalOutlierFactor(n_neighbors=n_neighbors)
    outliers = lof.fit_predict(data)
    return [(i, data[i][0]) for i in range(len(data)) if outliers[i] == -1]


# COF 方法
def detect_outliers_cof(data, n_neighbors=20):
    data = np.array(data).reshape(-1, 1)
    cof = COF(n_neighbors=n_neighbors)
    cof.fit(data)
    outlier_scores = cof.decision_function(data)
    threshold = np.percentile(outlier_scores, 95)
    return [(i, data[i][0]) for i in range(len(data)) if outlier_scores[i] > threshold]


# SOS 方法
def detect_outliers_sos(data):
    data = np.array(data).reshape(-1, 1)
    sos = SOS()
    sos.fit(data)
    outlier_scores = sos.decision_function(data)
    threshold = np.percentile(outlier_scores, 95)
    return [(i, data[i][0]) for i in range(len(data)) if outlier_scores[i] > threshold]


# DBSCAN 方法
def detect_outliers_dbscan(data, eps=0.5, min_samples=5):
    data = np.array(data).reshape(-1, 1)
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(data)
    labels = db.labels_
    return [(i, data[i][0]) for i in range(len(data)) if labels[i] == -1]


# iForest 方法
def detect_outliers_iforest(data, contamination=0.1):
    data = np.array(data).reshape(-1, 1)
    iforest = IsolationForest(contamination=contamination)
    outliers = iforest.fit_predict(data)
    return [(i, data[i][0]) for i in range(len(data)) if outliers[i] == -1]


# PCA 方法
def detect_outliers_pca(data, n_components=1, threshold=3):
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(np.array(data).reshape(-1, 1))
    pca = PCA(n_components=n_components)
    pca.fit(data_scaled)
    data_reconstructed = pca.inverse_transform(pca.transform(data_scaled))
    reconstruction_error = np.abs(data_scaled - data_reconstructed)
    outliers = reconstruction_error > np.percentile(reconstruction_error, 100 - threshold)
    return [(i, data[i]) for i in range(len(data)) if outliers[i]]


# AutoEncoder 方法（使用 PyTorch 实现）
class AutoEncoder(nn.Module):
    def __init__(self, input_dim, encoding_dim):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, encoding_dim),
            nn.Tanh()
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, input_dim),
            nn.Tanh()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def detect_outliers_autoencoder(data, epochs=100, batch_size=10, threshold=0.1):
    data = np.array(data).reshape(-1, 1).astype(np.float32)
    input_dim = data.shape[1]
    encoding_dim = 2

    autoencoder = AutoEncoder(input_dim, encoding_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(autoencoder.parameters(), lr=0.01)

    data_tensor = torch.tensor(data)

    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = autoencoder(data_tensor)
        loss = criterion(outputs, data_tensor)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        reconstructed = autoencoder(data_tensor).numpy()
    reconstruction_error = np.mean(np.power(data - reconstructed, 2), axis=1)
    outliers = reconstruction_error > threshold
    return [(i, float(data[i][0])) for i in range(len(data)) if outliers[i]]


# One-Class SVM 方法
def detect_outliers_ocsvm(data, nu=0.05, kernel='rbf', gamma='scale'):
    data = np.array(data).reshape(-1, 1)
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    ocsvm = OneClassSVM(nu=nu, kernel=kernel, gamma=gamma)
    outliers = ocsvm.fit_predict(data_scaled)
    return [(i, data[i][0]) for i in range(len(data)) if outliers[i] == -1]


# ARIMA 预测并检测异常
def detect_outliers_arima(data, order=(5, 1, 0), threshold=2):
    model = ARIMA(data, order=order)
    model_fit = model.fit()
    prediction = model_fit.predict(start=0, end=len(data) - 1, dynamic=False)
    residuals = np.abs(data - prediction)
    mean_residual = np.mean(residuals)
    std_residual = np.std(residuals)
    return [(i, data[i]) for i in range(len(data)) if residuals[i] > mean_residual + threshold * std_residual]


def calc_entropy(utility_t, interval_len):
    N = len(utility_t)
    categories = np.zeros(1000)
    for utility in utility_t:
        categories[int(utility / interval_len)] += 1

    tmp_num = h = 0
    for num in categories:
        if num == 0:
            continue
        p = num / N
        h -= p * math.log2(p)
        tmp_num += num
        if tmp_num == N:
            break
    return h


# 熵模型_Pre: value entropy = exp(1 - |ht-h_best|/h_best)
def detect_outliers_value_entropy_pre(system_data, individual_data, interval_len=0.5):
    value_entropy_list = []
    h_t = 0
    h_best = 0
    max_len = 0
    # 数据清洗，使其长度一致
    for data in individual_data:
        max_len = max(max_len, len(data))

    for i in range(len(individual_data)):
        while len(individual_data[i]) < max_len:
            individual_data[i].append(0)
    individual_data = np.array(individual_data)

    N = len(individual_data)
    Ht_list = []
    for t in range(len(individual_data[0])):
        # 计算 Ht
        h_t = calc_entropy(utility_t=individual_data[:, t], interval_len=interval_len)
        # 计算 H_best
        h_best = 1 / 2 * math.log2(N)
        Ht_list.append(h_t)
        value_entropy_list.append(math.exp(1 - math.fabs((h_t - h_best) / h_best)))
    print('ht_list', Ht_list)
    print('value_entropy_list', value_entropy_list)
    arr_x = []
    for i in range(len(value_entropy_list)):
        arr_x.append(i)

    # 绘制ht和value entropy的图
    plt.figure(figsize=(7, 10))
    plt.subplot(2, 1, 1)
    plt.plot(arr_x, value_entropy_list, marker='o', linestyle='-', color='b', label='数据线')
    plt.ylabel('Value Entropy')

    plt.subplot(2, 1, 2)
    plt.plot(arr_x, Ht_list, marker='o', linestyle='-', color='b', label='数据线')
    plt.ylabel('Ht')
    plt.show()

    res = []
    for i, value in enumerate(value_entropy_list):
        if value < 1.5:
            if len(system_data) <= i:
                res.append((i, 0))
            else:
                res.append((i, system_data[i]))
    print('熵模型异常点', res)
    return res, value_entropy_list


# 熵模型_New: value entropy = exp(|h0 - h_best|/h_best - |ht-h_best|/h_best)
def detect_outliers_value_entropy_new(system_data, individual_data_train, individual_data_test, interval_len=0.5):
    value_entropy_list = []
    h_t = 0
    h_best = 0
    h_0 = 0
    max_len = 0
    # 数据清洗，使其长度一致
    for data in individual_data_train:
        max_len = max(max_len, len(data))
    for data in individual_data_test:
        max_len = max(max_len, len(data))

    for i in range(len(individual_data_train)):
        while len(individual_data_train[i]) < max_len:
            individual_data_train[i].append(0)
    individual_data_train = np.array(individual_data_train)

    for i in range(len(individual_data_test)):
        while len(individual_data_test[i]) < max_len:
            individual_data_test[i].append(0)
    individual_data_test = np.array(individual_data_test)

    N = len(individual_data_train)
    Ht_list = []
    for t in range(len(individual_data_train[0])):
        # 计算 Ht
        h_t = calc_entropy(utility_t=individual_data_train[:, t], interval_len=interval_len)
        # 计算 H0
        h_0 = calc_entropy(utility_t=individual_data_test[:, t], interval_len=interval_len)
        # 计算 H_best
        h_best = 1 / 2 * math.log2(N)
        Ht_list.append(h_t)
        value_entropy_list.append(math.exp(math.fabs((h_0 - h_best) / h_best) - math.fabs((h_t - h_best) / h_best)))

    print('ht_list', Ht_list)
    print('value_entropy_list', value_entropy_list)
    arr_x = []
    for i in range(len(value_entropy_list)):
        arr_x.append(i)

    plt.figure(figsize=(7, 10))
    plt.subplot(2, 1, 1)
    plt.plot(arr_x, value_entropy_list, marker='o', linestyle='-', color='b', label='数据线')
    plt.ylabel('Value Entropy')

    plt.subplot(2, 1, 2)
    plt.plot(arr_x, Ht_list, marker='o', linestyle='-', color='b', label='数据线')
    plt.ylabel('Ht')
    plt.show()

    res = []
    for i, value in enumerate(value_entropy_list):
        if value < 1.5:
            if len(system_data) <= i:
                res.append((i, 0))
            else:
                res.append((i, system_data[i]))
    print('熵模型异常点', res)
    return res, value_entropy_list


if __name__ == "__main__":
    # 示例使用
    # 生成数据
    data = generate_random_data(40, 50, 100)
    data_with_outliers = add_outliers(data, [30, 35, 59, 62])

    # 检测异常值
    outliers_3sigma = detect_outliers_3sigma(data_with_outliers)
    outliers_zscore = detect_outliers_zscore(data_with_outliers)
    outliers_boxplot = detect_outliers_boxplot(data_with_outliers)
    outliers_grubbs = detect_outliers_grubbs(data_with_outliers)
    outliers_knn = detect_outliers_knn(data_with_outliers)
    outliers_lof = detect_outliers_lof(data_with_outliers)
    outliers_cof = detect_outliers_cof(data_with_outliers)
    outliers_sos = detect_outliers_sos(data_with_outliers)
    outliers_dbscan = detect_outliers_dbscan(data_with_outliers)
    outliers_iforest = detect_outliers_iforest(data_with_outliers)
    outliers_pca = detect_outliers_pca(data_with_outliers)
    outliers_autoencoder = detect_outliers_autoencoder(data_with_outliers)
    outliers_ocsvm = detect_outliers_ocsvm(data_with_outliers)
    outliers_arima = detect_outliers_arima(data_with_outliers)

    # 打印结果
    print("3 sigma 方法识别的异常值:", outliers_3sigma)
    print("z-score 方法识别的异常值:", outliers_zscore)
    print("箱线图方法识别的异常值:", outliers_boxplot)
    print("Grubbs 假设检验识别的异常值:", outliers_grubbs)
    print("KNN 方法识别的异常值:", outliers_knn)
    print("LOF 方法识别的异常值:", outliers_lof)
    print("COF 方法识别的异常值:", outliers_cof)
    print("SOS 方法识别的异常值:", outliers_sos)
    print("DBSCAN 方法识别的异常值:", outliers_dbscan)
    print("iForest 方法识别的异常值:", outliers_iforest)
    print("PCA 方法识别的异常值:", outliers_pca)
    print("AutoEncoder 方法识别的异常值:", outliers_autoencoder)
    print("One-Class SVM 方法识别的异常值:", outliers_ocsvm)
    print("ARIMA 方法识别的异常值:", outliers_arima)

    # 可视化数据和异常值
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.hist(data_with_outliers, bins=30, color='blue', alpha=0.7)
    plt.axvline(np.mean(data_with_outliers), color='black', linestyle='--', label='Mean')
    plt.axvline(np.mean(data_with_outliers) + 3 * np.std(data_with_outliers), color='red', linestyle='--',
                label='3σ upper bound')
    plt.axvline(np.mean(data_with_outliers) - 3 * np.std(data_with_outliers), color='red', linestyle='--',
                label='3σ lower bound')
    plt.legend()
    plt.title('Histogram of Data with 3 sigma Bounds')

    plt.subplot(1, 2, 2)
    plt.boxplot(data_with_outliers, vert=False)
    plt.title('Boxplot of Data')

    plt.show()
