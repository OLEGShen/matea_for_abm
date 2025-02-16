import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from shap import KernelExplainer, summary_plot
import shap
from statsmodels.tsa.arima.model import ARIMA
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error
import xgboost


# 生成随机数据
def generate_random_data(start, end, size, seed=0):
    np.random.seed(seed)
    return np.random.uniform(start, end, size)


# 向数据中添加异常值
def add_outliers(data, outliers, positions):
    for i, pos in enumerate(positions):
        data[pos] = outliers[i]
    return data


# 计算加权和
def weighted_sum(data, weights):
    return np.dot(data, weights)


# 检测加权和的异常值
def detect_outliers_weighted_sum(weighted_sums, threshold=3):
    mean = np.mean(weighted_sums)
    std_dev = np.std(weighted_sums)
    outliers = [x for x in weighted_sums if x > mean + threshold * std_dev or x < mean - threshold * std_dev]
    return outliers


# 计算各个二级指标的贡献度
def calculate_contributions(data, weights, threshold=3):
    contributions = np.zeros_like(data)
    for i in range(data.shape[1]):
        temp_weights = weights.copy()
        temp_weights[i] = 0
        temp_weighted_sum = weighted_sum(data, temp_weights)
        deviations = weighted_sum(data, weights) - temp_weighted_sum
        contributions[:, i] = deviations
    return contributions


# 计算各个二级指标的Shapley值贡献度
def calculate_shap_contributions(data, weights):
    def model(X):
        return np.dot(X, weights)

    explainer = KernelExplainer(model, data)
    shap_values = explainer.shap_values(data)
    return shap_values


# 示例使用
# 生成四个二级指标的数据
data1 = generate_random_data(40, 50, 100)
data2 = generate_random_data(30, 40, 100)
data3 = generate_random_data(50, 60, 100)
data4 = generate_random_data(20, 30, 100)

# 组合成一个矩阵
data = np.vstack((data1, data2, data3, data4)).T

# 添加异常值
outliers = [100, 90, 110, 80]
positions = [10, 20, 30, 40]
data_with_outliers = data.copy()
data_with_outliers = add_outliers(data_with_outliers, outliers, positions)
print('data_with_outliers', data_with_outliers)
print('data.shape', data_with_outliers.shape)
# 设置权重
weights = np.array([0.25, 0.25, 0.25, 0.25])

# 计算加权和
weighted_sums = weighted_sum(data_with_outliers, weights)
print('weighted_sum', weighted_sums)

# 检测加权和的异常值
outliers_weighted_sum = detect_outliers_weighted_sum(weighted_sums)

# 计算各个二级指标的贡献度
contributions = calculate_contributions(data_with_outliers, weights)

# 打印结果
print("加权和的异常值:", outliers_weighted_sum)

for i, outlier in enumerate(outliers):
    print(f"在位置 {positions[i]}，异常值 {outlier} 的贡献度为: {contributions[positions[i]]}")

# 计算各个二级指标的Shapley值贡献度
shap_contributions = calculate_shap_contributions(data_with_outliers, weights)
print('shap_contributions', shap_contributions)

# 打印结果
print("加权和的异常值:", outliers_weighted_sum)

for i, outlier in enumerate(outliers):
    print(f"在位置 {positions[i]}，异常值 {outlier} 的Shapley值贡献度为: {shap_contributions[positions[i]]}")

# 可视化贡献度
plt.figure(figsize=(12, 6))
for i in range(data.shape[1]):
    plt.plot(contributions[:, i], label=f'contribution of indicator {i + 1}')
plt.legend()
plt.title('Secondary indicator contribution')
plt.xlabel('sample index')
plt.ylabel('contribution')
plt.show()


# ARIMA 模型进行时间序列预测并计算残差
def arima_residuals(data, order=(5, 1, 0)):
    model = ARIMA(data, order=order)
    model_fit = model.fit()
    prediction = model_fit.predict(start=0, end=len(data) - 1, dynamic=False)
    residuals = data - prediction
    return residuals


# LSTM 自动编码器模型
class LSTM_Autoencoder(nn.Module):
    def __init__(self, n_features, seq_len, embedding_dim=64):
        super(LSTM_Autoencoder, self).__init__()
        self.encoder = nn.LSTM(
            input_size=n_features,
            hidden_size=embedding_dim,
            num_layers=1,
            batch_first=True
        )
        self.decoder = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=n_features,
            num_layers=1,
            batch_first=True
        )

    def forward(self, x):
        x, (hidden, cell) = self.encoder(x)
        x, (hidden, cell) = self.decoder(x)
        return x


# 使用LSTM自动编码器进行多变量时间序列异常检测
def lstm_autoencoder_residuals(data, seq_len=10, embedding_dim=64, epochs=50, lr=0.001):
    n_samples, n_features = data.shape
    model = LSTM_Autoencoder(n_features, seq_len, embedding_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    data_scaled = torch.tensor(data_scaled, dtype=torch.float32)

    # 准备序列数据
    def create_sequences(data, seq_len):
        xs = []
        for i in range(len(data) - seq_len):
            x = data[i:i + seq_len]
            xs.append(x)
        return torch.stack(xs)

    sequences = create_sequences(data_scaled, seq_len)

    # 训练模型
    model.train()
    for epoch in range(epochs):
        for seq in sequences:
            optimizer.zero_grad()
            seq = seq.unsqueeze(0)
            y_pred = model(seq)
            loss = criterion(y_pred, seq)
            loss.backward()
            optimizer.step()

    # 计算残差
    model.eval()
    residuals = []
    with torch.no_grad():
        for seq in sequences:
            seq = seq.unsqueeze(0)
            y_pred = model(seq)
            residuals.append(criterion(y_pred, seq).item())

    return np.array(residuals)


# 结合多变量分析检测异常
def detect_anomalies_multivariate(data, threshold=2, method='arima'):
    residuals_list = []
    if method == 'arima':
        for i in range(data.shape[1]):
            residuals = arima_residuals(data[:, i])
            residuals_list.append(residuals)
    elif method == 'lstm':
        residuals = lstm_autoencoder_residuals(data)
        residuals_list.append(residuals)

    residuals_matrix = np.vstack(residuals_list).T
    anomalies = np.any(np.abs(residuals_matrix) > threshold * np.std(residuals_matrix, axis=0), axis=1)
    return np.where(anomalies)[0]


# 检测多变量时间序列中的异常值 (ARIMA)
anomalies_arima = detect_anomalies_multivariate(data_with_outliers, method='arima')
print("多变量时间序列中的异常时间点 (ARIMA):", anomalies_arima)

# 检测多变量时间序列中的异常值 (LSTM)
anomalies_lstm = detect_anomalies_multivariate(data_with_outliers, method='lstm')
print("多变量时间序列中的异常时间点 (LSTM):", anomalies_lstm)


# 计算Shapley值
def calculate_shap_contributions(data, model, anomalous_point):
    explainer = shap.Explainer(model)
    shap_values = explainer(data)
    return shap_values.values[anomalous_point]


# 计算每个指标的残差
residuals_list = []
for i in range(data_with_outliers.shape[1]):
    residuals = arima_residuals(data_with_outliers[:, i])
    residuals_list.append(residuals)

# 组合残差矩阵
residuals_matrix = np.vstack(residuals_list).T

# 重新训练模型以残差为特征
model_residuals = xgboost.XGBRegressor()
model_residuals.fit(residuals_matrix, weighted_sums)

# 计算Shapley值
shap_values = calculate_shap_contributions(residuals_matrix, model_residuals, 10)

# 打印和可视化Shapley值
print(f"Shapley values for point: {shap_values}")
