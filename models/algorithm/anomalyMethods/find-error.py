#1.敏感性分析，因素分析，通过改变某个因素，确定具体原因
#2.通过具体的数据集，训练得到多标签的原因，如决策树

#1.敏感性分析
import pandas as pd
import numpy as np
from itertools import combinations, product

# 数据集
data = {
    '人群分布': [500, 1000, 200, 800, 300, 1200],
    '基站分布密度': [10, 15, 8, 12, 9, 20],
    '基站之间的交互强度': ['高', '中', '低', '高', '中', '高'],
    '覆盖范围调节': ['低', '中', '高', '中', '低', '低'],
    '异常标志': [1, 1, 0, 1, 0, 1]
}

df = pd.DataFrame(data)

# 将分类数据转换为数值数据
df['基站之间的交互强度'] = df['基站之间的交互强度'].map({'低': 0, '中': 1, '高': 2})
df['覆盖范围调节'] = df['覆盖范围调节'].map({'低': 0, '中': 1, '高': 2})


# 定义一个函数来模拟改变某一个因素后的异常检测
def simulate_change(dataframe, feature, new_value):
    temp_df = dataframe.copy()
    temp_df[feature] = new_value
    return temp_df['异常标志'].sum()  # 返回异常数量


# 初始异常数量
initial_anomalies = df['异常标志'].sum()

# 存储每个因素的敏感性结果
sensitivity_results = {}

# 对每个特征进行敏感性分析
for feature in ['人群分布', '基站分布密度', '基站之间的交互强度', '覆盖范围调节']:
    original_values = df[feature].unique()
    for value in original_values:
        anomalies_after_change = simulate_change(df, feature, value)
        sensitivity_results[(feature, value)] = initial_anomalies - anomalies_after_change

# 输出敏感性分析结果
for key, value in sensitivity_results.items():
    print(f'改变 {key[0]} 为 {key[1]} 后，异常减少 {value} 个')

from itertools import combinations


# 定义一个函数来模拟改变多个因素后的异常检测
def simulate_multiple_changes(dataframe, features_values):
    temp_df = dataframe.copy()
    for feature, value in features_values:
        temp_df[feature] = value
    return temp_df['异常标志'].sum()  # 返回异常数量


# 存储每个因素组合的敏感性结果
multi_sensitivity_results = {}

# 对每个特征组合进行敏感性分析
features = ['人群分布', '基站分布密度', '基站之间的交互强度', '覆盖范围调节']
for comb_length in range(2, len(features) + 1):
    for feature_comb in combinations(features, comb_length):
        original_values_combinations = [df[feature].unique() for feature in feature_comb]
        for values_comb in product(*original_values_combinations):
            anomalies_after_change = simulate_multiple_changes(df, zip(feature_comb, values_comb))
            multi_sensitivity_results[(feature_comb, values_comb)] = initial_anomalies - anomalies_after_change

# 输出多因素敏感性分析结果
for key, value in multi_sensitivity_results.items():
    print(f'改变 {key[0]} 为 {key[1]} 后，异常减少 {value} 个')
#

#2.数据集训练
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

# 数据集
data = {
    '人群分布': [500, 1000, 200, 800, 300, 1200],
    '基站分布密度': [10, 15, 8, 12, 9, 20],
    '基站之间的交互强度': ['高', '中', '低', '高', '中', '高'],
    '覆盖范围调节': ['低', '中', '高', '中', '低', '低'],
    '基站过密': [0, 1, 0, 0, 0, 1],
    '人群分布过密': [0, 0, 0, 1, 0, 0],
    '信号干扰': [0, 1, 0, 0, 0, 1]
}

df = pd.DataFrame(data)

# 将分类数据转换为数值数据
df['基站之间的交互强度'] = df['基站之间的交互强度'].map({'低': 0, '中': 1, '高': 2})
df['覆盖范围调节'] = df['覆盖范围调节'].map({'低': 0, '中': 1, '高': 2})

X = df[['人群分布', '基站分布密度', '基站之间的交互强度', '覆盖范围调节']]  # 特征
y = df[['基站过密', '人群分布过密', '信号干扰']]  # 多标签

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建决策树分类器
dt_clf = DecisionTreeClassifier(criterion='entropy', max_depth=3)

# 创建多输出分类器
multi_target_clf = MultiOutputClassifier(dt_clf, n_jobs=-1)

# 训练模型
multi_target_clf.fit(X_train, y_train)

# 对测试集进行预测
y_pred = multi_target_clf.predict(X_test)

# 输出分类报告
print(classification_report(y_test, y_pred, zero_division=1, target_names=['基站过密', '人群分布过密', '信号干扰']))