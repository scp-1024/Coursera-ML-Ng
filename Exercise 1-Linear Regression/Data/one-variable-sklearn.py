'''
利用scikit-learn来实现线性回归
'''
from sklearn import linear_model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#导入数据集
path = 'Data\ex1data1.txt'

# 变量data数据结构为DataFrame
# read_csv()：将csv文件导入到DataFrame
# csv文件简单来说就是：纯文本，以行为一条记录，每条记录被分隔符分隔为字段
data = pd.read_csv(
    path, header=None,
    names=['Population',
           'Profit'])  # names为列名，如果不自己设置，就会默认从0列开始。如果header=None不设置，将不会有列名。

#插入第1列：X0=1
data.insert(0, 'one', 1)

cols = data.shape[1]  # .shape返回一个元组，[0]为行数，[2]为列数
# 提取X，y的值
X = data.iloc[:, 0:cols - 1]
y = data.iloc[:, cols - 1:cols]
# IndexError: single positional indexer is out-of-bounds
# 输入的列号超过了索引值范围
'''
数据处理
'''
# 将dataframe结构转化成np的matrix
# 当Theta取0时计算平均误差
X = np.matrix(X.values)
y = np.matrix(y.values)
model = linear_model.LinearRegression()
model.fit(X, y)  # 根据提供的方法来拟合数组，并将系数矩阵存储在成员变量coef_中
x = np.array(X[:, 1].A1)  # X[:, 1].A1：把向量X[:, 1]转化为一维数组
f = model.predict(X).flatten()

plt.figure(figsize=(8, 5))
plt.plot(x, f, 'r', label='Prediction')
plt.scatter(data.Population, data.Profit, label='Training Data')
plt.xlabel('Population')
plt.ylabel('Profit')
plt.title('Predicted Profit vs. Population Size')

plt.show()