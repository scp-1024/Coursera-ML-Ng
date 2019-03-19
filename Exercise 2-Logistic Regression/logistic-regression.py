import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import Sigmoid

path = 'ex2data1.txt'
data = pd.read_csv(path, names=('exam1', 'exam2', 'admitted'))


# 可视化观察数据
def plotData(data):
    cols = data.shape[1]
    feature = data.iloc[:, 0:cols - 1]
    label = data.iloc[:, cols - 1]
    # iloc 根据列的位置索引来切片
    postive = feature[label == 1]
    negtive = feature[label == 0]

    plt.figure(figsize=(8, 5))
    plt.scatter(postive.iloc[:, 0], postive.iloc[:, 1])
    plt.scatter(negtive.iloc[:, 0], negtive.iloc[:, 1], c='r', marker='x')
    plt.legend(['Admitted', 'Not admitted'], loc=1)
    plt.xlabel('Exam1 score')
    plt.ylabel('Exam2 score')
    plt.show()


# 进一步准备数据，对结构初始化
data.insert(0, 'One', 1)
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values
theta = np.zeros(X.shape[1]) # 注意这里theta创建的是一维的数组，对于ndarray一定要注意一维时它的shape（和matrix有很大区别）


def cost(theta, X, y):
    first = (-y) * np.log(Sigmoid.sigmoid(X @ theta))  # 这里*号是对应位置相乘而不是矩阵运算
    second = (1 - y) * np.log(1 - Sigmoid.sigmoid(X @ theta))
    return np.mean(first - second)

# 也可以用矩阵实现，但建议使用ndarray
# cols = data.shape[1]
# X=np.matrix(data.iloc[:,0:cols-1].values)
# y=np.matrix(data.iloc[:,cols-1].values)
# theta=np.matrix(np.zeros((X.shape[1],1)))
# def cost(theta, X, y):
#     first = np.multiply(-y, np.log(Sigmoid.sigmoid(X @ theta)))  # 这里*号是对应位置相乘而不是矩阵运算
#     second = np.multiply(1 - y, np.log(1 - Sigmoid.sigmoid(X @ theta)))
#     return np.mean(first - second)


print(cost(theta, X, y))
