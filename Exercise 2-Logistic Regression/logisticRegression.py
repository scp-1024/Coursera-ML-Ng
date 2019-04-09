import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt

import Sigmoid


'''可视化数据'''
def plot_data(data):
    cols = data.shape[1]
    feature = data.iloc[:, 0:cols - 1]
    label = data.iloc[:, cols - 1]
    # iloc 根据列的位置索引来切片
    postive = feature[label == 1]
    negtive = feature[label == 0]

    # plt.figure(figsize=(8, 5))
    plt.scatter(postive.iloc[:, 0], postive.iloc[:, 1])
    plt.scatter(negtive.iloc[:, 0], negtive.iloc[:, 1], c='r', marker='x')
    plt.legend(['Admitted', 'Not admitted'], loc=1)
    plt.xlabel('Exam1 score')
    plt.ylabel('Exam2 score')
    # plt.show()


'''
cost function可以用矩阵实现也可以用ndarray实现，更建议使用后者
'''
def cost(theta, X, y):
    first = (-y) * np.log(Sigmoid.sigmoid(X @ theta))  # 这里*号是对应位置相乘而不是矩阵运算
    second = (1 - y) * np.log(1 - Sigmoid.sigmoid(X @ theta))
    return np.mean(first - second)


# 以下是用矩阵实现的代码
# cols = data.shape[1]
# X=np.matrix(data.iloc[:,0:cols-1].values)
# y=np.matrix(data.iloc[:,cols-1].values)
# theta=np.matrix(np.zeros((X.shape[1],1)))
# def cost(theta, X, y):
#     first = np.multiply(-y, np.log(Sigmoid.sigmoid(X @ theta)))  # 这里*号是对应位置相乘而不是矩阵运算
#     second = np.multiply(1 - y, np.log(1 - Sigmoid.sigmoid(X @ theta)))
#     return np.mean(first - second)

'''
梯度函数以及优化算法
'''
def gradient(theta, X, y):
    return (1 / len(X)) * (X.T @ (Sigmoid.sigmoid(X @ theta) - y))



'''
检测一下准确率
'''
def predict(theta, X):
    probability = Sigmoid.sigmoid(X @ theta)
    return [1 if x >= 0.5 else 0 for x in probability]
# 也可以用classification_report计算准确率
# from sklearn.metrics import classification_report
# print(classification_report(y, predictions))


# 决策边界
def plot_decision_boundary(theta, X):
    plot_x = np.linspace(20, 110)
    plot_y = -(theta[0] + plot_x * theta[1]) / theta[2]
    plt.plot(plot_x, plot_y, c='y')



def main():
    path = 'ex2data1.txt'
    data = pd.read_csv(path, names=('exam1', 'exam2', 'admitted'))
    data_copy = pd.read_csv(path, names=('exam1', 'exam2', 'admitted'))

    # 进一步准备数据，对结构初始化
    data.insert(0, 'One', 1)
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    theta = np.zeros(
        X.shape[1])  # 注意这里theta创建的是一维的数组，对于ndarray一定要注意一维时它的shape（和matrix有很大区别）


    # 这里不使用梯度下降法，换成其他优化算法来迭代
    result = opt.fmin_tnc(func=cost, x0=theta, fprime=gradient, args=(X, y))
    final_theta = result[0]

    predictions = predict(final_theta, X)
    correct = [1 if a == b else 0 for (a, b) in zip(predictions, y)]
    accurcy = np.sum(correct) / len(X)  # 准确率89%

    # 输入一个数据进行预测
    test = np.array([1, 45, 85]).reshape(1, 3)
    predict_result = predict(final_theta, test)  # 预测值y=1，概率为0.776


    '''
    打印输出图像
    '''
    # 这里的健壮性很差，如果调换plot_data()和plot_decision_boundary会出错
    plt.figure(figsize=(8, 5))
    plot_data(data_copy)
    plot_decision_boundary(final_theta, X)
    plt.show()

if __name__ == "__main__":
    main()