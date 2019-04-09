import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt

import Sigmoid


def plot_data(data):
    feature = data.iloc[:, 0:2]
    label = data.iloc[:, 2]
    positive = feature[label == 1]
    negative = feature[label == 0]

    plt.scatter(positive.iloc[:, 0].values, positive.iloc[:, 1].values)
    plt.scatter(
        negative.iloc[:, 0].values,
        negative.iloc[:, 1].values,
        c='r',
        marker='x')


'''
特征映射(feature mapping)
'''


def map_feature(x1, x2):
    degree = 6

    x1 = x1.reshape((x1.size, 1))  # ndarray.size：数组中元素的个数
    x2 = x2.reshape((x2.size, 1))
    result = np.ones(x1.shape[0])  # 初始化一个值为1的数组(列向量)

    for i in range(1, degree + 1):
        for j in range(0, i + 1):
            result = np.c_[result, (x1**(i - j) * (x2**j))]  # np.c_：列拼接
    return result  # 返回值即为特征值X


def cost_reg(theta,X, y, lmd):
    # 不惩罚第一项
    _theta = theta[1:]
    reg = (lmd / (2 * len(X))) * (_theta @ _theta)

    first = (y) * np.log(Sigmoid.sigmoid(X @ theta))
    second = (1 - y) * np.log(1 - Sigmoid.sigmoid(X @ theta))
    final = -np.mean(first + second)
    return final + reg


def gradient_reg(theta,X, y, lmd):
    # 因为不惩罚第一项，所以要分开计算
    grad = (1 / len(X)) * (X.T @ (Sigmoid.sigmoid(X @ theta) - y))
    grad[1:] += (lmd / len(X)) * theta[1:]
    return grad


# # 用sklearn来计算theta计算
# model=linear_model.LogisticRegression(penalty='l2',C=1.0)
# model.fit(X,y.ravel())
# model.score(X,y)

# 计算精确度
def predict(theta, X):
    probability = Sigmoid.sigmoid(X @ theta)
    return [1 if x >= 0.5 else 0 for x in probability]

# 决策边界
def plot_decision_boundary(theta):
    x=np.linspace(-1,1.5,50)
    plot_x,plot_y=np.meshgrid(x,x)

    z=map_feature(plot_x,plot_y)
    z=z@theta
    z=z.reshape(plot_x.shape)
    plt.contour(plot_x,plot_y,z,0,colors='yellow')    


def main():
    path = 'ex2data2.txt'
    data = pd.read_csv(
        path, names=('Microchip Test1', 'Microchip Test2', 'Accept'))
    x1 = data.iloc[:, 0].values
    x2 = data.iloc[:, 1].values
    
    X = map_feature(x1, x2)
    y = data['Accept'].values
    theta = np.zeros(X.shape[1])

    result = opt.fmin_tnc(
    func=cost_reg,
    x0=theta,
    fprime=gradient_reg,
    args=(X, y, 1),
    )


    final_theta=result[0]
    predictions=predict(final_theta,X)
    correct=[1 if a==b else 0 for (a,b) in zip(predictions,y)]
    accuracy=sum(correct)/len(correct) # 精确度83.05%
    print('模型预测精确度：{}%'.format(accuracy*100))

    plt.figure(figsize=(8, 5))
    plot_data(data)
    plot_decision_boundary(final_theta)
    plt.show()


if __name__ == "__main__":
    main()
