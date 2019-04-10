import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import scipy.optimize as opt
from sklearn.metrics import classification_report
import Sigmoid as sg


def load_mat(path):
    data = loadmat(path)
    X = data['X']
    y = data['y']
    y = y.flatten()

    return X, y


def load_weight():
    weight = loadmat('ex4weights.mat')
    theta1 = weight['Theta1']
    theta2 = weight['Theta2']

    return theta1, theta2


def plot_data(X):
    '''随机画100个数字'''
    index = np.random.choice(range(5000),
                             100)  # np.random.choice(arrange,size),返回ndarray
    images = X[index]  # 随机选择100个样本
    fig, ax_array = plt.subplots(
        10, 10, sharex=True, sharey=True, figsize=(8, 8))  # ax_array为Axes对象
    for r in range(10):
        for c in range(10):
            ax_array[r, c].matshow(
                images[r * 10 + c].reshape(20, 20), cmap='gray_r'
            )  # matshow() 第一个参数为要显示的矩阵（Display an array as a matrix in a new figure window）
    plt.yticks([])
    plt.xticks([])
    plt.show()


def expand_y(y):
    result = []
    for i in y:
        y_array = np.zeros(10)
        y_array[i - 1] = 1
        result.append(y_array)
    return np.array(result)


def feed_forward(theta1, theta2, X):
    z2 = X @ theta1.T
    a2 = sg.sigmoid(z2)  #(5000,25)
    a2 = np.insert(a2, 0, 1, axis=1)  #(5000,26)
    z3 = a2 @ theta2.T
    a3 = sg.sigmoid(z3)

    return z2, a2, z3, a3


def cost(theta1, theta2, X, y):
    z2, a2, z3, h = feed_forward(theta1, theta2, X)
    # 这里的y是矩阵而不是向量了
    first = -y * np.log(h)
    second = (1 - y) * np.log(1 - h)

    return (np.sum(first - second)) / len(X)  # 这里不能用np.mean()，否则会相差10倍
    '''
    # or use loop
    for i in range(len(X)):
        first = - y[i] * np.log(h[i])
        second = (1 - y[i]) * np.log(1 - h[i])
        J = J + np.sum(first - second)
    J = J / len(X)
    return J
    '''


def cost_reg(theta1, theta2, X, y, lmd):
    c = cost(theta1, theta2, X, y)
    reg = (lmd / (2 * len(X))) * (
        np.sum(theta1[:, 1:]**2) + np.sum(theta2[:, 1:]**2))
    return reg + c


def main():
    path = 'ex4data1.mat'
    X, y = load_mat(path)
    X = np.insert(X, 0, 1, axis=1)
    y = expand_y(y)
    theta1, theta2 = load_weight()
    print(cost_reg(theta1, theta2, X, y, 1))  #0.38376985909092354
    

if __name__ == "__main__":
    main()
