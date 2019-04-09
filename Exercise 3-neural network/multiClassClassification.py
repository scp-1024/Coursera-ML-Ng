import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.optimize import minimize

import Sigmoid


# 5000个训练样本，每个样本20*20的灰度值，展开为400列向量。输出为0~9的数字
def load_data(path):
    data = loadmat(path)
    X = data['X']
    y = data['y']
    return X, y


def plot_an_image(X, y):
    pick_one = np.random.randint(0, 5000)
    image = X[pick_one, :]
    fig, ax = plt.subplots(figsize=(1, 1))
    ax.matshow(image.reshape(20, 20), cmap='gray_r')
    plt.xticks([])
    plt.yticks([])
    plt.show()
    print('this should be {}'.format(y[pick_one]))


def cost_reg(theta, X, y, lmd):
    theta_reg = theta[1:]
    first = y * np.log(Sigmoid.sigmoid(
        X @ theta)) + (1 - y) * np.log(1 - Sigmoid.sigmoid(X @ theta))
    reg = (lmd / (2 * len(X))) * (theta_reg @ theta_reg)  # 惩罚项不从第一项开始
    return -np.mean(first) + reg


# 梯度项
def gradient_reg(theta, X, y, lmd):
    theta_reg = theta[1:]
    first = (1 / len(X)) * (X.T @ (Sigmoid.sigmoid(X @ theta) - y))
    reg = np.concatenate([np.array([0]), (lmd / len(X)) * theta_reg])
    return first + reg


# 训练样本
def one_vs_all(X, y, lmd, K):
    all_theta = np.zeros((K, X.shape[1]))
    for i in range(1, K + 1):
        theta = np.zeros(X.shape[1])

        y_i = np.array([1 if label == i else 0 for label in y])

        ret = minimize(
            fun=cost_reg,
            x0=theta,
            args=(X, y_i, lmd),
            method='TNC',
            jac=gradient_reg,
            options={'disp': True})

        all_theta[i - 1, :] = ret.x
    return all_theta


# 预测精确度
def predict_all(X, all_theta):
    h = Sigmoid.sigmoid(X @ all_theta.T)
    h_argmax = np.argmax(h, axis=1) # 返回指定方向上的最大值的索引 axis=0:按列索引，axis=1：按行索引
    h_argmax = h_argmax + 1
    return h_argmax


def main():
    X, y = load_data('ex3data1.mat')
    # plot_an_image(X,y)
    X = np.insert(X, 0, 1, axis=1)
    y = y.flatten()
    all_theta = one_vs_all(X, y, 1, 10)
    y_pred = predict_all(X, all_theta)
    accuracy = np.mean(y_pred == y)  # 精确度94.46%


if __name__ == "__main__":
    main()
