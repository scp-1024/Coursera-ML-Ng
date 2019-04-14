import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import scipy.optimize as opt
from sklearn.metrics import classification_report
import Sigmoid as sg
import feedForward as ff
import copy


def random_init(size):
    return np.random.uniform(-0.12, 0.12, size)


def gradient(theta1, theta2, X, y):
    z2, a2, z3, h = ff.feed_forward(theta1, theta2, X)
    d3 = h - y  # (5000,10)
    d2 = d3 @ theta2[:, 1:] * sg.sigmoid_gradient(z2)  # (5000,25)
    D2 = d3.T @ a2  # (10,26)
    D1 = d2.T @ X  # (25,401)
    D = (1 / len(X)) * serialize(D1, D2)  #(10285,)

    return D


def serialize(a, b):
    '''展开参数'''
    return np.r_[a.flatten(), b.flatten()]  # 按行拼接


def deserialize(seq):
    '''提取参数'''
    return seq[:25 * 401].reshape(25, 401), seq[25 * 401:].reshape(10, 26)


def cost_reg(theta, X, y, lmd):
    theta1, theta2 = deserialize(theta)
    c = ff.cost(theta1, theta2, X, y)
    reg = (lmd / (2 * len(X))) * (
        np.sum(theta1[:, 1:]**2) + np.sum(theta2[:, 1:]**2))
    return reg + c



def gradient_check(theta1,theta2,X,y,e):
    theta_temp=serialize(theta1,theta2) # (10285,)
    numeric_grad=[]
    for i in range(len(theta_temp)):
        plus=copy.copy(theta_temp)
        minus=copy.copy(theta_temp)
        plus[i]+=e
        minus[i]-=e
        grad_i=(cost_reg(plus,X,y,1)-cost_reg(minus,X,y,1))/(e*2)
        numeric_grad.append(grad_i)

    numeric_grad=np.array(numeric_grad) #近似梯度矩阵 (10285,)  数值梯度

    reg_D1,reg_D2=regularized_gradient(theta_temp,X,y)
    analytic_grad=serialize(reg_D1,reg_D2) # 解析梯度
    
    diff = np.linalg.norm(numeric_grad - analytic_grad) / np.linalg.norm(numeric_grad + analytic_grad) # 求范数

    print('If your backpropagation implementation is correct,\nthe relative difference will be smaller than 10e-9 (assume epsilon=0.0001).\nRelative Difference: {}\n'.format(diff))



def regularized_gradient(theta, X, y, lmd=1):
    theta1, theta2 = deserialize(theta)
    D1, D2 = deserialize(gradient(theta1, theta2, X, y))
    theta1[:, 0] = 0
    theta2[:, 0] = 0
    reg_D1 = D1 + (lmd / len(X)) * theta1
    reg_D2 = D2 + (lmd / len(X)) * theta2
    # print(serialize(reg_D1,reg_D2))
    return serialize(reg_D1, reg_D2)


def nn_training(X, y):

    init_theta = random_init(10285)
    res = opt.minimize(
        fun=cost_reg,
        x0=init_theta,
        args=(X, y, 1),
        method='TNC',
        jac=regularized_gradient,
        options={'maxiter':400}
        )
    return res


def accuracy(theta, X, y):
    theta1, theta2 = deserialize(theta)
    _, _, _, h = ff.feed_forward(theta1, theta2, X)
    y_pred = np.argmax(h, axis=1) + 1

    print(classification_report(y, y_pred))  


'''可视化隐藏层'''
def plot_hidden(theta):
    t1,_=deserialize(theta)
    t1=t1[:,1:]
    fig,ax_array=plt.subplots(5,5,sharex=True,sharey=True,figsize=(6,6))
    for r in range(5):
        for c in range(5):
            ax_array[r,c].matshow(t1[r*5+c].reshape(20,20),cmap='gray_r')
    plt.xticks([])
    plt.yticks([])
    plt.show()


def main():
    path = 'ex4data1.mat'
    X, raw_y = ff.load_mat(path)
    X = np.insert(X, 0, 1, axis=1)
    y = ff.expand_y(raw_y)  # y的一行表示一个样本
    # theta1, theta2 = ff.load_weight()
    # gradient_check(theta1,theta2,X,y,0.0001)
    theta_unroll = nn_training(X, y)
    accuracy(theta_unroll.x, X, raw_y)


if __name__ == "__main__":
    main()
