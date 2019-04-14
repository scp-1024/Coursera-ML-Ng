'''
用水位变化来预测水库出水量
X：水位变化
y：出水量
'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import scipy.optimize as opt


def plot_data(theta, X, y):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(X[:, 1:], y, 'x', color='r')

    x_f = np.linspace(-50, 40)
    f = theta[0] + theta[1:] * x_f
    ax.plot(x_f, f)

    plt.xlabel('Change in water level (x)')
    plt.ylabel('Water flowing out of the dam (y)')

    plt.show()


def cost_reg(theta, X, y, lmd):
    theta_reg = theta[1:]
    first = np.sum((X @ theta - y.flatten())**2)  # 这里的y是(12,1)需要转成(12,)
    second = lmd * np.sum(theta_reg**2)
    return (1 / (2 * len(X))) * (first + second)


def gradient_reg(theta, X, y, lmd):
    grad = ((X @ theta - y.flatten()) @ X) / len(X)  # xj项是@x
    grad[1:] += (lmd / len(X)) * theta[1:]  # 用+=计算正则化项
    return grad

def linear_train(theta,X,y):
    res = opt.minimize(
        fun=cost_reg, x0=theta, args=(X, y, 0), method='TNC',
        jac=gradient_reg)
    return res.x

def plot_learning_curve(theta,X,y,Xval,yval,lmd):
    error_train,error_cv=[],[]
    for i in range(1,X.shape[0]+1):
        XX=X[:i]
        yy=y[:i]
        theta_i=linear_train(theta,XX,yy)

        '''计算误差'''
        cost_train=cost_reg(theta_i,XX,yy,lmd) # 训练误差不包括正则项，lmd=0
        cv_train=cost_reg(theta_i,Xval,yval,lmd)
        error_train.append(cost_train)
        error_cv.append(cv_train)

    fig,ax=plt.subplots(figsize=(8,5))



def main():
    path = 'ex5data1.mat'
    data = loadmat(path)
    X, y = data['X'], data['y']
    Xval, yval = data['Xval'], data['yval']
    Xtest, ytest = data['Xtest'], data['ytest']

    X = np.insert(X, 0, 1, axis=1)
    Xval = np.insert(Xval, 0, 1, axis=1)
    Xtest = np.insert(Xtest, 0, 1, axis=1)
    
    # final_theta=linear_train(theta,X,y)
    # plot_data(final_theta, X, y)

    # 初始化theta
    theta = np.ones(X.shape[1])
    plot_learning_curve(theta,X,y,Xval,yval,0)


    


if __name__ == "__main__":
    main()