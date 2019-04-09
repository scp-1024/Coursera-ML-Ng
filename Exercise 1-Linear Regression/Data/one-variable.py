import numpy as np
import pandas as pd

import matplotlib.pyplot as plt  #画图函数
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LogNorm


# 计算代价函数
def costFunciton(X, y, theta):
    inner = np.power(((X * theta.T) - y), 2)
    return np.sum(inner) / (2 * len(X))


# 批量梯度下降
def gradientDescent(X, y, theta, alpha, epoch):

    temp = np.matrix(np.zeros(theta.shape))

    # flatten(m)折叠数组至m维，default=1
    # parameters = int(theta.flatten().shape[1]) # 获取Theta的数量

    cost = np.zeros(epoch)  # epoch为迭代次数
    m = X.shape[0]  # 样本数量

    for i in range(epoch):
        temp = theta - (alpha / m) * (X.dot(theta.T) - y).T.dot(X)
        theta = temp
        # 记录一下每次更新后的误差
        cost[i] = costFunciton(X, y, theta)

    return theta, cost



'''
Normal Equation(正规方程法)
'''
def norEquation(X,y):
    theta=np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

    return theta


def main():
    #导入数据集
    path = 'Data\ex1data1.txt'

    # 变量data数据结构为DataFrame
    # read_csv()：将csv文件导入到DataFrame
    # csv文件简单来说就是：纯文本，以行为一条记录，每条记录被分隔符分隔为字段
    data = pd.read_csv(
        path, header=None,
        names=['Population',
            'Profit'])  # names为列名，如果不自己设置，就会默认从0列开始。如果header=None不设置，将不会有列名。

    # head(m)：获取m条记录，m未定义时默认为5条
    # data.head()

    # describe() 返回数据集的描述性信息：
    # count:样本个数  mean：均值  std:标准差  min:最小值  25%:四分之一位数  50%:中位数  75%:四分之三位数  max:最大值
    # data.describe()

    #dataframe也可以调用plot()用来画坐标图
    # data.plot(kind='scatter',x='Population',y='Profit',figsize=(8,5))
    # plt.show()


    '''
    从样本集中分离出X和y
    '''
    #插入第1列：X0=1
    data.insert(0, 'one', 1)

    cols = data.shape[1]  # .shape返回一个元组，[0]为行数，[1]为列数
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
    theta = np.matrix([0, 0])
    # 初始化学习速率和迭代次数
    alpha = 0.01
    epoch = 1000

    final_theta, cost = gradientDescent(X, y, theta, alpha, epoch)
    final_theta2 = norEquation(X,y)

    '''
    画图函数
    '''
    x = np.linspace(
        data.Population.min(), data.Population.max(),
        100)  # 横坐标:linspace(start,end, num)从start开始到end结束，平均分成num份，返回一个数组
    f = final_theta[0, 0] + (final_theta[0, 1] * x)  # 假设函数

    plt.figure(figsize=(8, 5))
    plt.plot(x, f, 'r', label='Prediction')
    plt.scatter(data['Population'], data.Profit, label='Training Data')
    plt.xlabel('Population')
    plt.ylabel('Profit')
    plt.title('Predicted Profit vs. Population Size')


    '''
    绘制代价函数与迭代次数的图像
    '''
    plt.figure(figsize=(8, 5))
    plt.plot(np.arange(epoch), cost, 'r')
    plt.xlabel('iteration')
    plt.ylabel('cost')


    '''
    绘制代价函数3D图像
    '''
    fig = plt.figure(figsize=(8, 5))
    ax = Axes3D(fig)

    # 绘制网格
    # X,Y value
    theta0 = np.linspace(-10, 10, 100)  # 网格theta0范围
    theta1 = np.linspace(-1, 4, 100)  # 网格theta1范围
    x1, y1 = np.meshgrid(theta0, theta1)  # 画网格
    # height value
    z = np.zeros(x1.shape)
    for i in range(0, theta0.size):
        for j in range(0, theta1.size):
            t = np.matrix([theta0[i], theta1[j]])
            z[i][j] = costFunciton(X, y, t)
    # 由循环可以看出，这里是先取x=-10时，y的所有取值，然后计算代价函数传入z的第一行
    # 因此在绘图过程中，需要把行和列转置过来
    z = z.T
    ax.set_xlabel(r'$\theta_0$')
    ax.set_ylabel(r'$\theta_1$')
    # 绘制函数图像
    ax.plot_surface(x1, y1, z, rstride=1, cstride=1)


    '''
    绘制等高线图
    '''
    plt.figure(figsize=(8, 5))
    lvls = np.logspace(-2, 3, 20)
    plt.contour(x1, y1, z, levels=lvls, norm=LogNorm())  # 画出等高线
    plt.plot(final_theta[0, 0], final_theta[0, 1], 'r', marker='x')  # 标出代价函数最小值点

    # 打印图像
    plt.show()

if __name__ == "__main__":
    main()