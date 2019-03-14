import numpy as np
import pandas as pd 

import matplotlib.pyplot as plt #画图函数

#导入数据集
path = 'Data\ex1data1.txt'

# 变量data数据结构为DataFrame
# read_csv()：将csv文件导入到DataFrame
# csv文件简单来说就是：纯文本，以行为一条记录，每条记录被分隔符分隔为字段
data = pd.read_csv(path, header = None, names = ['Population','Profit']) # names为列名，如果不自己设置，就会默认从0列开始。如果header=None不设置，将不会有列名。


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


cols = data.shape[1] # .shape返回一个元组，[0]为行数，[2]为列数
# 提取X，y的值
X = data.iloc[:,0:cols-1]
y = data.iloc[:,cols-1:cols]
# IndexError: single positional indexer is out-of-bounds
# 输入的列号超过了索引值范围


# 将dataframe结构转化成np的matrix
# 当Theta取0时计算平均误差
X = np.matrix(X.values)
y = np.matrix(y.values)
theta = np.matrix([0,0])



'''
数据处理
'''
# 计算代价函数
def CostFunciton(X,y,theta):
    inner = np.power(((X * theta.T) - y),2)
    return np.sum(inner)/(2 * len(X))

# 批量梯度下降
def gradientDescent(X, y, theta, alpha, epoch):

    temp = np.matrix(np.zeros(theta.shape))
    
    # flatten(m)折叠数组至m维，default=1
    # parameters = int(theta.flatten().shape[1]) # 获取Theta的数量    
    
    cost = np.zeros(epoch) # epoch为迭代次数
    m = X.shape[0] # 样本数量

    for i in range(epoch):
        temp = theta - (alpha / m) * (X * theta.T - y).T * X
        theta = temp
        
        # 记录一下每次更新后的误差
        cost[i] = CostFunciton(X, y, theta)

    return theta,cost

# 初始化学习速率和迭代次数
alpha = 0.01
epoch = 1000

final_theta, cost = gradientDescent(X, y, theta, alpha, epoch)
# CostFunciton(X, y, final_theta)



'''
画图函数
'''
x = np.linspace(data.Population.min(), data.Population.max(), 100) # 横坐标
f = final_theta[0,0] + (final_theta[0,1] * x)# 假设函数

plt.figure(figsize=(8,5))
plt.plot(x, f, 'r', label='Prediction')
plt.scatter(data['Population'], data.Profit, label='Training Data')
plt.xlabel('Population')
plt.ylabel('Profit')
plt.title('Predicted Profit vs. Population Size')
plt.show()