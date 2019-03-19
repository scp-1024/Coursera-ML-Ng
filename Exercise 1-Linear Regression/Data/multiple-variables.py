import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

path = 'Data\ex1data2.txt'

data = pd.read_csv(path, header=None, names=['Size', 'Bedrooms', 'Price'])

# 样本的特征值之间相差比较大，会导致迭代时间变长
# 特征归一化
data = (data - data.mean()) / data.std()  # 除数可以用标准差也可以用max-min，因为pandas方便，所以使用标准差


data.insert(0, 'one', 1)
cols = data.shape[1]
X = data.iloc[:, 0:cols - 1]
y = data.iloc[:, cols - 1:cols]

X = np.matrix(X.values)
y = np.matrix(y.values)
theta = np.matrix([0, 0, 0])

def costFunction(X, y, theta):
    inner = np.power(((X * theta.T) - y),2)
    return np.sum(inner)/(2*len(X))

def gradientDescent(X,y,theta,alpha,epoch):
    temp=np.matrix(np.zeros(theta.shape))
    cost=np.zeros(epoch)
    for i in range(epoch):
        temp=theta-(alpha/len(X))*(X.dot(theta.T)-y).T.dot(X)
        theta=temp
        cost[i]=costFunction(X,y,theta)
    return theta,cost

alpha=0.01
epoch=1000
final_theta,cost=gradientDescent(X,y,theta,alpha,epoch)

plt.figure(figsize=(8,5))
plt.plot(np.arange(epoch),cost,'r')
plt.xlabel('iteration')
plt.ylabel('cost')

plt.show()
