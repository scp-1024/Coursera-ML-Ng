import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn import svm

#==================================观察数据
def plot_data(X,y):
    positive=X[y==1]
    negative=X[y==0]

    fig,ax=plt.subplots(figsize=(8,5))    
    plt.scatter(positive[:,0],positive[:,1],marker='+',label='positive')
    plt.scatter(negative[:,0],negative[:,1],color='red',label='negative')

#=============================================可视化决策边界
def visualize_boundary(clf,X):
    x_min,x_max=X[:,0].min()*1.2,X[:,0].max()*1.1
    y_min,y_max=X[:,1].min()*1.2,X[:,1].max()*1.1
    xx,yy=np.meshgrid(np.linspace(x_min,x_max,500),np.linspace(y_min,y_max,500)) # 画网格点
    Z=clf.predict(np.c_[xx.ravel(),yy.ravel()]) # 用训练好的分类器对网格点进行预测

    Z=Z.reshape(xx.shape) # 转换成对应的网格点
    plt.contour(xx,yy,Z,level=[0],colors='black') # 等高线图，画出0/1分界线
    plt.show()

#=============================================高斯核函数
def gaussKernel(x1,x2,sigma):
    return np.exp(-((x1-x2)**2).sum()/(2*sigma**2))


def main():
    data=loadmat('ex6data1.mat')
    # print(data.keys()) 用于查看标签名称
    X=data['X']
    y=data['y'].flatten()

    c=1
    clf=svm.SVC(c,kernel='linear') # 参数 c,kernel 返回一个分类器对象
    clf.fit(X,y) # 用训练数据拟合分类器模型
    # plot_data(X,y)
    # visualize_boundary(clf,X)

    '''非线性拟合'''
    data2=loadmat('ex6data2.mat')
    X2=data2['X']
    y2=data2['y'].flatten()
    sigma=0.1
    gamma=np.power(sigma,-2)/2
    clf=svm.SVC(c,kernel='rbf',gamma=gamma) # 注意这里的参数gamma是整个分母，且要写成乘法形式
    clf.fit(X2,y2)
    plot_data(X2,y2)
    visualize_boundary(clf,X2)

    data3=loadmat('ex6data3.mat')
    X3=data3['X']
    y3=data3['y'].flatten()
    clf=svm.SVC(c,kernel='rbf',gamma=gamma)
    clf.fit(X3,y3)
    # plot_data(X3,y3)
    # visualize_boundary(clf,X3)
    

if __name__ == "__main__":
    main()