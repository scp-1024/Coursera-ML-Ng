import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt

def plot_data(X):
    fig,ax=plt.subplots(figsize=(8,5))
    ax.scatter(X[:,0],X[:,1],marker='x')
    plt.show()

#=========================================高斯分布函数
def gaussion_distribution(X,mu,sigma):
    m,n=X.shape
    # 这里这个if，如果sigma是一维的，就把它存入对角矩阵中的对角线
    if np.ndim(sigma)==1: #np.ndim:判断矩阵的维度
        sigma=np.diag(sigma) # np.diag(parameter):parameter为矩阵，则返回ndarray，存入对角线值；parameter为向量，这返回一个矩阵
 
    norm = 1./(np.power((2*np.pi), n/2)*np.sqrt(np.linalg.det(sigma))) # np.linalg.det():返回行列式

    exp=np.zeros((m,1))
    for row in range(m):
        xrow=X[row]
        exp[row]=np.exp(-0.5*((xrow-mu).T).dot(np.linalg.inv(sigma)).dot(xrow-mu))
    return norm*exp
#==========================================参数估计

def main():
    data=loadmat('ex8data1.mat')
    X,Xval,yval=data['X'],data['Xval'],data['yval']


if __name__ == "__main__":
    main()