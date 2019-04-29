import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

#========================找聚类中心
def find_closet_centroids(X,centroids):
    '''
    返回每个样本所在的cluster索引(1,2,3)
    '''
    idx=[]
    max_dist=10000 # 给一个距离限定
    for i in range(len(X)):
        minus=X[i]-centroids # minus是3x2的矩阵，每一行代表了第i个样本到一个centroids的x1,x2距离
        dist=minus[:,0]**2+minus[:,1]**2 #求范式，即直线距离,dist是3x1的向量
        if dist.min()<max_dist:
            ci=np.argmin(dist) #返回沿axis的最小值索引
            idx.append(ci)
    return np.array(idx)

#=========================移动聚类中心
def compute_centroids(X,idx):
    centroids=[]
    for i in range(len(np.unique(idx))):
        u_k=X[idx==i].mean(axis=0) # 布尔索引，idx==i运算返回bool value，根据值为true的下标来输出X中对应元素值
        centroids.append(u_k)
    
    return np.array(centroids)

def plot_data(X,centroids,idx=None):

    colors = ['b','g','gold','darkorange','salmon','olivedrab', 
              'maroon', 'navy', 'sienna', 'tomato', 'lightgray', 'gainsboro'
             'coral', 'aliceblue', 'dimgray', 'mintcream', 'mintcream']

    assert len(centroids[0])<=len(colors),'colors not enough' # 这里centroids需要从ndarray转成list,但centroids[0]仍为ndarray.(那转个屁？)
    


def main():
    data=loadmat('ex7data2.mat')
    X=data['X']
    init_centroids=np.array([[3,3],[6,2],[8,5]])
    # idx=find_closet_centroids(X,init_centroids)
    # centroids=compute_centroids(X,idx)


if __name__ == "__main__":
    main()