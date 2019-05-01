import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from skimage import io


#========================找聚类中心
def find_closet_centroids(X,centroids):
    '''
    返回每个样本所在的cluster索引
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


#=========================画图
def plot_data(X,centroids,idx=None):

    colors = ['b','g','gold','darkorange','salmon','olivedrab', 
              'maroon', 'navy', 'sienna', 'tomato', 'lightgray', 'gainsboro'
             'coral', 'aliceblue', 'dimgray', 'mintcream', 'mintcream']

    assert len(centroids[0])<=len(colors),'colors not enough' # 这里centroids需要从ndarray转成list,但centroids[0]仍为ndarray.(那转个屁？)
    
    subX=[]
    if idx is not None:
        for i in range(centroids[0].shape[0]): #循环3次
            x_i=X[idx==i]
            subX.append(x_i) #把数据按cluster分开存储，subX是个list
    
    else:
        subX=[X]
    
    fig=plt.figure(figsize=(8,5))
    for i in range(len(subX)):
        xx=subX[i]
        plt.scatter(xx[:,0],xx[:,1],c=colors[i])    
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Plot of X Points')

    xx,yy=[],[]
    for centroid in centroids: 
        xx.append(centroid[:,0])
        yy.append(centroid[:,1])
    
    plt.plot(xx,yy,'rx--')

    plt.show()


#=============================聚类中心移动过程
def run_kmeans(X,centroids,max_iters):

    centroids_all=[]
    centroids_all.append(centroids)
    centroid_i=centroids
    for i in range(max_iters):
        idx=find_closet_centroids(X,centroid_i)
        centroid_i=compute_centroids(X,idx) 
        centroids_all.append(centroid_i) #每次移动后的聚类中心坐标都记录下来
    return idx,centroids_all


#==============================随机初始化聚类中心
def random_centroids(X,K):
    m=X.shape[0]
    index=np.random.choice(m,K)
    
    return X[index]


def main():
    '''
    data=loadmat('ex7data2.mat')
    X=data['X']

    for i in range(3):
        init_centroids=random_centroids(X,3)

    # init_centroids=np.array([[3,3],[6,2],[8,5]])
        idx,centroids_all=run_kmeans(X,init_centroids,20)
        plot_data(X,centroids_all,idx=idx) #idx为每个样本所在cluster的索引
    '''
    img=io.imread('bird_small.png')
    # plt.imshow(img)
    # plt.show()
    # print(img.shape)
    img=img/255
    X=img.reshape(-1,3) #转换成128x128行，3列为RGB三通道
    K=16 #16个聚类中心，就是把所有颜色压缩为16种RGB颜色，那么每个像素值需要4bit存储即可
    init_centroids=random_centroids(X,K)
    idx,centroids_all=run_kmeans(X,init_centroids,10)
    img_2=np.zeros(X.shape)
    centroids=centroids_all[-1]
    
    for i in range(len(centroids)):
        img_2[idx==i]=centroids[i]
    
    
    img_2=img_2.reshape((128,128,3))
    fig,axes=plt.subplots(1,2,figsize=(12,6))
    axes[0].imshow(img)
    axes[1].imshow(img_2)
    plt.show()
    

if __name__ == "__main__":
    main()