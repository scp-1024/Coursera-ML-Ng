import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat


#==================================打印数据
def plot_data(X):
    fig, ax = plt.subplots(figsize=(8, 5))
    plt.scatter(
        X[:, 0], X[:, 1], facecolors='none', edgecolors='b',
        s=20)  # facecolors:填充颜色;edgecolors:边缘颜色;s:标记大小


#=================================压缩后子空间的基
def plot_reduce(means, U, S):
    plt.plot([means[0], means[0] + 1.5 * S[0] * U[0, 0]],
             [means[1], means[1] + 1.5 * S[0] * U[0, 1]],
             c='r',
             linewidth=3,
             label='First Principle Component')
    plt.plot([means[0], means[0] + 1.5 * S[1] * U[1, 0]],
             [means[1], means[1] + 1.5 * S[1] * U[1, 1]],
             c='g',
             linewidth=3,
             label='Second Principal Component')
    plt.axis('equal')
    plt.legend()


#==================================特征缩放
def feature_normalize(X):
    means = X.mean(axis=0)
    stds = X.std(axis=0, ddof=1)
    X_norm = (X - means) / stds

    return X_norm, means


#===================================PCA
def pca(X):
    sigma = (1 / len(X)) * (X.T @ X)  #求出协方差矩阵
    U, S, V = np.linalg.svd(sigma)  #奇异值分解

    return U, S, V


#===================================降维后的样本点
def project_data(X, U, K):
    Z = X @ U[:, 0:K]
    return Z


#===================================重建数据
def recover_data(Z, U, K):
    X_rec = Z @ U[:, 0:K].T

    return X_rec


#====================================PCA样本投影可视化
def visualize_data(X_norm, X_rec):
    fig, ax = plt.subplots(figsize=(8, 5))
    plt.axis('equal')
    plt.scatter(
        X_norm[:, 0],
        X_norm[:, 1],
        s=30,
        facecolors='none',
        edgecolors='blue',
        label='Original Data Points')

    plt.scatter(
        X_rec[:, 0],
        X_rec[:, 1],
        s=30,
        facecolors='none',
        edgecolors='red',
        label='PCA Reduced Data Points')

    plt.title("Example Dataset: Reduced Dimension Points Shown",fontsize=14)
    plt.xlabel('x1 [Feature Normalized]',fontsize=14)
    plt.ylabel('x2 [Feature Normalized]',fontsize=14)

    for x in range(X_norm.shape[0]):
        plt.plot([X_norm[x,0],X_rec[x,0]],[X_norm[x,1],X_rec[x,1]],'k--')
        plt.legend(loc=0)
        # 输入第一项全是X坐标，第二项都是Y坐标
    plt.show()


#====================================face图片显示
def plot_face(X,row,col):
    fig,ax=plt.subplots(row,col,figsize=(8,8))
    for i in range(row):
        for j in range(col):
            ax[i][j].imshow(X[i*col+j].reshape(32,32).T,cmap='Greys_r')
            ax[i][j].set_xticks([])
            ax[i][j].set_yticks([])
    plt.show()


def main():
    data = loadmat('ex7data1.mat')
    X = data['X']
    '''
    X_norm, means = feature_normalize(X)
    U, S, V = pca(X_norm)

    # plot_data(X)
    # plot_reduce(means,U,S)

    K = 1
    Z = project_data(X_norm, U, K)
    X_rec = recover_data(Z, U, K)
    visualize_data(X_norm, X_rec)
    '''
    data_2=loadmat('ex7faces.mat')
    X_2=data_2['X']
    X_2_norm,means_2=feature_normalize(X_2)
    U_2,S_2,V_2=pca(X_2_norm)

    Z_2=project_data(X_2_norm,U_2,K=36)
    X_2_rec=recover_data(Z_2,U_2,K=36)
    plot_face(X_2_rec,10,10)    
    

if __name__ == "__main__":
    main()