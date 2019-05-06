import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt

def plot_data(X):
    fig,ax=plt.subplots(figsize=(8,5))
    ax.scatter(X[:,0],X[:,1],marker='x')
    plt.show()

#=========================================高斯分布函数
def gaussion_distribution(X,mu,sigma2):
    m,n=X.shape
    # 这里这个if，如果sigma2是一维的，就把它存入对角矩阵中的对角线
    if np.ndim(sigma2)==1: #np.ndim:判断矩阵的维度
        sigma2=np.diag(sigma2) # np.diag(parameter):parameter为矩阵，则返回ndarray，存入对角线值；parameter为向量，这返回一个矩阵
 
    norm = 1./(np.power((2*np.pi), n/2)*np.sqrt(np.linalg.det(sigma2))) # np.linalg.det():返回行列式

    exp=np.zeros((m,1))
    for row in range(m):
        xrow=X[row]
        exp[row]=np.exp(-0.5*((xrow-mu).T).dot(np.linalg.inv(sigma2)).dot(xrow-mu))
    return norm*exp
#==========================================参数估计
def get_gaussian_params(X,multi):
    # mu=(1/len(X))*X.sum(axis=0)
    mu=np.mean(X,axis=0)
    if multi:
        sigma2=((X-mu)@(X-mu).T)/len(X)
    else:
        sigma2=X.var(axis=0,ddof=0) # ddof=0:意味着除以1/m
    return mu,sigma2

#==========================================可视化概率
def visual_contour(X,mu,sigma2):
    fig,ax=plt.subplots(figsize=(8,5))
    ax.scatter(X[:,0],X[:,1],marker='x')
    x=np.arange(0,30,.3)
    y=np.arange(0,30,.3)
    
    xx,yy=np.meshgrid(x,y)
    point=np.c_[xx.ravel(),yy.ravel()] # 按列合并，一列横坐标，一列纵坐标
    p_point=gaussion_distribution(point,mu,sigma2)
    p_point=p_point.reshape(xx.shape)
    cont_levels = [10**h for h in range(-20,0,3)]
    plt.contour(xx,yy,p_point,cont_levels)

#=========================================求阈值 varepsilon
def select_threshold(yval, pval):
    def computeF1(yval, pval):
        m = len(yval)
        tp = float(len([i for i in range(m) if pval[i] and yval[i]]))
        fp = float(len([i for i in range(m) if pval[i] and not yval[i]]))
        fn = float(len([i for i in range(m) if not pval[i] and yval[i]]))
        prec = tp/(tp+fp) if (tp+fp) else 0
        rec = tp/(tp+fn) if (tp+fn) else 0
        F1 = 2*prec*rec/(prec+rec) if (prec+rec) else 0
        return F1    
    

    epsilons = np.linspace(min(pval), max(pval), 1000)
    bestF1, bestEpsilon = 0, 0
    
    for e in epsilons: # 试验每个epsilon
        pval_ = pval < e # 将pval概率和当前阈值e比对。返回的pval_是预测是否为异常。其值是bool value
        thisF1 = computeF1(yval, pval_)# 计算F1
    
        # 一般来说 F1值越高说明模型质量越高，但过高可能导致过拟合   
        if thisF1 > bestF1:
            bestF1 = thisF1
            bestEpsilon = e

    return bestF1, bestEpsilon


def main():
    data=loadmat('ex8data1.mat')
    X,Xval,yval=data['X'],data['Xval'],data['yval']

    '''
    mu,sigma2=get_gaussian_params(X,False) 
    p=gaussion_distribution(X,mu,sigma2)
    pval=gaussion_distribution(Xval,mu,sigma2) # 计算验证集的概率
    bestF1,bestEpsilon=select_threshold(yval,pval)
    # 记录异常点
    x_anomaly=np.array([X[i] for i in range(len(p)) if p[i]<bestEpsilon])# 不是异常点的概率大于epsilon
    
    visual_contour(X,mu,sigma2)
    plt.scatter(x_anomaly[:,0],x_anomaly[:,1],s=100,facecolors='none',edgecolors='red')
    plt.show()
    '''

    '''
    高维数据:(feature:11)
    '''
    data2=loadmat('ex8data2.mat')
    X2,Xval2,yval2=data2['X'],data2['Xval'],data2['yval']
    
    mu2,sigma2_2=get_gaussian_params(X2,False)
    p2=gaussion_distribution(X2,mu2,sigma2_2)
    p2_val=gaussion_distribution(Xval2,mu2,sigma2_2)
    bestF1_2,bestEpsilon_2=select_threshold(yval2,p2_val)
    X2_anomaly=[X2[i] for i in range(X2.shape[0]) if p2[i]<bestEpsilon_2]
    print(bestEpsilon_2,len(X2_anomaly))

if __name__ == "__main__":
    main()