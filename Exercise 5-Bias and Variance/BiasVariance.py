'''
用水位变化来预测水库出水量
X：水位变化
y：出水量
'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import scipy.optimize as opt
import ployFeatures as pf
import featureNormalization as fn


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

def linear_train(X,y,lmd):
    theta = np.zeros(X.shape[1])
    res = opt.minimize(
        fun=cost_reg, x0=theta, args=(X, y, lmd), method='TNC',
        jac=gradient_reg)
    return res.x

def plot_learning_curve(theta,X,y,Xval,yval,lmd):
    error_train,error_cv=[],[]
    for i in range(1,X.shape[0]+1):
        XX=X[:i]
        yy=y[:i]
        theta_i=linear_train(XX,yy,lmd)

        '''计算误差'''
        cost_train=cost_reg(theta_i,XX,yy,lmd) # 训练误差不包括正则项，lmd=0
        cv_train=cost_reg(theta_i,Xval,yval,lmd)
        error_train.append(cost_train)
        error_cv.append(cv_train)

    fig,ax=plt.subplots(figsize=(8,5))
    ax.plot(range(1,len(X)+1),error_train,label="Train")
    ax.plot(range(1,len(X)+1),error_cv,label="Cross Validation",color="green")
    ax.legend()
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    plt.xlabel("Number of training examples")
    plt.ylabel('error')
    plt.title('Learning curve of linear regression')
    plt.show()


def poly_fit(X,Xval,Xtest,y,yval):
    # 增加多项式特征
    p=8
    lmd=1
    X_ploy=pf.ploy_features(X,p)
    Xval_ploy=pf.ploy_features(Xval,p)
    Xtest_ploy=pf.ploy_features(Xtest,p)

    X_norm,X_std,X_mean=fn.feature_normalization(X_ploy)
    Xval_norm,Xval_std,Xval_mean=fn.feature_normalization(Xval_ploy)


    theta_poly_final=linear_train(X_norm,y,lmd)

    return X_norm,Xval_norm
    # plot_learning_curve(theta_poly_final,X_norm,y,Xval_norm,yval,1)

def plot_poly_fit(theta,X,y,p):
    # 画图函数
    # 画点
    fig,ax=plt.subplots(figsize=(8,5))
    ax.plot(X[:,1:],y,'x',color='r')

    # 画拟合
    # 这里需要把X_plot重新构造
    x=np.linspace(-75,55,50)
    xmat=x.reshape(-1,1)
    xmat=np.insert(xmat,0,1,axis=1)
    xmat=pf.ploy_features(xmat,p)
    xmat_norm,xmat_std,xmat_mean=fn.feature_normalization(xmat)
    ax.plot(x,xmat_norm@theta,'b--')
    plt.show()

def validation_curve(X,y,Xval,yval):
    # 设置一组lmd值
    lambdas=[0.,0.001,0.003,0.01,0.03,0.1,0.3,1.,3.,10.]
    # 对每组lmd值进行训练
    error_cv,error_train=[],[]
    for l in lambdas:
        theta_i=linear_train(X,y,l)
        error_train.append(cost_reg(theta_i,X,y,0)) # 计算误差时lmd=0
        error_cv.append(cost_reg(theta_i,Xval,yval,0))
    
    fig,ax=plt.subplots(figsize=(8,5))
    ax.plot(lambdas,error_train,label='Train')
    ax.plot(lambdas,error_cv,label='Cross Validation')
    plt.xlabel('Lambda')
    plt.ylabel('Error')
    plt.legend()
    plt.show()
    return error_train,error_cv
    


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
    # theta = np.ones(X.shape[1])
    # plot_learning_curve(theta,X,y,Xval,yval,0)

    # plot_poly_fit(X,Xval,Xtest,y,yval)
    X_norm,Xval_norm=poly_fit(X,Xval,Xtest,y,yval)
    error_train,error_cv=validation_curve(X_norm,y,Xval_norm,yval)

    print(error_train,error_cv)
    


if __name__ == "__main__":
    main()