import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import scipy.optimize as opt

#======================================可视化评分矩阵
def plot_mat(Y):
    nm,nu=Y.shape
    fig = plt.figure(figsize=(8,8*(1682./943.)))
    plt.imshow(Y, cmap='rainbow')
    plt.colorbar()
    plt.ylabel('Movies (%d)'%nm,fontsize=20)
    plt.xlabel('Users (%d)'%nu,fontsize=20)
    plt.show()

'''为了使用优化算法，忍痛将X和theta降为一维'''
#========================================一维化
def serialize(X,Theta):
    return np.r_[X.flatten(),Theta.flatten()]

#========================================维度展开
def deserialize(seq,nm,nu,nf):
    return seq[:nm*nf].reshape(nm,nf),seq[nm*nf:].reshape(nu,nf)

#========================================代价函数
def cf_cost_fun(param,Y,R,nm,nu,nf,lmd):
    X,theta=deserialize(param,nm,nu,nf)
    cost=(1/2)*np.sum(((X@theta.T-Y)*R)**2)
    reg1=(lmd/2)*np.sum(theta**2)
    reg2=(lmd/2)*np.sum(X**2)
    return cost+reg1+reg2

#========================================梯度下降
def cf_gradient(param,Y,R,nm,nu,nf,lmd):

    X,theta=deserialize(param,nm,nu,nf)
    X_grad=((X@theta.T-Y)*R)@theta+lmd*X
    theta_grad=((X@theta.T-Y)*R).T@X+lmd*theta

    return serialize(X_grad,theta_grad)


#========================================梯度检测
def checkGradient(params, Y, myR, nm, nu, nf, lmd = 0.):
    """
    Let's check my gradient computation real quick
    """
    print('Numerical Gradient \t cofiGrad \t\t Difference')
    
    # 分析出来的梯度
    grad = cf_gradient(params,Y,myR,nm,nu,nf,lmd)
    
    # 用 微小的e 来计算数值梯度。
    e = 0.0001
    nparams = len(params)
    e_vec = np.zeros(nparams)

    # Choose 10 random elements of param vector and compute the numerical gradient
    # 每次只能改变e_vec中的一个值，并在计算完数值梯度后要还原。
    for i in range(10):
        idx = np.random.randint(0,nparams)
        e_vec[idx] = e
        loss1 = cf_cost_fun(params-e_vec,Y,myR,nm,nu,nf,lmd)
        loss2 = cf_cost_fun(params+e_vec,Y,myR,nm,nu,nf,lmd)
        numgrad = (loss2 - loss1) / (2*e)
        e_vec[idx] = 0
        diff = np.linalg.norm(numgrad - grad[idx]) / np.linalg.norm(numgrad + grad[idx])
        print('%0.15f \t %0.15f \t %0.15f' %(numgrad, grad[idx], diff))

#========================================均值化
def normalizeRatings(Y, R):
    """
    The mean is only counting movies that were rated
    """
    Ymean = (Y.sum(axis=1) / R.sum(axis=1)).reshape(-1,1)
#     Ynorm = (Y - Ymean)*R  # 这里也要注意不要归一化未评分的数据
    Ynorm = (Y - Ymean)*R  # 这里也要注意不要归一化未评分的数据
    return Ynorm, Ymean

def main():
    data=loadmat('ex8_movies.mat')
    Y,R=data['Y'],data['R'] #矩阵Y(电影数x用户数):用户对电影评分(1~5),为0则是还未评分;矩阵R:如果用户有对电影评分R(i,j)=1,否则等于0;   
    # plot_mat(Y)

    '''
    mat=loadmat('ex8_movieParams.mat')
    X=mat['X'] #(1682,10)
    Theta=mat['Theta'] #(943,10)
    nu=int(mat['num_users']) #943
    nm=int(mat['num_movies']) #1682
    nf=int(mat['num_features']) #10
    nu=4;nm=5;nf=3
    X=X[:nm,:nf]
    Theta=Theta[:nu,:nf]
    Y=Y[:nm,:nu]
    R=R[:nm,:nu]
    # print(checkGradient(serialize(X,Theta),Y,R,nm,nu,nf,lmd=0))
    '''

    '''电影名字和上映年份'''
    movies=[]
    with open('movie_ids.txt','r',encoding='utf-8') as f:
        for line in f:
            movies.append(' '.join(line.strip().split(' ')[1:]))
    
    '''对看过的电影评分'''
    my_ratings = np.zeros((1682,1))
    my_ratings[0]   = 4
    my_ratings[97]  = 2
    my_ratings[6]   = 3
    my_ratings[11]  = 5
    my_ratings[53]  = 4
    my_ratings[63]  = 5
    my_ratings[65]  = 3
    my_ratings[68]  = 5
    my_ratings[182] = 4
    my_ratings[225] = 5
    my_ratings[354] = 5


    '''将添加的用户数据合并到Y和R'''
    Y=np.c_[Y,my_ratings]
    R=np.c_[R,my_ratings!=0]
    nm,nu=Y.shape
    nf=10
    Ynorm, Ymean = normalizeRatings(Y, R)
    X = np.random.random((nm, nf))
    Theta = np.random.random((nu, nf))
    params = serialize(X, Theta)
    lmd = 10

    res = opt.minimize(fun=cf_cost_fun,
                   x0=params,
                   args=(Y, R, nm, nu, nf, lmd),
                   method='TNC',
                   jac=cf_gradient,
                   options={'maxiter': 100})
    ret=res.x

    fit_X, fit_Theta = deserialize(ret, nm, nu, nf)
    
    '''用户预测'''
    # 所有用户的剧场分数矩阵
    pred_mat = fit_X @ fit_Theta.T
    # 最后一个用户的预测分数， 也就是我们刚才添加的用户
    pred = pred_mat[:,-1] + Ymean.flatten()
    pred_sorted_idx = np.argsort(pred)[::-1] # 排序并翻转，使之从大到小排列
    print("Top recommendations for you:")
    for i in range(10):
        print('Predicting rating %0.1f for movie %s.' \
            %(pred[pred_sorted_idx[i]],movies[pred_sorted_idx[i]]))

    print("\nOriginal ratings provided:")
    for i in range(len(my_ratings)):
        if my_ratings[i] > 0:
            print('Rated %d for movie %s.'% (my_ratings[i],movies[i]))
    


if __name__ == "__main__":
    main()