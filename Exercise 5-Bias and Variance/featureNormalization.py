import numpy as np
def feature_normalization(X):

    X_norm=X.copy()
    X_mean=np.mean(X,axis=0)
    X_std=np.std(X,axis=0,ddof=1) # ddof=1 标准样本差,ddof=0则为总体样本差
    X_norm[:,1:]=(X_norm[:,1:]-X_mean[1:])/X_std[1:]
    return X_norm,X_std,X_mean