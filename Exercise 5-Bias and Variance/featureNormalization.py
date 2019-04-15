import numpy as np
def feature_normalization(X):
    X_mean=np.mean(X,axis=0)
    X_std=np.std(X,axis=0,ddof=1)
    X_norm=(X-X_mean)/X_std

    return X_norm,X_std,X_mean