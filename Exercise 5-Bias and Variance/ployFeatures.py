import numpy as np

def ploy_features(X, p):

    X_ploy = X.copy()
    for i in range(2, p + 1):
        X_ploy = np.c_[X_ploy, X[:, 1:]**i]
    return X_ploy
