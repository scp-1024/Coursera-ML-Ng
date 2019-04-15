import numpy as np
def ploy_features(X,p):

    X_ploy=X[:]
    for i in range(2,p+1):
        X_ploy=np.c_[X**i]
    return X_ploy