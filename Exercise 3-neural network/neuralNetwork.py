from scipy.io import loadmat
import numpy as np
import Sigmoid
import multiClassClassification as mc


def load_weigth(path):
    data = loadmat(path)
    return data['Theta1'], data['Theta2']


def main():
    theta1, theta2 = load_weigth('ex3weights.mat')
    X, y = mc.load_data('ex3data1.mat')
    y = y.flatten()
    X = np.insert(X, 0, values=np.ones(X.shape[0]), axis=1)

    z2 = X @ theta1.T
    z2 = np.insert(z2, 0, 1, axis=1)
    a2 = Sigmoid.sigmoid(z2)
    
    z3 = a2 @ theta2.T
    a3 = Sigmoid.sigmoid(z3)
    
    y_pred = np.argmax(a3, axis=1) + 1
    accurcy = np.mean(y_pred == y)  # 精确度97.52%


if __name__ == "__main__":
    main()
