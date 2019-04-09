import numpy as np


# import matplotlib.pyplot as plt
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# Sigmoid函数测试代码
# x1=np.arange(-10,10,1)
# plt.plot(x1,sigmoid(x1),'r')
# plt.show()