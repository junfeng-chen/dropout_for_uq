import numpy as np


N = 10000 #samples for train and test
N1 = 160 # interpolation samples for uncertainty quantification
N2 = 20 # extrapolation samples for uncertainty quantification
K = 1000 # repect K times when predicting

p = 0.1#dropout rate

##function used to generate data
def func(x):
    return np.sin(np.pi * x) + 3.73 * np.sin(np.e ** 2 * x)