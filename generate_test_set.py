import numpy as np

np.random.seed(6560)
from bazards import *

data = np.random.rand(N1+2*N2)
X1 = 2 * data[:N1]  #data points in [0,2]
X2 = 2 + data[N1:N1+N2] #Extrapolation to right
X3 = data[N1+N2:] - 1 #Extrapolation to left

X = np.concatenate((X1, X2, X3))
y = func(X)
dataset = np.hstack([X.reshape((N1+2*N2, 1)),y.reshape((N1+2*N2, 1))])
np.savetxt('data_for_uncertainty_quantification.csv', dataset)