from keras.models import load_model
import pandas as pd
import matplotlib.pyplot as plt
from bazards import *

model = load_model('model.h5')
model.summary()

df = pd.read_csv('data_for_uncertainty_quantification.csv', sep=' ', header=None)
X = df[0]
f = df[1]
all_pred = np.zeros((N1+2*N2, K))
for i in range(K):
    all_pred[:, i] = list(model.predict(np.sort(X)))

nu = np.mean(all_pred, axis=1)
sigma = np.std(all_pred, axis=1)

fig = plt.figure(figsize=(14.0, 10.0))
ax = fig.add_subplot(1, 1, 1)
ax.plot(np.sort(X), nu, 'ro', markersize=6)
ax.plot(np.sort(X), nu + 2 * sigma, linestyle='--', linewidth=6, color='black')
ax.plot(np.sort(X), nu + sigma, linestyle='--', linewidth=6, color='black')
ax.plot(np.sort(X), nu - sigma, linestyle='--', linewidth=6, color='black')
ax.plot(np.sort(X), nu - 2 * sigma, linestyle='--', linewidth=6, color='black')
ax.plot(np.arange(-1, 3, 0.01), func(np.arange(-1, 3, 0.01)), linewidth=8)
plt.savefig('demonstration_uncertainty_quantification')


