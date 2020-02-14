import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout
from bazards import *


###generate data
X = 2 * np.random.rand(N)  # X
f = func(X)
error = np.random.normal(scale=0.5, size=N)
obs = f + error  # Y


###show the data
fig = plt.figure(figsize=(14.0, 10.0))
ax = fig.add_subplot(1, 1, 1)
xx = np.asarray(range(200)) / 50 - 1.0
ax.plot(xx, func(xx), color='tab:orange', linewidth=6)
ax.plot(X, obs, 'bo', markersize=12)
plt.savefig('data_generation')

# split the data
X_train = X[ :int(0.8*N)]
y_train = obs[ :int(0.8*N)]
X_test = X[int(0.8*N): ]
y_test = obs[int(0.8*N): ]


# define the keras model
model = Sequential()
model.add(Dropout(p, input_shape=(1,)))
model.add(Dense(50, activation='relu'))
model.add(Dropout(p))
model.add(Dense(25, activation='relu'))
model.add(Dropout(p))
model.add(Dense(15, activation='tanh'))
model.add(Dropout(p))
model.add(Dense(1, activation='linear'))
model.compile(loss='mse', optimizer='adam', metrics=['mse'])

# fit the keras model on the dataset
history = model.fit(X_train, y_train, validation_split=0.25, epochs=200, batch_size=64, verbose=2)
# training history
fig2 = plt.figure(figsize=(14.0, 10.0))
ax2 = fig2.add_subplot(1,1,1)
ax2.plot(history.history['loss'], linewidth = 6)
ax2.plot(history.history['val_loss'], linewidth = 6)
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.savefig('loss_epoch')
model.save('model.h5')


# performance on test set
y_pred = model.predict(X_test)
fig1 = plt.figure(figsize=(14.0, 10.0))
ax1 = fig1.add_subplot(1, 1, 1)
ax1.plot(X_test, func(X_test), 'ro', markersize=12)
#ax1.plot(X_test, y_test, 'go', markersize=12)
ax1.plot(X_test, y_pred, 'bo', markersize=12)
plt.savefig('test_prediction')