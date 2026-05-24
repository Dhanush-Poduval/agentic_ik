import pickle
import numpy as np
from main import DeltaNN, X_train, Y_train

nn = DeltaNN()

X_mean = np.mean(X_train, axis=0)
X_std = np.std(X_train, axis=0)

Y_mean = np.mean(Y_train, axis=0)
Y_std = np.std(Y_train, axis=0)

X_std[X_std == 0] = 1
Y_std[Y_std == 0] = 1

nn.X_mean = X_mean
nn.X_std = X_std
nn.Y_mean = Y_mean
nn.Y_std = Y_std

nn.train(X_train, Y_train, epochs=500)

with open("model.pkl", "wb") as f:
    pickle.dump(nn, f)

print("Model saved!")
