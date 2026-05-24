import numpy as np
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
import requests

MATLAB_FK_URL = "http://127.0.0.1:8000/learn"

mat = loadmat('/home/dhanush/Downloads/fk_dataset.mat')
data = mat['data']

X = data[:, :6]   
Y = data[:, 6:] 

X_mean, X_std = X.mean(axis=0), X.std(axis=0)
Y_mean, Y_std = Y.mean(axis=0), Y.std(axis=0)

X_std[X_std == 0] = 1
Y_std[Y_std == 0] = 1

X_norm = (X - X_mean) / X_std
Y_norm = (Y - Y_mean) / Y_std

X_train, X_test, Y_train, Y_test = train_test_split(
    X_norm, Y_norm, test_size=0.2, random_state=42
)

class DeltaNN:
    def __init__(self, lr=0.0005):
        self.lr = lr

        self.W1 = np.random.randn(6, 256) * 0.01
        self.b1 = np.zeros((1, 256))

        self.W2 = np.random.randn(256, 256) * 0.01
        self.b2 = np.zeros((1, 256))

        self.W3 = np.random.randn(256, 128) * 0.01
        self.b3 = np.zeros((1, 128))

        self.W4 = np.random.randn(128, 6) * 0.01
        self.b4 = np.zeros((1, 6))

    def relu(self, x):
        return np.maximum(0, x)

    def relu_deriv(self, x):
        return (x > 0).astype(float)

    def forward(self, X):
        self.Z1 = X @ self.W1 + self.b1
        self.A1 = self.relu(self.Z1)

        self.Z2 = self.A1 @ self.W2 + self.b2
        self.A2 = self.relu(self.Z2)

        self.Z3 = self.A2 @ self.W3 + self.b3
        self.A3 = self.relu(self.Z3)

        self.Z4 = self.A3 @ self.W4 + self.b4
        return self.Z4

    def backward(self, X, Y, output):
        m = X.shape[0]

        dZ4 = (output - Y) / m
        dW4 = self.A3.T @ dZ4
        db4 = np.sum(dZ4, axis=0, keepdims=True)

        dA3 = dZ4 @ self.W4.T
        dZ3 = dA3 * self.relu_deriv(self.Z3)
        dW3 = self.A2.T @ dZ3
        db3 = np.sum(dZ3, axis=0, keepdims=True)

        dA2 = dZ3 @ self.W3.T
        dZ2 = dA2 * self.relu_deriv(self.Z2)
        dW2 = self.A1.T @ dZ2
        db2 = np.sum(dZ2, axis=0, keepdims=True)

        dA1 = dZ2 @ self.W2.T
        dZ1 = dA1 * self.relu_deriv(self.Z1)
        dW1 = X.T @ dZ1
        db1 = np.sum(dZ1, axis=0, keepdims=True)

        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W3 -= self.lr * dW3
        self.b3 -= self.lr * db3
        self.W4 -= self.lr * dW4
        self.b4 -= self.lr * db4

    def train(self, X, Y, epochs=600):
        for epoch in range(epochs):
            out = self.forward(X)
            loss = np.mean((out - Y) ** 2)
            self.backward(X, Y, out)

            if (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch+1}, Loss: {loss:.6f}")

class IKAgent:
    def __init__(self, nn):
        self.nn = nn
        self.prev_q = np.zeros(6)
        self.history = []

    def predict(self, goal):
        goal_norm = (goal - X_mean) / X_std
        pred = self.nn.forward(goal_norm.reshape(1, -1))
        pred = pred * Y_std + Y_mean
        return pred.flatten()

    def apply_constraints(self, q):
        return np.clip(q, -np.pi, np.pi)

    def is_singular(self, q):
        return np.linalg.norm(q[2:4]) < 0.05

    def call_matlab_fk(self, q):
        try:
            response = requests.post(
                MATLAB_FK_URL,
                json={"joints": q.tolist()},
                timeout=1.0
            )
            return np.array(response.json()["pos"])
        except:
            return None

    def utility(self, q, goal, actual_pos):
        if actual_pos is None:
            return -1e9

        error = np.linalg.norm(goal - actual_pos)
        smoothness = np.linalg.norm(q - self.prev_q)
        energy = np.sum(q ** 2)

        return -(1.0 * error + 0.3 * smoothness + 0.1 * energy)

    def learn(self, goal, q):
        goal_norm = (goal - X_mean) / X_std
        q_norm = (q - Y_mean) / Y_std

        pred = self.nn.forward(goal_norm.reshape(1, -1))
        self.nn.backward(goal_norm.reshape(1, -1), q_norm.reshape(1, -1), pred)

    def act(self, goal):

        q = self.predict(goal)
        q = self.apply_constraints(q)

        for _ in range(3):

            actual = self.call_matlab_fk(q)

            best_q = q
            best_score = self.utility(q, goal, actual)

            for _ in range(6):
                candidate = q + np.random.normal(0, 0.05, 6)
                candidate = self.apply_constraints(candidate)

                if self.is_singular(candidate):
                    continue

                actual_c = self.call_matlab_fk(candidate)

                score = self.utility(candidate, goal, actual_c)

                if score > best_score:
                    best_q = candidate
                    best_score = score

            q = best_q

        self.prev_q = q
        self.history.append(q)

        self.learn(goal, q)

        return q

if __name__ == "__main__":

    nn = DeltaNN()
    nn.train(X_train, Y_train)

    agent = IKAgent(nn)

    print("\nAgent Evaluation:\n")

    for i in range(20):

        goal = X_test[i] * X_std + X_mean
        true_q = Y_test[i] * Y_std + Y_mean

        pred_q = agent.act(goal)

        err = np.linalg.norm(pred_q - true_q)

        max_err = np.linalg.norm(np.full(6, 2 * np.pi))
        err = err / (max_err + 1e-8)
        err = np.clip(err, 0.0, 1.0)

        print(f"Sample {i} has error: {err:.4f}")
