import numpy as np

# Activation functions
def relu(z):
    return np.maximum(0, z)

def relu_grad(z):
    return (z > 0).astype(float)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Binary cross entropy loss
def binary_cross_entropy(y, y_hat):
    eps = 1e-8 # safety buffer to avoid log(0)
    return -np.mean(
        y * np.log(y_hat + eps) +
        (1 - y) * np.log(1 - y_hat + eps)
    )

# Initialize parameters
def init_params(input_dim, hidden_dim):
    W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2 / input_dim) # He initialization (better for ReLU)
    b1 = np.zeros((1, hidden_dim))
    W2 = np.random.randn(hidden_dim, 1) * np.sqrt(2 / hidden_dim) # He initialization (better for ReLU)
    b2 = np.zeros((1, 1))
    return W1, b1, W2, b2

# X → Linear → ReLU → Linear → Sigmoid → y_hat
# Z1 = X @ W1 + b1 → A1 = ReLU(Z1)
# Z2 = A1 @ W2 + b2 → y_hat = Sigmoid(Z2)

def forward(X, W1, b1, W2, b2):
    Z1 = X @ W1 + b1
    A1 = relu(Z1)
    Z2 = A1 @ W2 + b2
    y_hat = sigmoid(Z2)
    return Z1, A1, Z2, y_hat

# X -> X*W1 + b1 = Z1 -> ReLU(Z1) = A1 -> A1*W2 + b2 = Z2 ->  Sigmoid(Z2) = y_hat

# Loss = -1/n * Σ [y*log(y_hat) + (1-y)*log(1-y_hat)]

# Gradient of dL/dZ2 = (y_hat - y) / n

# dL/ dW2 = dL/dZ2 * dZ2/dW2 = (y_hat - y) * A1.T / n
# dL/ db2 = dL/dZ2 * dZ2/db2 = (y_hat - y) / n

# dL/ dA1 = dL/dZ2 * dZ2/dA1 = (y_hat - y) * W2.T
# dL/ dZ1 = dL/ dA1 * dA1/dZ1 = (y_hat - y) * W2.T * relu_grad(Z1)
# dL/ dW1 = dL/dZ1 * dZ1/dW1 = (y_hat - y) * W2.T * relu_grad(Z1) * X.T / n
# dL/ db1 = dL/dZ1 * dZ1/db1 = (y_hat - y) * W2.T * relu_grad(Z1) / n


def backward(X, y, Z1, A1, y_hat, W2):
    n = X.shape[0]

    dZ2 = y_hat - y
    dW2 = (A1.T @ dZ2) / n
    db2 = np.mean(dZ2, axis=0, keepdims=True)

    dA1 = dZ2 @ W2.T
    dZ1 = dA1 * relu_grad(Z1)
    dW1 = (X.T @ dZ1) / n
    db1 = np.mean(dZ1, axis=0, keepdims=True)

    return dW1, db1, dW2, db2

# L2 regularization version
# Loss_L2 = Loss + lam * (||W1||^2 + ||W2||^2)

def backward_l2(X, y, Z1, A1, y_hat, W1, W2, lam):
    n = X.shape[0]

    dZ2 = y_hat - y
    dW2 = (A1.T @ dZ2) / n + 2 * lam * W2
    db2 = np.mean(dZ2, axis=0, keepdims=True)

    dA1 = dZ2 @ W2.T
    dZ1 = dA1 * relu_grad(Z1)
    dW1 = (X.T @ dZ1) / n + 2 * lam * W1
    db1 = np.mean(dZ1, axis=0, keepdims=True)

    return dW1, db1, dW2, db2


# Training loop
from Data.Scaling_Split_dataset import X_train_scaled, y_train, X_val_scaled, y_val
lr = 0.01
epochs = 500
lamda = 0.001

# He Initialize parameters with 16 hidden units
W1, b1, W2, b2 = init_params(X_train_scaled.shape[1], 16)

y_train_np = y_train.reshape(-1, 1) # reshape to 2D array

for epoch in range(epochs):
    Z1, A1, Z2, y_hat = forward(X_train_scaled, W1, b1, W2, b2)
    # loss = binary_cross_entropy(y_train_np, y_hat)

    # dW1, db1, dW2, db2 = backward(
    #     X_train_scaled, y_train_np, Z1, A1, y_hat, W2
    # )

    loss_l2 = binary_cross_entropy(y_train_np, y_hat) + lamda * (np.sum(W1**2) + np.sum(W2**2))
    dW1, db1, dW2, db2 = backward_l2(
        X_train_scaled, y_train_np, Z1, A1, y_hat, W1, W2, lamda
    )

    W1 -= lr * dW1
    b1 -= lr * db1
    W2 -= lr * dW2
    b2 -= lr * db2

    if epoch % 20 == 0:
        print(f"Epoch {epoch:03d} | Loss: {loss_l2:.4f}")


# Evaluate on validation set

_, _, _, y_val_hat = forward(X_val_scaled, W1, b1, W2, b2)
y_val_pred = (y_val_hat >= 0.5).astype(int)

from Model.Evaluation import evaluation
evaluation(y_val, y_val_pred)

# Save all parameters into one .npz file
np.savez('Model/mlp_params.npz', W1=W1, b1=b1, W2=W2, b2=b2)
print("MLP parameters saved!")