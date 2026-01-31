import numpy as np
import joblib
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from Data.Scaling_Split_dataset import X_test_scaled, y_test


def relu(z):
    return np.maximum(0, z)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def forward(X, W1, b1, W2, b2):
    Z1 = X @ W1 + b1
    A1 = relu(Z1)
    Z2 = A1 @ W2 + b2
    y_hat = sigmoid(Z2)
    return Z1, A1, Z2, y_hat

# 1. Load Logistic Regression
loaded_log_reg = joblib.load('Model/logistic_model.pkl')

# 2. Load MLP Parameters
data = np.load('Model/mlp_params.npz')
W1_test = data['W1']
b1_test = data['b1']
W2_test = data['W2']
b2_test = data['b2']

print("Models loaded successfully.")


# Evaluate Logistic Regression
print("\nEvaluating Logistic Regression on Test Set:")
y_log_reg_pred = loaded_log_reg.predict(X_test_scaled)

from Model.Evaluation import evaluation
evaluation(y_test, y_log_reg_pred)

# Evaluate MLP
print("\nEvaluating MLP on Test Set:")
_, _, _, y_mlp_hat = forward(X_test_scaled, W1_test, b1_test, W2_test, b2_test)
y_mlp_pred = (y_mlp_hat >= 0.5).astype(int)
evaluation(y_test, y_mlp_pred)