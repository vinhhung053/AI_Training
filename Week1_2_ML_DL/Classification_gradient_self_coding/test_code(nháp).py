import numpy as np
import pandas as pd

# Load data
data_file_path = "/home/lap13385/Downloads/breast+cancer+wisconsin+diagnostic/wdbc.data"
data = pd.read_csv(data_file_path, header=None)
# Use 30 features (3-32 columns)
X = data.iloc[:, 2:].values
# Set Malignant to 1 and Benign to 0
y = (data.iloc[:, 1].values == 'M').astype(int)
y = 1 - y
# Normalize the data
X_min = np.min(X, axis=0)
X_max = np.max(X, axis=0)
X_norm = (X - X_min) / (X_max - X_min)
# Add a column of ones to X to account for the bias term
X = np.hstack((np.ones((X_norm.shape[0], 1)), X_norm))
# Initialize weights randomly
np.random.seed(42)
W1 = np.random.randn(X.shape[1])
W2 = np.random.randn()
# Define sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Define cross-entropy loss function
def cross_entropy_loss(y_true, y_pred):
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

# Define the model
def model(X, W1, W2):
    z1 = np.dot(X, W1)
    a1 = sigmoid(z1)
    z2 = a1
    y_pred = sigmoid(z2)
    return y_pred, a1

# Gradient descent parameters
learning_rate = 0.0001
num_iterations = 100
y_pred, a1 = model(X, W1, W2)

# Gradient descent loop
for i in range(num_iterations):
    # Forward pass
    y_pred, a1 = model(X, W1, W2)
    # print(y_pred)
    # Compute loss
    loss = cross_entropy_loss(y, y_pred)
    # Compute gradients
    dL_dy = - (y / y_pred - (1 - y) / (1 - y_pred))
    dy_dz2 = y_pred * (1 - y_pred)
    dz2_da1 = W2
    da1_dz1 = a1 * (1 - a1)
    dz1_dW1 = X
    dz2_dW2 = a1

    dL_dz2 = dL_dy * dy_dz2
    dL_da1 = dL_dz2 * dz2_da1
    dL_dz1 = dL_da1 * da1_dz1

    dL_dW1 = np.dot(dz1_dW1.T, dL_dz1)
    dL_dW2 = np.dot(dz2_dW2.T, dL_dz2)

    # Update weights
    W1 -= learning_rate * dL_dW1
    W2 -= learning_rate * dL_dW2
    # Print loss every 100 iterations
    if i % 10 == 0:
        print(f'Iteration {i}, Loss: {loss}')

print('Training finished.')