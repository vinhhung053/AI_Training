import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# sigmoid derivative function
def sigmoid_derivative(x):
    return x * (1-x)

def gradient_descent(x, y, max_iter, learning_rate):
    n = x.shape[0]
    current_w1 = np.ones((x.shape[1],4))
    current_w2 = np.ones((4,1))
    for i in range(max_iter):
        y_predicted = sigmoid(sigmoid(x.dot(current_w1)).dot(current_w2))
        print(y_predicted)
