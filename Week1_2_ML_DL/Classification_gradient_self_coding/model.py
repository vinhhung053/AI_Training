import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# sigmoid derivative function
def sigmoid_derivative(x):
    return sigmoid(x) * (1-sigmoid(x))


def gradient_descent(data_loader, max_iter, learning_rate):
    n = data_loader.get_num_samples()
    np.random.seed(42)
    current_w1 = np.random.rand(data_loader.get_num_feature(),4)
    current_w2 = np.random.rand(4,1)
    res = []
    for i in range(max_iter):
        x, y = next(data_loader)
        x = np.array(x)
        y = np.array(y)
        y = y.reshape(-1,1)
        y_predicted = sigmoid(sigmoid(x.dot(current_w1)).dot(current_w2))
        cost = -np.sum(np.multiply(y, np.log(y_predicted)) + np.multiply(1-y, np.log(1-y_predicted)))
        # print((sigmoid_derivative(x.dot(current_w1))).shape)
        w1_derivative = 1/n * (x.T).dot((y_predicted - y).dot(current_w2.T)*sigmoid_derivative(x.dot(current_w1)))
        w2_derivative = 1/n * sigmoid(x.dot(current_w1)).T.dot(y_predicted-y)
        current_w1 -= learning_rate * w1_derivative
        current_w2 -= learning_rate * w2_derivative
        res.append(cost)
    return current_w1, current_w2


