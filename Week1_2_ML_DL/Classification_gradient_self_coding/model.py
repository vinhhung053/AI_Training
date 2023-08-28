import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# sigmoid derivative function
def sigmoid_derivative(x):
    return x * (1-x)


def gradient_descent(x, y, max_iter, learning_rate):
    n = x.shape[0]
    current_w1 = np.random.rand(x.shape[1],4)
    current_w2 = np.random.rand(4,1)
    for i in range(max_iter):
        y_predicted = sigmoid(sigmoid(x.dot(current_w1)).dot(current_w2))
        # print((sigmoid_derivative(x.dot(current_w1))).shape)
        w1_derivative =1/n * (x.T).dot((y_predicted - y).dot(current_w2.T)*sigmoid_derivative(x.dot(current_w1)))
        w2_derivative =1/n * sigmoid(x.dot(current_w1)).T.dot(y_predicted-y)
        # print(w1_derivative)
        # print(w2_derivative)
        current_w1 -= learning_rate * w1_derivative
        current_w2 -= learning_rate * w2_derivative
        cost = -np.sum(np.multiply(y, np.log(y_predicted)) + np.multiply(1-y, np.log(1-y_predicted)))
        print(cost)
