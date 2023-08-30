import numpy as np


class Gradient_descent:
    def __init__(self, max_iter, learning_rate):
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.column_w1 = 4  # Ma trận W1 sẽ có số cột là column_w1
        self.w1 = None
        self.w2 = None

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    # sigmoid derivative function
    def sigmoid_derivative(self, x):
        return self.sigmoid(x) * (1-self.sigmoid(x))

    def predict(self, x):
        return self.sigmoid(self.sigmoid(x.dot(self.w1)).dot(self.w2))


    def fit(self, data_loader):
        n = data_loader.get_num_samples()
        np.random.seed(42)
        self.w1 = np.random.rand(data_loader.get_num_feature(),self.column_w1)
        self.w2 = np.random.rand(self.column_w1,1)
        for i in range(self.max_iter):
            x_batch, y_batch = next(data_loader)
            x_batch = np.array(x_batch)
            y_batch = np.array(y_batch)
            y_batch = y_batch.reshape(-1,1)
            y_batch_predicted = self.predict(x_batch)
            cost = -np.sum(np.multiply(y_batch, np.log(y_batch_predicted)) + np.multiply(1-y_batch, np.log(1-y_batch_predicted)))
            w1_derivative = 1/n * x_batch.T.dot((y_batch_predicted - y_batch).dot(self.w2.T) * self.sigmoid_derivative(x_batch.dot(self.w1)))
            w2_derivative = 1/n * self.sigmoid(x_batch.dot(self.w1)).T.dot(y_batch_predicted-y_batch)
            self.w1 -= self.learning_rate * w1_derivative
            self.w2 -= self.learning_rate * w2_derivative
        return self.w1, self.w2


