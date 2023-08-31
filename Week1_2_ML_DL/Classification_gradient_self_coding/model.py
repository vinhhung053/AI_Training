import numpy as np


class Gradient_descent:
    def __init__(self, max_iter, learning_rate):
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.column_w1 = 4  # Ma trận W1 sẽ có số cột là column_w1
        self.w1 = None
        self.w2 = None

    def sigmoid(self, x):
        # print(1 / (1 + np.exp(-x)))
        return 1 / (1 + np.exp(-x))

    # sigmoid derivative function
    def sigmoid_derivative(self, x):
        return self.sigmoid(x) * (1-self.sigmoid(x))

    def predict(self, x):
        return self.sigmoid(self.sigmoid(x.dot(self.w1)).dot(self.w2))


    def fit(self, data_loader):
        np.random.seed(42)
        # self.w1 = np.random.rand(data_loader.get_num_feature(),self.column_w1)
        # self.w2 = (np.random.rand(self.column_w1,1)-0.5)*100

        self.w1 = np.random.uniform(low=-1, high=1, size=(data_loader.get_num_feature(),self.column_w1))
        self.w2 = np.random.uniform(low=-10, high=10, size=(self.column_w1,1))
        for i in range(self.max_iter):
            x_batch, y_batch = next(data_loader)
            x_batch = np.array(x_batch)
            y_batch = np.array(y_batch)
            y_batch = y_batch.reshape(-1,1)
            y_batch_predicted = self.predict(x_batch)
            n = x_batch.shape[0]
            w1_derivative = x_batch.T.dot((y_batch_predicted - y_batch).dot(self.w2.T) * self.sigmoid_derivative(x_batch.dot(self.w1)))
            w2_derivative = self.sigmoid(x_batch.dot(self.w1)).T.dot(y_batch_predicted-y_batch)
            self.w1 -= self.learning_rate * w1_derivative
            self.w2 -= self.learning_rate * w2_derivative
            print(self.sigmoid_derivative(x_batch.dot(self.w1)))
        return self.w1, self.w2


