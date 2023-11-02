import torch
import torch.nn as nn

class GradientDescent(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.column_w1 = 4  # Ma trận W1 sẽ có số cột là column_w1
        self.w1 = torch.FloatTensor(num_features, self.column_w1).uniform_(-1, 1)
        self.w2 = torch.FloatTensor(self.column_w1, 1).uniform_(-1, 1)

    def sigmoid(self, x):
        # print(1 / (1 + np.exp(-x)))
        return 1 / (1 + torch.exp(-x))

    # sigmoid derivative function
    def sigmoid_derivative(self, x):
        return self.sigmoid(x) * (1-self.sigmoid(x))


    def forward(self, x):
        return self.sigmoid(self.sigmoid(x.mm(self.w1)).mm(self.w2))


    @torch.jit.ignore
    def fit(self, data_loader, max_iter, learning_rate):
        # np.random.seed(42)
        # print(type(data_loader))

        for i in range(max_iter):
            x_batch, y_batch = next(data_loader)
            x_batch = torch.FloatTensor(x_batch.values)
            y_batch = torch.FloatTensor(y_batch.values).view(-1, 1)
            y_batch_predicted = self.forward(x_batch)
            n = x_batch.shape[0]
            w1_derivative = x_batch.T.mm((y_batch_predicted - y_batch).mm(self.w2.T) * self.sigmoid_derivative(x_batch.mm(self.w1)))
            w2_derivative = self.sigmoid(x_batch.mm(self.w1)).T.mm(y_batch_predicted-y_batch)
            self.w1 -= learning_rate * w1_derivative
            self.w2 -= learning_rate * w2_derivative
        return self.w1, self.w2


