# import library
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split



# sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# sigmoid derivative function
def sigmoid_derivative(x):
    return x * (1-x)

def load_data():
    data_file_path = "/home/lap13385/Downloads/breast+cancer+wisconsin+diagnostic/wdbc.data"
    data = pd.read_csv(data_file_path, delimiter= ',')
    x = data.iloc[:,2:]
    y = data.iloc[:,1]
    x_train, x_temp, y_train, y_temp = train_test_split(x,y, test_size = 0.4, random_state = 42)
    x_val, x_test, y_val, y_test= train_test_split(x_temp,y_temp, test_size = 0.5, random_state = 42)
    return x_train, y_train, x_val, y_val, x_test, y_test

# Neural Network
class Neural_Network():
    def __init__(self, layers, alpha = 0.1):
        self.layers = layers
        self.alpha = alpha
        self.w = []
        self.b = []

        #Khoi tao cac tham so moi layer
        for i in range(0, len(layers) - 1):
            w_ = np.random.randn(layers[i], layers[i+1])
            b_ = np.zeros((layers[i+1],1))
            self.w.append(w_/layers[i])
            self.b.append(b_)

    def fit_partial(self,x,y):
        A = [x]

        out = A[-1]
        for i in range(0, len(self.layers) - 1):
            out = sigmoid(np.dot(out,self.w[i]) + self.b[i].T)
            A.append(out)

        y = y.reshape(-1,1)
        da = [-(y/A[-1] - (1-y)/(1-A[-1]))]
        dw = []
        db = []
        for i in reversed(range(0,len(self.layers)-1)):


def main():
    x_train, y_train, x_val, y_val, x_test, y_test = load_data()
    model = Neural_Network()

main()