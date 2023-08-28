# import library
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def get_args():
    parser = argparse.ArgumentParser(description='test Week_1_2 anh Chien')

    parser.add_argument('--max_iter',
                        type=int,
                        default= 1,
                        help='max_iter')

    parser.add_argument('--learning_rate',
                        type= float,
                        default= 0.000001,
                        help='The learning rate')

    args = parser.parse_args()
    return args

# sigmoid functional
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
    y_mapping = {"M": 1, "B": 0}
    y = [y_mapping[val] for val in y]
    x_train, x_temp, y_train, y_temp = train_test_split(x,y, test_size = 0.4, random_state = 42)
    x_val, x_test, y_val, y_test= train_test_split(x_temp,y_temp, test_size = 0.5, random_state = 42)
    return x_train, y_train, x_val, y_val, x_test, y_test

# Gradient_descent
def gradient_descent(x, y, max_iter, learning_rate):
    n = x.shape[0]
    current_w1 = np.ones((x.shape[1],4))
    current_w2 = np.ones((4,1))
    for i in range(max_iter):
        y_predicted = sigmoid(sigmoid(x.dot(current_w1)).dot(current_w2))
        print(y_predicted)



def main():
    args = get_args()
    x_train, y_train, x_val, y_val, x_test, y_test = load_data()
    model = gradient_descent(x_train, y_train, max_iter = args.max_iter, learning_rate = args.learning_rate)


main()