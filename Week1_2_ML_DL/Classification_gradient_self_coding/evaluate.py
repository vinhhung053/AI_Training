import numpy as np
from model import sigmoid

def binany_cross_entropy(y_predict,y):
    return -1/(y.shape[0]) * np.sum(y * np.log(y_predict) + (1-y)*np.log(1-y_predict))

def evaluate_model(x, y, w1, w2):
    y_predict = sigmoid(sigmoid(x.dot(w1)).dot(w2))
    return binany_cross_entropy(y_predict,y)
