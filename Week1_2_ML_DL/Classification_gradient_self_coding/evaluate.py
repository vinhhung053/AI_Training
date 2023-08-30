import numpy as np


def binany_cross_entropy(y_predict, y):
    y = np.array(y).reshape(-1, 1)
    y_predict = np.array(y_predict)
    return -1 / (y.shape[0]) * np.sum(y * np.log(y_predict) + (1 - y) * np.log(1 - y_predict))


def evaluate_model(x, y, model):
    y_predict = model.predict(x)
    return binany_cross_entropy(y_predict, y)
