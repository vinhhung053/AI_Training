import numpy as np


def binary_cross_entropy(y_predict, y):

    y = np.array(y).reshape(-1, 1)
    y_predict = np.array(y_predict)
    score = -1 / (y.shape[0]) * np.sum(y * np.log(y_predict) + (1 - y) * np.log(1 - y_predict))
    print("  Cross entropy: ", score)

def positive_negative(y_predict, y):
    y = np.array(y).reshape(-1, 1)
    y_predict = np.array(y_predict)
    true_pos = 0
    true_neg = 0
    for i in range(y_predict.shape[0]):
        y_predict[i][0] = (y_predict[i][0] >= 0.5)
        if(y_predict[i][0] + y[i][0] == 2):
            true_pos = true_pos + 1
        elif (y_predict[i][0] + y[i][0] == 0):
            true_neg = true_neg + 1
    print("  true_pos / number sample: ", true_pos,"/",y.shape[0])
    print("  true_neg / number sample: ", true_neg,"/",y.shape[0])


def evaluate_model(x, y, model,type_evaluate):
    y_predict = model.predict(x)

    if(type_evaluate == "binary_cross_entropy"):
        binary_cross_entropy(y_predict, y)
    elif(type_evaluate == "positive_negative"):
        positive_negative(y_predict, y)
