import argparse
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score

def get_args():
    parser = argparse.ArgumentParser(description='SGDC parameter')

    parser.add_argument('--max_iter',
                        type=int,
                        default= 200,
                        help='max_iter')

    parser.add_argument('--learning_rate',
                        type=str,
                        default="optimal",
                        help='The learning rate schedule:‘constant’ ‘optimal’ ‘invscaling’ ‘adaptive’')

    args = parser.parse_args()
    return args

def load_data():
    digits = load_digits()
    x = digits.data
    y = digits.target
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)
    return x_train, x_test, y_train, y_test
def train_sgd_classifier(x_train, y_train):
    clt = SGDClassifier(max_iter= 1000, random_state = 2)
    clt.fit(x_train, y_train)
    return clt


def main():
    x_train, x_test, y_train, y_test = load_data()
    classifier = train_sgd_classifier(x_train, y_train)
    y_predict = classifier.predict(x_test)
    print(y_predict)
    print(100 * accuracy_score(y_test, y_predict))
main()


