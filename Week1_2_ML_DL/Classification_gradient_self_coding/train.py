from config import get_args
from model import *
from dataloader import load_data
from evaluate import evaluate_model
def main():
    args = get_args()
    x_train, y_train, x_val, y_val, x_test, y_test = load_data()
    w1, w2 = gradient_descent(x_train, y_train, max_iter = args.max_iter, learning_rate = args.learning_rate)
    print("Cross entropy val: ",evaluate_model(x_val,y_val,w1,w2))
    print("Cross entropy test: ",evaluate_model(x_test,y_test,w1,w2))

if __name__ == "__main__":
    main()