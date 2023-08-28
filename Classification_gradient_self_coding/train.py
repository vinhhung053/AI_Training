from config import get_args
from model import *
from dataloader import load_data
def main():
    args = get_args()
    x_train, y_train, x_val, y_val, x_test, y_test = load_data()
    model = gradient_descent(x_train, y_train, max_iter = args.max_iter, learning_rate = args.learning_rate)


if __name__ == "__main__":
    main()