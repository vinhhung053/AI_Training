from config import get_args
from model import *
import pandas as pd
from sklearn.model_selection import train_test_split
from dataloader import Data_loader
from model import Gradient_descent
from evaluate import evaluate_model
def main():
    args = get_args()

    #Doc du lieu
    data_file_path = "/home/lap13385/Downloads/breast+cancer+wisconsin+diagnostic/wdbc.data"
    data = pd.read_csv(data_file_path, delimiter= ',')
    x = data.iloc[:,2:]
    y = (data.iloc[:, 1] == 'M').astype(int)
    x_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size=0.4, random_state=40)
    x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=40)

    #Khoi tao dataloader
    data_loader = Data_loader(x_train, y_train,batch_size=args.batch_size, shuffle=args.shuffle)
    model_gd = Gradient_descent(max_iter=args.max_iter, learning_rate=args.learning_rate)
    model_gd.fit(data_loader)
    print("Cross entropy val: ", evaluate_model(x_val, y_val, model_gd))
    print("Cross entropy test: ", evaluate_model(x_test, y_test, model_gd))


if __name__ == "__main__":
    main()