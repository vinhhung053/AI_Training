import pandas as pd
from sklearn.model_selection import train_test_split

def load_data():
    data_file_path = "/home/lap13385/Downloads/breast+cancer+wisconsin+diagnostic/wdbc.data"
    data = pd.read_csv(data_file_path, delimiter= ',')
    x = data.iloc[:,2:]
    y = data.iloc[:,1]
    y = y.apply(lambda x: 1 if x == "M" else 0)
    y = y.values.reshape(-1,1)
    x_train, x_temp, y_train, y_temp = train_test_split(x,y, test_size = 0.4, random_state = 42)
    x_val, x_test, y_val, y_test= train_test_split(x_temp,y_temp, test_size = 0.5, random_state = 42)
    return x_train, y_train, x_val, y_val, x_test, y_test
