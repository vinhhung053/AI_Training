import argparse

def get_args():
    parser = argparse.ArgumentParser(description='test Week_1_2 anh Chien')

    parser.add_argument('--max_iter',
                        type=int,
                        default= 400,
                        help='max_iter')

    parser.add_argument('--learning_rate',
                        type= float,
                        default= 0.01,
                        help='The learning rate')

    args = parser.parse_args()
    return args