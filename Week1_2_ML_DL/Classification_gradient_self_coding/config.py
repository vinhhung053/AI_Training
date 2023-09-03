import argparse


def get_args():
    parser = argparse.ArgumentParser(description='test Week_1_2 anh Chien')

    parser.add_argument('--type_evaluate',
                        type = str,
                        default="binany_cross_entropy", # "binany_cross_entropy" or "positive_negative"
                        help='max_iter')

    parser.add_argument('--max_iter',
                        type=int,
                        default=100,
                        help='max_iter')

    parser.add_argument('--learning_rate',
                        type=float,
                        default=0.1,
                        help='The learning rate')
    parser.add_argument('--batch_size',
                        type=int,
                        default=32,
                        help='Batch size data')
    parser.add_argument('--shuffle',
                        type=bool,
                        default=True,
                        help='Shuffle data')
    args = parser.parse_args()
    return args
