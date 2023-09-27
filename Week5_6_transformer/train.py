import string
import torch
import torch.nn as nn
import pandas as pd
import argparse
from preprocess import Prerpocessing
from tokenizer import Tokenizer
from tqdm.notebook import tqdm
from dataloader import Dataloader
from sklearn.model_selection import train_test_split
def get_args():
    parser = argparse.ArgumentParser(description='training GPT2')
    parser.add_argument('--train_path',
                        default="source/commonlitreadabilityprize/train.csv",
                        help='data train path')

    parser.add_argument('--test_path',
                        default="commonlitreadabilityprize/train.csv",
                        help='data test path')

    parser.add_argument('--vector_size',
                        type=int,
                        default=100,
                        help='vector size of word embedding')

    parser.add_argument('--top_k',
                        type = int,
                        default= 1000,
                        help='number token used')

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

def read_data_csv(path):
    return pd.read_csv(path)

def train(train_loader):
    device = torch.device('cpu')
    classifier = Net().to(device)
    optimizer = optim.Adam(classifier.parameters(), lr=0.003)  # 0.002 dives 85% acc

    epoch_bar = tqdm(range(40),
                     desc="Training",
                     position=0,
                     total=2)
    for epoch in epoch_bar:
        batch_bar = tqdm(enumerate(train_loader),
                         desc="Epoch: {}".format(str(epoch)),
                         position=1)

        optimizer.zero_grad()
        batch_data = train_loader.next()


def main():
    args = get_args()

    # Đọc file CSV và lưu vào một DataFrame
    data = read_data_csv(args.train_path)
    sentences_train = data['excerpt'][:200]
    # lower + split
    list_tokens = list(map(Prerpocessing().preprocess_text,sentences_train))
    # word -> id
    tokenizer = Tokenizer(args.top_k)
    tokenizer.build(list_tokens)
    list_token_ids, token2id = tokenizer.call(list_tokens)
    # chuyen id -> vector
    embedding = nn.Embedding(len(token2id), args.vector_size)
    list_token_tensor = torch.LongTensor(list_token_ids)
    embedding_list = embedding(list_token_tensor)
    print(embedding_list.shape)
    mid_split = int(embedding_list.shape[0] * 10 / 9)
    data_train = embedding_list[:mid_split]
    data_test = embedding_list[mid_split:]
    train_loader = Dataloader(data_train, batch_size = args.batch_size, shuffle = args.shuffle)
    # train(train_loader)
    print(next(train_loader).size())
if __name__ == '__main__':
    main()