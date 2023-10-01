import torch
import torch.nn as nn
import pandas as pd
import argparse
from preprocess import Prerpocessing
from tokenizer import Tokenizer

from dataloader import Dataloader
from model import GPT2
import torch.optim as optim
from trainer import Trainer


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
                        default= 10000,
                        help='number token used')

    parser.add_argument('--batch_size',
                        type=int,
                        default=16,
                        help='Batch size data')
    parser.add_argument('--shuffle',
                        type=bool,
                        default=True,
                        help='Shuffle data')

    parser.add_argument('--hidden_sz',
                        type=int,
                        default= 300,
                        help='Hidden size in Attention')

    parser.add_argument('--num_block',
                        type=int,
                        default= 2,
                        help='Num transformer block')

    parser.add_argument('--lr',
                        type=float,
                        default= 0.01,
                        help='Num transformer block')
    args = parser.parse_args()
    return args

def read_data_csv(path):
    return pd.read_csv(path)

# word -> id
def convert_word2id(args, list_tokens):
    tokenizer = Tokenizer(args.top_k)
    tokenizer.build(list_tokens)
    list_token_ids, token2id = tokenizer.call(list_tokens)
    return list_token_ids, token2id

# chuyen id -> vector
def convert_id2vector(args, list_token_ids, token2id):
    embedding = nn.Embedding(len(token2id), args.vector_size)
    list_token_tensor = torch.LongTensor(list_token_ids)
    embedding_list = embedding(list_token_tensor)
    return embedding_list

def split_data(x, y):
    mid_split = int(x.shape[0] * 10 / 9)
    return x[:mid_split], y[:mid_split], x[mid_split:], y[mid_split:]

def main():
    print("Start prepare for training ...")
    args = get_args()

    # Đọc file CSV và lưu vào một DataFrame
    data = read_data_csv(args.train_path)
    sentences_train = data['excerpt']
    # lower + split
    list_tokens = list(map(Prerpocessing().preprocess_text,sentences_train))
    # word -> id
    list_token_ids, token2id = convert_word2id(args, list_tokens)
    y = torch.LongTensor(list_token_ids)

    # chuyen id -> vector
    embedding_list = convert_id2vector(args,list_token_ids, token2id)

    #Split data
    x_train, y_train, x_test, y_test = split_data(embedding_list, y)

    train_loader = Dataloader(x_train, y_train, batch_size=args.batch_size, shuffle=args.shuffle)
    classifier = GPT2(input_sz=x_train.shape[-1], hidden_sz=args.hidden_sz, vocab_size=len(token2id), num_block=args.num_block)
    optimizer = optim.Adam(classifier.parameters(), lr=args.lr)  # 0.002 dives 85% acc
    criterion = nn.CrossEntropyLoss()

    trainer = Trainer(classifier, optimizer, criterion)
    trainer.train(train_loader)

if __name__ == '__main__':
    main()