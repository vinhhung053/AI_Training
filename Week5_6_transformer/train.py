import string
import torch
import torch.nn as nn
import pandas as pd
import argparse
from preprocess import Prerpocessing
from tokenizer import Tokenizer
from tqdm.notebook import tqdm
from dataloader import Dataloader
from model import GPT2, Attention
import torch.optim as optim

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
                        default= 10000,
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

def update_lr(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr'] *= 0.95

def train(train_loader, input_sz, hidden_sz, len_word):
    classifier = GPT2(input_sz, hidden_sz, len_word)
    optimizer = optim.Adam(classifier.parameters(), lr=0.01)  # 0.002 dives 85% acc
    criterion = nn.CrossEntropyLoss()
    print(len_word)
    epoch_bar = tqdm(range(40),
                     desc="Training",
                     position=0,
                     total=2)
    for epoch in epoch_bar:
        batch_bar = tqdm(enumerate(train_loader),
                         desc="Epoch: {}".format(str(epoch)),
                         position=1)
        for i, (x_batch, y_batch) in batch_bar:

            optimizer.zero_grad()
            preds = [classifier(x) for x in x_batch]
            preds = torch.stack(preds)

            preds = preds.view(-1,preds.shape[-1])
            y_batch = y_batch.view(-1)
            # print(preds.shape)
            # print(y_batch.shape)

            loss = criterion(preds, y_batch)

            loss.backward(retain_graph=True)
            optimizer.step()
            acc = (preds.argmax(dim=1) == y_batch).float().mean().cpu().item()
            print(acc)
        update_lr(optimizer)  # cap nhat lr

def main():
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

    # print(embedding_list.shape)
    mid_split = int(embedding_list.shape[0] * 10 / 9)
    x_train = embedding_list[:mid_split]
    x_test = embedding_list[mid_split:]
    y_train = y[:mid_split]
    y_test = y[:mid_split]
    train_loader = Dataloader(x_train, y_train, batch_size = args.batch_size, shuffle = args.shuffle)
    train(train_loader,x_train.shape[-1], 300, len(token2id))

if __name__ == '__main__':
    main()