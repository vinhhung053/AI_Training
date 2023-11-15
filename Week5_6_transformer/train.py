import torch
import torch.nn as nn
import pandas as pd
import argparse
import time
import preprocess
from preprocess import Prerpocessing
from tokenizer import Tokenizer
import pickle

from dataloader import Dataloader
from model import GPT2
import torch.optim as optim
from trainer import Trainer


def get_args():
    parser = argparse.ArgumentParser(description='training GPT2')
    parser.add_argument('--train_path',
                        default="source/train.csv",
                        help='data train path')
    parser.add_argument('--pre_model_path',
                        default="pre_model/pre_train_model5.pth",
                        help='pre model path')

    parser.add_argument('--pre_tokenizer_path',
                        default="pre_tokenizer/pre_train_tokenizer5.pkl",
                        help='pre tokenizer path')
    parser.add_argument('--test_path',
                        default="commonlitreadabilityprize/train.csv",
                        help='data test path')

    parser.add_argument('--vector_size',
                        type=int,
                        default=128,#1028
                        help='vector size of word embedding')

    parser.add_argument('--max_len',
                        type=int,
                        default=32, # 32
                        help='max len senteces')

    parser.add_argument('--max_word',
                        type=int,
                        default=3000,
                        help='number token used')

    parser.add_argument('--batch_size',
                        type=int,
                        default=8,#8
                        help='Batch size data')
    parser.add_argument('--shuffle',
                        type=bool,
                        default=True,
                        help='Shuffle data')

    parser.add_argument('--hidden_sz',
                        type=int,
                        default=10,
                        help='Hidden size in Attention')

    parser.add_argument('--num_block',
                        type=int,
                        default=2,
                        help='Num transformer block')

    parser.add_argument('--lr',
                        type=float,
                        default=0.001,
                        help='Num transformer block')

    parser.add_argument('--epoch',
                        type=int,
                        default=50,
                        help='Num epoch')

    parser.add_argument('--num_head',
                        type=int,
                        default=2,
                        help='Num head')

    args = parser.parse_args()
    return args


def read_data_csv(path):
    return pd.read_csv(path)


def split_data(x, y):
    mid_split = int(x.shape[0] * 10 / 9)
    return x[:mid_split], y[:mid_split], x[mid_split:], y[mid_split:]


def main():
    t1 = time.time()
    print("Start prepare for training ...")
    args = get_args()

    # Đọc file CSV và lưu vào một DataFrame
    data = read_data_csv(args.train_path)
    data = data[:100]
    sentences = data['excerpt']
    # lower + split
    list_words = list(map(Prerpocessing().preprocess_text,sentences))
    # word -> id
    tokenizer = Tokenizer(args.max_word)
    tokenizer.build(list_words)
    # print(tokenizer.encode(Prerpocessing().preprocess_text("when i was young, i play")))
    # print(tokenizer.decode([3,157,4,15]))
    # exit()

    # #save tokenizer file
    # token2id = tokenizer.get_token_2id()
    # with open('pre_tokenizer/tokenizer.txt', 'w') as file:
    #     for word, id in token2id.items():
    #         file.write('{} {}\n'.format(word, id))

    #save tokenizer
    with open(args.pre_tokenizer_path, 'wb') as file:
        pickle.dump(tokenizer, file)

    list_token_ids = tokenizer(list_words, args.max_len)
    list_token_tensor = torch.LongTensor(list_token_ids)
    #Split data
    x_train, y_train, x_test, y_test = split_data(list_token_tensor, list_token_tensor)

    train_loader = Dataloader(x_train, y_train, batch_size=args.batch_size, shuffle=args.shuffle)
    test_loader = Dataloader(x_test, y_test, batch_size=args.batch_size, shuffle=args.shuffle)

    model = GPT2(input_sz=args.vector_size, hidden_sz=args.hidden_sz, vocab_size=len(tokenizer), num_block=args.num_block, num_head = args.num_head)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    trainer = Trainer(model, optimizer, criterion, args.epoch)
    trainer.train(train_loader, args)
    trainer.val(test_loader)
    torch.save(model, args.pre_model_path)
    model.eval()
    m = torch.jit.script(model)
    m.save("m.pt")
    t2 = time.time()
    print('Elapsed time: {} seconds'.format((t2 - t1)))


if __name__ == '__main__':
    main()