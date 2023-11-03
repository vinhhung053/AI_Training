import torch
import argparse
import pickle
import time
import numpy
from preprocess import Prerpocessing

def get_args():
    parser = argparse.ArgumentParser(description='hi')

    parser.add_argument('--prompt',
                        type=str,
                        required=True,
                        help='prompt')

    parser.add_argument('--max_new_tokens',
                        type=int,
                        default=20,
                        help='number of tokens generated')

    parser.add_argument('--pre_model_path',
                        default="pre_model/pre_train_model5.pth",
                        help='pre model path')

    parser.add_argument('--pre_tokenizer_path',
                        default="pre_tokenizer/pre_train_tokenizer5.pkl",
                        help='pre tokenizer path')
    args = parser.parse_args()
    return args

class Generator:
    def __init__(self, args):
        with(open(args.pre_tokenizer_path, 'rb') as file):
            self.tokenizer = pickle.load(file)
        self.model = torch.load(args.pre_model_path)
        self.max_new_tokens = args.max_new_tokens

    def encode(self, text):
        return self.tokenizer.encode(text)

    def decode(self, text):
        return self.tokenizer.decode(text)

    def get_top1(self, prob):
        score, token_id = torch.max(prob, dim=-1)
        return token_id

    def get_topk(self, prob, k=2):
        scores, token_ids = torch.topk(prob, k=k, dim=-1)
        return token_ids

    def get_next_token_prob(self, list_token_ids:torch.Tensor, kv_cache):
        prob, kv_cache_new = self.model(list_token_ids, kv_cache)
        return prob, kv_cache_new


def generator_no_use_kv(list_token_ids, generator, kv_cache, args):
    for i in range(args.max_new_tokens):
        kv_cache = None
        next_token_prob, kv_cache = generator.get_next_token_prob(list_token_ids, kv_cache)
        top1 = generator.get_top1(next_token_prob)
        new_value = top1[:, list_token_ids.shape[1] - 1]
        new_value = new_value.reshape(list_token_ids.shape[0],1)
        list_token_ids = torch.cat((list_token_ids, new_value), dim=-1)
    return list_token_ids


def generator_use_kv(list_token_ids, generator, kv_cache, args):
    for i in range(args.max_new_tokens):
        next_token_prob, kv_cache = generator.get_next_token_prob(list_token_ids, kv_cache)
        top1 = generator.get_top1(next_token_prob)
        if(i == 0):
            new_value = top1[:, list_token_ids.shape[1] - 1]
        else:
            new_value = top1[:, 0]
        new_value = new_value.reshape(list_token_ids.shape[0],1)
        list_token_ids = torch.cat((list_token_ids, new_value), dim=-1)
    return list_token_ids


def main():
    t1 = time.time()
    args = get_args()
    generator = Generator(args)
    # sentences = ["when the young people", "all", "he is my"]
    sentences = ["when the young people"]
    # sentences = ["i am mr hung me"]
    list_token_ids = []
    max_len_sentence = 0
    start = [0] * len(sentences)
    for sentence in sentences:
        token_ids = generator.encode(Prerpocessing().preprocess_text(sentence))
        list_token_ids.append(token_ids)
        max_len_sentence = max(max_len_sentence, len(token_ids))

    for i in range(len(list_token_ids)):
        start[i] = (max_len_sentence - len(list_token_ids[i]))
        list_token_ids[i] = [2] * (max_len_sentence - len(list_token_ids[i])) + list_token_ids[i]
    kv_cache = None
    list_token_ids = torch.tensor(list_token_ids, dtype=torch.long)

    print("------------------no use kv cache-------------------------------------------")
    list_token_ids_no_use_kv = generator_no_use_kv(list_token_ids,generator,kv_cache, args)
    for id in range(len(list_token_ids_no_use_kv)):
        token = list_token_ids_no_use_kv[id].tolist()
        print(generator.decode(token[start[id]:]))
    print("----------------------------------------------------------------------------")

    print("--------------- use kv cache -----------------------------------------------")
    list_token_ids_use_kv = generator_use_kv(list_token_ids, generator, kv_cache, args)
    for id in range(len(list_token_ids_use_kv)):
        token = list_token_ids_use_kv[id].tolist()
        print(generator.decode(token[start[id]:]))
    print("----------------------------------------------------------------------------")


    t2 = time.time()
    print('Elapsed time: {} seconds'.format((t2 - t1)))

    # top1 = top1.numpy()
    # print(generator.decode(top1))


if __name__ == '__main__':
    main()

