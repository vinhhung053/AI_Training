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
                        default=1,
                        help='number of tokens generated')

    parser.add_argument('--pre_model_path',
                        default="pre_model/pre_train_model3.pth",
                        help='pre model path')

    parser.add_argument('--pre_tokenizer_path',
                        default="pre_tokenizer/pre_train_tokenizer3.pkl",
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
        score, token_id = torch.max(prob, dim = -1)
        return token_id

    def get_topk(self, prob, k = 2):
        scores, token_ids = torch.topk(prob, k = k, dim = -1)
        return token_ids

    def get_next_token_prob(self, token_ids:torch.Tensor, kv_cache):
        prob, kv_cache_new = self.model(token_ids, kv_cache)
        return prob, kv_cache_new


def main():
    t1 = time.time()
    args = get_args()
    generator = Generator(args)

    token_ids = generator.encode(Prerpocessing().preprocess_text(args.prompt))

    kv_cache = None
    for i in range(20):
        # print(token_ids)
        token_ids = torch.tensor(token_ids, dtype=torch.long)
        next_token_prob, kv_cache = generator.get_next_token_prob(token_ids, kv_cache)
        kv_cache = None
        # print(next_token_prob)
        top1 = generator.get_top1(next_token_prob)
        print(top1)

        new_value = torch.tensor([top1[token_ids.shape[0] - 1]])

        # if(i == 0):
        #     new_value = torch.tensor([top1[token_ids.shape[0] - 1]])
        # else:
        #     new_value = torch.tensor([top1[0]])

        token_ids = torch.cat((token_ids, new_value), dim=-1)
    print(token_ids)

    tokens = token_ids.tolist()
    print(generator.decode(tokens))
    t2 = time.time()
    print('Elapsed time: {} seconds'.format((t2 - t1)))

    # top1 = top1.numpy()
    # print(generator.decode(top1))


if __name__ == '__main__':
    main()

