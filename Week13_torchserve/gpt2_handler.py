from .base_handler import BaseHandler
from .preprocess import Preprocessing
import torch

import pickle


class Gpt2_handler(BaseHandler):
    def __init__(self):
        super(Gpt2_handler, self).__init__()
        with open("pre_train_tokenizer5.pkl", 'rb') as file:
            self.tokenizer = pickle.load(file)
        self.start = [0] * 100

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

    def get_next_token_prob(self, list_token_ids: torch.Tensor, kv_cache):
        prob, kv_cache_new = self.model(list_token_ids, kv_cache)
        return prob, kv_cache_new

    def generator_no_use_kv(self, list_token_ids, generator, kv_cache, args):
        for i in range(args.max_new_tokens):
            kv_cache = torch.empty((1))
            next_token_prob, kv_cache = generator.get_next_token_prob(list_token_ids, kv_cache)
            top1 = generator.get_top1(next_token_prob)
            new_value = top1[:, list_token_ids.shape[1] - 1]
            new_value = new_value.reshape(list_token_ids.shape[0], 1)
            list_token_ids = torch.cat((list_token_ids, new_value), dim=-1)
        return list_token_ids

    def generator_use_kv(self, list_token_ids, kv_cache):
        for i in range(32):
            next_token_prob, kv_cache = self.get_next_token_prob(list_token_ids, kv_cache)
            top1 = self.get_top1(next_token_prob)
            if (i == 0):
                new_value = top1[:, list_token_ids.shape[1] - 1]
            else:
                new_value = top1[:, 0]
            new_value = new_value.reshape(list_token_ids.shape[0], 1)
            list_token_ids = torch.cat((list_token_ids, new_value), dim=-1)
        return list_token_ids
    def preprocess(self, data):
        # Implement preprocessing logic
        # Convert input data to the format expected by the model
        list_sentences = data[0]['body']['data']
        list_token_ids = []
        max_len_sentence = 0
        for sentence in list_sentences:
            token_ids = list_sentences.encode(Preprocessing().preprocess_text(sentence))
            list_token_ids.append(token_ids)
            max_len_sentence = max(max_len_sentence, len(token_ids))

        for i in range(len(list_token_ids)):
            self.start[i] = (max_len_sentence - len(list_token_ids[i]))
            list_token_ids[i] = [2] * (max_len_sentence - len(list_token_ids[i])) + list_token_ids[i]

        list_token_ids = torch.tensor(list_token_ids, dtype=torch.long)
        print("in data preprocess ", list_sentences)

        return list_token_ids

    def inference(self, list_token_ids, *args, **kwargs):
        kv_cache = torch.empty((1))
        list_token_ids_no_use_kv = self.generator_no_use_kv(list_token_ids, kv_cache)
        list_output = []
        for id in range(len(list_token_ids_no_use_kv)):
            token = list_token_ids_no_use_kv[id].tolist()
            print(self.decode(token[self.start[id]:]))
            list_output.append(self.decode(token[self.start[id]:]))


        print("in data inference ", list_output)
        return list_output

    def postprocess(self, data):
        #print("in data postprocess ", data)
        # Implement postprocessing logic
        # Convert model output to the desired format
        return [("data", data)]

