import torch

class Tokenizer:
    def __init__(self, top_k):
        self.token2id = {}
        self.id2token = {}
        self.top_k_token = []
        self.top_k = top_k
        self.special_tokens = {"bos_token": "<|beginoftext|>", "eos_token": "<|endoftext|>",
                      "pad_token": "<|padding|>", "unk_token": "<|unknown|>",
                      "eov_token": ".", "con_token": "<|consonant|>"}
    def build(self,vi_sentences):
        for key, value in self.special_tokens.items():
            if value not in self.token2id:
                self.token2id[value] = len(self.token2id)

        token2freq = {}
        count_token = {}
        for sentence in vi_sentences:
            for word in sentence:
                if(word not in self.token2id):
                    token2freq[word] = len(token2freq)
                    count_token[word] = 0
                count_token[word] = count_token[word] + 1
        sorted_count_token = dict(sorted(count_token.items(), key = lambda x:x[1], reverse = True))
        self.top_k_token = list(sorted_count_token.keys())[:self.top_k]
        print(len(sorted_count_token))
        if(len(sorted_count_token) < self.top_k):
            print('SOS')

        for key, value in token2freq.items():
            if(key in self.top_k_token and key not in self.token2id):
                self.token2id[key] = len(self.token2id)
                # print(key, len(self.token2id))

        self.id2token = {value: key for key, value in self.token2id.items()}

    def encoder(self,text):
        token_ids = []
        for word in text:
            if(word in self.token2id):
                token_ids.append(self.token2id[word])
            else:
                token_ids.append(self.token2id[self.special_tokens['unk_token']])
        return token_ids

    def decoder(self,token_ids):
        tokens = [self.id2token[x] for x in token_ids]
        return tokens

    def call(self, sentences, max_len = 64):
        pad_token_id = self.token2id[self.special_tokens["pad_token"]]
        bos_token = self.token2id[self.special_tokens["bos_token"]]
        eos_token_id = self.token2id[self.special_tokens["eos_token"]]
        list_token_ids = []
        for sentence in sentences:
            token_ids = self.encoder(sentence)
            if(len(token_ids) > max_len):
                token_ids = token_ids[:max_len]
            else:
                token_ids += pad_token_id * (max_len - len(token_ids))
            token_ids = [bos_token] + token_ids[:62] + [eos_token_id]
            list_token_ids.append(token_ids)
        return list_token_ids, self.token2id