import torch
import torch.nn as nn

class Embedding:
    def __init__(self, vocab_size, embedding_dim):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

    def word2vec_convert(self, list_token_ids, token2id):
        embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        list_token_tensor = torch.LongTensor(list_token_ids)
        embedding_list_token_ids = embedding(list_token_tensor[0])
        return embedding_list_token_ids
