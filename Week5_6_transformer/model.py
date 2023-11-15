import math
import torch
import torch.nn as nn
from typing import List

class Attention(nn.Module):

    def __init__(self, input_sz: int, hidden_sz: int):
        super().__init__()
        self.input_sz = input_sz
        self.hidden_sz = hidden_sz
        # W_q, W_k, W_v = (len_vec , hidden)
        self.W_q = nn.Parameter(torch.Tensor(input_sz, hidden_sz))
        self.W_k = nn.Parameter(torch.Tensor(input_sz, hidden_sz))
        self.W_v = nn.Parameter(torch.Tensor(input_sz, hidden_sz))

        self.inp_linear_self_attention = nn.Linear(input_sz, input_sz)  # Input (seq_sz, len_vec) -> Output: (seq_sz, len_vec)

    def forward(self, x_batch, kv_cache):
        "x.shape = (batch_size, seq_sz, len_vector)"
        _, seq_sz, len_vec = x_batch.size()
        x_batch_ = self.inp_linear_self_attention(x_batch)
        # Matmul
        Q = x_batch_.matmul(self.W_q)  # Q shape [seq_sz, hidden)
        K = x_batch_.matmul(self.W_k)  # Q shape [seq_sz, hidden)
        V = x_batch_.matmul(self.W_v)   # Q shape [seq_sz, hidden)
        if kv_cache.shape != torch.empty((2)).shape:
            print(kv_cache)
            old_k = kv_cache[0]
            old_v = kv_cache[1]
            # print(old_k.shape)
            # print(K.shape)
            new_k = torch.cat((old_k, K), dim=1)
            new_v = torch.cat((old_v, V), dim=1)
            K = new_k
            V = new_v
        # print(K, V)
        current_cache = [K,V]
        current_cache = torch.stack(current_cache)
        K_T = K.transpose(1,2)
        I = Q.matmul(K_T)  # I shape [seq_sz, seq_sz]
        # Attention Mask
        if kv_cache.shape != torch.empty((2)).shape:
            mask = torch.zeros((1, K.shape[0]))
        else:
            mask = torch.triu(torch.ones(seq_sz, seq_sz, dtype=torch.bool), diagonal=1)
            I[:,mask] = float('-inf')
        F = torch.matmul(torch.nn.functional.softmax(I, dim=-1), V)  # [seq_sz, hidden]
        return F, current_cache # [K,V]

class MultiAttention(nn.Module):
    def __init__(self, input_sz: int, hidden_sz: int, num_heads: int):  # (len_vec , hidden, num head)
        super().__init__()

        self.input_sz = input_sz
        self.hidden_sz = hidden_sz
        self.num_heads = num_heads

        self.out_linear_self_attention_layer = nn.Linear(self.num_heads * self.hidden_sz, self.input_sz)
        # Create multi attention
        self.attention_heads = nn.ModuleList([Attention(input_sz, hidden_sz) for _ in range(num_heads)])

    def forward(self, x_batch, kv_caches):
        # output_attention_heads = [(head(x_batch, kv_caches[index])) for index, head in enumerate(self.attention_heads)]
        head_outputs = [(head(x_batch, kv_caches[index]))[0] for index, head in enumerate(self.attention_heads)]
        kv_caches = [(head(x_batch, kv_caches[index]))[1] for index, head in enumerate(self.attention_heads)]
        kv_caches = torch.stack(kv_caches)
        multi_head_output = torch.cat(head_outputs, dim=-1)
        output_multi_self_attention = self.out_linear_self_attention_layer(multi_head_output)
        return output_multi_self_attention, kv_caches # [[K,V] [K,V] ... ]]


class GPT2Block(nn.Module):
    def __init__(self, input_sz: int, hidden_sz: int, num_head: int):
        super().__init__()
        self.input_sz = input_sz
        self.hidden_sz = hidden_sz

        self.PE_layer_norm = nn.LayerNorm(input_sz)
        self.layer_norm_transformer_block = nn.LayerNorm(input_sz)

        self.multi_attentions = MultiAttention(input_sz, hidden_sz, num_head)

        # Linear function
        self.linear_before_gelu = nn.Linear(input_sz, hidden_sz)
        self.linear_after_gelu = nn.Linear(hidden_sz, input_sz)


    def gelu(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

    def forward(self, x_batch, kv_cache):
        "x.shape = (seq_sz, len_vector)"
        residual = x_batch
        x_batch = self.PE_layer_norm(x_batch)
        x_batch, kv_caches_updated = self.multi_attentions(x_batch, kv_cache)
        x_batch = (x_batch + residual)  # [seq_sz, len_vec]
        residual = x_batch  # [seq_sz, len_vec]
        x_batch = self.linear_before_gelu(x_batch)
        x_batch = self.gelu(x_batch)
        x_batch = self.linear_after_gelu(x_batch) + residual  # Output transformer block [seq_sz, len_vec]
        return x_batch, kv_caches_updated #[[K V] [KV] [KV]]


class GPT2(nn.Module):
    def __init__(self, input_sz: int, hidden_sz: int, vocab_size: int, num_block: int, num_head: int):
        super().__init__()
        self.input_sz = input_sz
        self.hidden_sz = hidden_sz
        self.num_block = num_block
        self.target_layer_norm = nn.LayerNorm(input_sz)
        self.target_linear = nn.Linear(input_sz, vocab_size)  # Input (seq_sz, len_vec) -> Output: (seq_sz, vocab_size)
        self.gpt2block = nn.ModuleList([GPT2Block(input_sz, input_sz, num_head) for _ in range(num_block)])
        self.num_head = num_head

        self.embedding = nn.Embedding(vocab_size, input_sz)
        self.init_weights()

    def init_weights(self):
        stdv = 0.1
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def positional_encoding(self, x, current_len, kv_caches):
        seq_sz, len_vec = x.size()
        encoding = torch.zeros(seq_sz, len_vec)  # Tạo tensor mới để lưu trữ kết quả
        if kv_caches.shape != torch.empty((self.num_block,self.num_head,2)).shape:
            encoding = torch.zeros(1, len_vec)
            for i in range(len_vec):
                if i % 2 == 0:
                    encoding[0][i] = math.sin(current_len.item() / (10000 ** (i / len_vec)))
                else:
                    encoding[0][i] = math.cos(current_len.item() / (10000 ** (i / len_vec)))
        else:
            for pos in range(seq_sz):
                for i in range(len_vec):
                    if i % 2 == 0:
                        encoding[pos][i] = math.sin(pos / (10000 ** (i / len_vec)))
                    else:
                        encoding[pos][i] = math.cos(pos / (10000 ** (i / len_vec)))
        return encoding

    def forward(self, x_batch, kv_caches):
        current_len = torch.tensor(x_batch.shape[1] - 2)
        if kv_caches.shape == torch.empty((1)).shape:
            kv_caches = torch.empty((self.num_block,self.num_head,2))
        else:
            current_len = current_len + 1
            x_batch = x_batch[:, -1]
            x_batch = x_batch.reshape(-1, 1)
        x_batch = torch.tensor(x_batch)
        x_batch = self.embedding(x_batch)
        for index in range(x_batch.shape[0]):
            x_batch[index] = x_batch[index] + self.positional_encoding(x_batch[index], current_len, kv_caches)
        new_kv_caches = []
        for index, gpt2block in enumerate(self.gpt2block):
            x_batch, updated_cache = gpt2block(x_batch, kv_caches[index])
            new_kv_caches.append(updated_cache)

        new_kv_caches = torch.stack(new_kv_caches)
        out = self.target_layer_norm(x_batch)
        out = self.target_linear(out)  # out [seq_sz, vocab_size]
        return out, new_kv_caches
