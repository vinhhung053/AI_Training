import math
import torch
import torch.nn as nn
import numpy as np
import tensorflow as tf


class GPT2(nn.Module):
    def __init__(self, input_sz: int, hidden_sz:int, vocab_size:int):
        super().init()
        self.input_sz = input_sz
        self.hidden_sz = hidden_sz

        # W_q, W_k, W_v = (len_vec , hidden)
        self.W_q = nn.Parameter(torch.Tensor(input_sz, hidden_sz))
        self.W_k = nn.Parameter(torch.Tensor(input_sz, hidden_sz))
        self.W_v = nn.Parameter(torch.Tensor(input_sz, hidden_sz))

        # Layer norm
        self.layer_norm1 = nn.LayerNorm(input_sz)
        self.layer_norm2 = nn.LayerNorm(input_sz)
        self.layer_norm3 = nn.LayerNorm(input_sz)

        # Linear function
        self.linear1 = nn.Linear(input_sz, input_sz)  # Input (batch, len_vec) -> Output: (batch, len_vec)
        self.linear2 = nn.Linear(input_sz, input_sz)  # Input (batch, len_vec) -> Output: (batch, len_vec)
        self.linear3 = nn.Linear(hidden_sz, 4*hidden_sz)
        self.linear4 = nn.Linear(4*hidden_sz, hidden_sz)
        self.linear5 = nn.Linear(input_sz, vocab_size)  # Input (batch, len_vec) -> Output: (batch, len_vec)

        self.init_weights()
    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def positional_encoding(self,x):
        bs, seq_sz, len_vec = x.size()
        for id in range(bs):
            for pos in range(seq_sz):  # pos la vi tri cua tu trong cau
                for i in range(len_vec):  # i chi so cac pt trong PE
                    if(i % 2 == 0):
                        x[id][pos][i] = math.sin(pos / (1000 ** (i / len_vec)))
                    else:
                        x[id][pos][i] = math.cos(pos / (1000 ** (i / len_vec)))

    def softmax(x, axis=1):
        exp_tensor = torch.exp(x - torch.max(x,dim=axis, keepdim=True)[0])
        softmax_tensor = exp_tensor / exp_tensor.sum(dim=axis, keepdim=True)
        return softmax_tensor

    def gelu(x):
        return 0.5 * x * (1 + tf.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3))))
    def forward(self, x, init_states = None):
        "x.shape = (batch, seq_sz, len_vector)"
        bs, seq_sz, len_vec = x.size()
        for id in range(bs):
            x_pre = (x[id][0] * seq_sz) # Input Khoi tao ban dau x_pre = ['<s>'] * seq_sz
            for i in range(bs):
                y = x[id, :, :]  # x_id shape [seq_sz, len_vec]
                x_i = x_i + self.positional_encoding(x_pre)
                save1_x_i = x_i
                x_i = self.layer_norm1(x_i)
                x_i = self.linear(x_i)  # Output: (seq_sz, len_vec)
                # Matmul
                Q = x_i @ self.W_Q  # Q shape [seq_sz, hidden)
                K = x_i @ self.W_K  # Q shape [seq_sz, hidden)
                V = x_i @ self.W_V  # Q shape [seq_sz, hidden)

                I = Q @ K.T  # I shape [seq_sz, seq_sz]
                # Attention Mask
                for i in range(seq_sz):
                    for j in range(seq_sz):
                        if(i < j):
                            I[i][j] = -10**10
                F = self.softmax(I/math.sqrt(len_vec)) * V # [seq_sz, hidden]
                x_i = self.layer_norm2(F) + save1_x_i # [seq_sz, hidden]
                x_i = self.layer_norm2(x_i) # [seq_sz, hidden]
                save2_x_i = x_i # [seq_sz, hidden]
                x_i = self.linear3(x_i)
                x_i = self.gelu(x_i)
                x_i = self.linear4(x_i) + save2_x_i # Output transformer block [seq_sz, len_vec]

                out = self.layer_norm3(x_i)
                out = self.linear4(x_i) # out [seq_sz, out]
                x_pre[id][i+1] = out[i+1]
