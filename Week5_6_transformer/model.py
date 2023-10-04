import math
import torch
import torch.nn as nn


class Attention(nn.Module):
    def __init__(self, input_sz: int, hidden_sz:int):
        super().__init__()
        self.input_sz = input_sz
        self.hidden_sz = hidden_sz

        # W_q, W_k, W_v = (len_vec , hidden)
        self.W_q = nn.Parameter(torch.Tensor(input_sz, hidden_sz))
        self.W_k = nn.Parameter(torch.Tensor(input_sz, hidden_sz))
        self.W_v = nn.Parameter(torch.Tensor(input_sz, hidden_sz))

        self.inp_linear_self_attention = nn.Linear(input_sz, input_sz)  # Input (seq_sz, len_vec) -> Output: (seq_sz, len_vec)

        self.init_weights()

    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_sz)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def softmax(self, x, axis=1):
        exp_x = torch.exp(x)
        sum_exp_x = torch.sum(exp_x, dim = axis, keepdim=True)
        softmax_x = exp_x / sum_exp_x
        return softmax_x

    def forward(self, x):
        "x.shape = (seq_sz, len_vector)"
        seq_sz, len_vec = x.size()
        x_ = self.inp_linear_self_attention(x)
        # Matmul
        Q = x_ @ self.W_q  # Q shape [seq_sz, hidden)
        K = x_ @ self.W_k  # Q shape [seq_sz, hidden)
        V = x_ @ self.W_v  # Q shape [seq_sz, hidden)
        I = Q @ K.T  # I shape [seq_sz, seq_sz]
        # Attention Mask
        for i in range(seq_sz):
            for j in range(seq_sz):
                if (i < j):
                    I[i][j] = -10 ** 10

        F = self.softmax(I / math.sqrt(len_vec)) @ V  # [seq_sz, hidden]
        return F



class Multi_attention(nn.Module):
    def __init__(self, input_sz: int, hidden_sz: int, num_heads: int): # (len_vec , hidden, num head)
        super().__init__()

        self.input_sz = input_sz
        self.hidden_sz = hidden_sz
        self.num_heads = num_heads

        self.out_linear_self_attention_layer = nn.Linear(self.num_heads * self.hidden_sz, self.input_sz)
        # Create multi attention
        self.attention_heads = nn.ModuleList([Attention(input_sz,hidden_sz) for _ in range(num_heads)])

    def forward(self,x):
        head_outputs = [head(x) for head in self.attention_heads]

        multi_head_output = torch.cat(head_outputs, dim = -1)
        output_multi_self_attention = self.out_linear_self_attention_layer(multi_head_output)
        return output_multi_self_attention


class GPT2Block(nn.Module):
    def __init__(self, input_sz: int, hidden_sz:int, vocab_size:int):
        super().__init__()
        self.input_sz = input_sz
        self.hidden_sz = hidden_sz

        self.PE_layer_norm = nn.LayerNorm(input_sz)
        self.layer_norm_transformer_block = nn.LayerNorm(input_sz)

        # Linear function
        self.linear_before_gelu = nn.Linear(input_sz, hidden_sz)
        self.linear_after_gelu = nn.Linear(hidden_sz, input_sz)

        self.init_weights()
    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_sz)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)


    def gelu(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

    def forward(self, x, init_states = None):
        "x.shape = (seq_sz, len_vector)"
        seq_sz, len_vec = x.size()
        save1_x = x
        x = self.PE_layer_norm(x)
        multi_attention = Multi_attention(len_vec, 300, 5)
        x = multi_attention(x)
        x = self.layer_norm_transformer_block(x + save1_x) # [seq_sz, len_vec]
        save2_x = x # [seq_sz, len_vec]
        x = self.linear_before_gelu(x)
        x = self.gelu(x)
        x = self.linear_after_gelu(x) + save2_x # Output transformer block [seq_sz, len_vec]

        return x


class GPT2(nn.Module):
    def __init__(self, input_sz: int, hidden_sz:int, vocab_size:int, num_block: int):
        super().__init__()
        self.input_sz = input_sz
        self.hidden_sz = hidden_sz

        self.target_layer_norm = nn.LayerNorm(input_sz)

        # Linear function

        self.target_linear = nn.Linear(input_sz, vocab_size)  # Input (seq_sz, len_vec) -> Output: (seq_sz, vocab_size)

        self.gpt2block = nn.ModuleList([GPT2Block(input_sz,input_sz*4,vocab_size) for _ in range(num_block)])

        self.init_weights()
    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_sz)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def positional_encoding(self,x):
        seq_sz, len_vec = x.size()
        encoding = torch.zeros(seq_sz, len_vec)  # Tạo tensor mới để lưu trữ kết quả

        for pos in range(seq_sz):
            for i in range(len_vec):
                if i % 2 == 0:
                    encoding[pos][i] = math.sin(pos / (1000 ** (i / len_vec)))
                else:
                    encoding[pos][i] = math.cos(pos / (1000 ** (i / len_vec)))
        return encoding

    def softmax(self, x, axis=1):
        exp_x = torch.exp(x)
        sum_exp_x = torch.sum(exp_x, dim=axis, keepdim=True)
        softmax_x = exp_x / sum_exp_x
        return softmax_x

    def forward(self, x, init_states = None):
        "x.shape = (seq_sz, len_vector)"
        seq_sz, len_vec = x.size()
        x = x + self.positional_encoding(x)

        for gpt2block in self.gpt2block:
            x = gpt2block(x)
        out = self.target_layer_norm(x)
        out = self.target_linear(out) # out [seq_sz, vocab_size]
        return out
