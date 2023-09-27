import torch
from torch import nn
criteria = nn.CrossEntropyLoss()
input = torch.tensor([[3.4, 1.5,0.4, 0.10]],dtype=torch.float)
target = torch.tensor([1], dtype=torch.long)
print(criteria(input, target))
