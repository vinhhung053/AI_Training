import torch
input = torch.randn(3, 5, requires_grad=True)
print(input.shape)
target = torch.empty(3, dtype=torch.long).random_(5)
print(target.shape)