import torch
a = torch.randn(2, 3, 4)
b = torch.randn(2, 4, 5)

print(torch.matmul(a,b).shape)