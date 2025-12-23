import numpy as np
import torch

# x = 3
x = torch.tensor(3.0, requires_grad=True)
w = torch.tensor(1.0, requires_grad=True)
z = w * x
out = z ** 2
out.backward()
# print('out을 w로 미분한 값', w.grad)
# print('out을 x로 미분한 값', x.grad)

data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# print(x.size())
# print(x.ndimension())
# print(x.shape)

