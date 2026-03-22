if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from dezero import Variable
import dezero.functions as F

x = Variable(np.array([1, 2, 3, 4, 5, 6]))
y = F.sum(x)
y.backward()
# print(y)
# print(x.grad)

x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
y = F.sum(x)
y.backward()
# print(y)
# print(x.grad)

x = np.array([[1, 2, 3], [4, 5, 6]])
y = np.sum(x, axis=0)
# print(y)
# print(x.shape, ' -> ', y.shape)

x = np.array([[1, 2, 3], [4, 5, 6]])
y = np.sum(x, keepdims=True) # keepdims: 입력과 출력 차원을 유질하지 정하는 플래그
# print(y)
# print(x.shape, ' -> ', y.shape)

x = np.array([[1, 2, 3], [4, 5, 6]])
y = np.sum(x, keepdims=False) # keepdims: 입력과 출력 차원을 유질하지 정하는 플래그
# print(y)
# print(x.shape, ' -> ', y.shape)

# value = Variable(x).sum(axis=None, keepdims=False)
# print(value)

x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
y = F.sum(x, axis=0)
y.backward()
print(y)
print(x.grad)

x = Variable(np.random.randn(2, 3, 4, 5))
y = x.sum(keepdims=True)
print(y.shape)