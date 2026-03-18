if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import dezero.functions as F
from dezero import Variable

x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
# y = F.sin(x)
c = Variable(np.array([[10, 20, 30], [40, 50, 60]]))
t = x + c
y = F.sum(t) # sum 함수 step39 에서 구현 예정.
# print(y)

y.backward(retain_grad=True)
print(y.grad)
print(t.grad)
print(x.grad)
print(c.grad)