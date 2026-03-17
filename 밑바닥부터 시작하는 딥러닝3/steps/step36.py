if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from dezero import Variable

x = Variable(np.array(2.0))
y = x ** 2
y.backward(create_graph=True)
gx = x.grad
x.cleargrad()

z = gx ** 3 + y # gx = x.grad는 단순한 변수가 아니라 계산 그래프(식). 따라서 x.grad의 계산 그래프에 대해 추가로 역전파 가능
z.backward()
print(x.grad)