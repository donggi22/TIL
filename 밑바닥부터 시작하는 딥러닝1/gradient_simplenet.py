import numpy as np
from functions import softmax, cross_entropy_error
from gradient import numerical_gradient

class SimpleNet:
    def __init__(self):
        self.W = np.random.randn(2, 3) # 정규분포로 초기화
    
    def predict(self, x):
        return np.dot(x, self.W)
    
    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)
        return loss
    
net = SimpleNet()
# print(net.W) # shape: (2, 3), size: 6
x = np.array([0.6, 0.9])
# print(x.size) # 2
pred = net.predict(x) # shape: (3,)
# print(pred)
# print(np.argmax(pred).shape) # shape: ()
t = np.array([0, 0, 1]) # 정답 레이블
# print(t.shape) # shape: (3,)
# print(net.loss(x, t)) # shape: ()

def f(W):
    return net.loss(x, t)

dW = numerical_gradient(f, net.W)
print(dW)