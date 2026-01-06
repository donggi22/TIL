import numpy as np

class MatMul:
    def __init__(self, W):
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.x = None

    def forward(self, x):
        W, = self.params
        out = np.matmul(x, W)
        self.x = x
        return out
    
    def backward(self, dout):
        W, = self.params
        dx = np.matmul(dout, W.T)
        dW = np.matmul(self.x.T, dout)
        self.grads[0][...] = dW
        return dx
    
class Affine(MatMul):
    def __init__(self, W, b):
        super().__init__(W)

        self.params.append(b)
        self.grads.append(np.zeros_like(b))

    def forward(self, x):
        out = super().forward(x)
        
        b = self.params[1]
        out += b
        return out

    def backward(self, dout):
        dx = super().backward(dout)

        db = np.sum(dout, axis=0)
        self.grads[1][...] = db
        return dx

# class Affine:
#     def __init__(self, W, b):
#         self.params = [W, b]
#         self.grads = [np.zeros_like(W), np.zeros_like(b)]
#         self.x = None
        
#     def forward(self, x):
#         W, b = self.params
#         out = np.matmul(x, W) + b
#         self.x = x
#         return out
    
#     def backward(self, dout):
#         W, b = self.params
#         dx = np.matmul(dout, W.T)
#         db = dout.sum(axis=0)
#         dW = np.matmul(self.x.T, dout)
#         self.grads[0][...] = dW # copy
#         self.grads[1][...] = db # copy
#         return dx

class ReLU:
    def __init__(self):
        self.params = []
        self.grads = []
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0) # 0과 음수 부분 True
        out = x.copy()
        out[self.mask] = 0
        return out
    
    def backward(self, dout):
        dx = dout.copy()
        dx[self.mask] = 0
        return dx
    
class Sigmoid:
    def __init__(self):
        self.params = []
        self.grads = []
        self.out = None

    def forward(self, x):
        self.out = 1/(1+np.exp(-x))
        return self.out
    
    def backward(self, dout):
        dx = dout * self.out * (1. - self.out)
        return dx
    
class Softmax