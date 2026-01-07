import numpy as np
from functions import sigmoid, softmax, binary_cross_entropy, cross_entropy_error

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
    
class Affine:
    def __init__(self, W, b):
        self.matmul = MatMul(W)

        self.params = [W, b]
        self.grads = [self.matmul.grads[0], np.zeros_like(b)]

    def forward(self, x):
        out = self.matmul.forward(x) # x dot W
        b = self.params[1]
        out += b
        return out

    def backward(self, dout):
        dx = self.matmul.backward(dout) # dx, dW
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
        self.out = sigmoid(x)
        return self.out
    
    def backward(self, dout):
        dx = dout * self.out * (1. - self.out)
        return dx
    
class Softmax:
    def __init__(self):
        self.params, self.grads = [], []
        self.out = None
    
    def forward(self, x):
        self.out = softmax(x)
        return self.out
    
    def backward(self, dout):
        # 야코비안의 대각 성분(i==j)과 비대각 성분(i!=j)을 구분할 것
        # out_i*(1 - out_i) 또는 -out_i*out_j

        # dx_i = ∂L/∂x_i
        # = out_i*(1-out_i)*dout_i + sum_{j≠i}(-out_i*out_j*dout_j)
        # (각 i에 대해 스칼라, 전체 dx는 벡터)

        # dx_i = sum_j{out_i*(delta_ij-out_j)*dout_j} = sum_j(out_i*delta_ij*dout_j) - sum_j(out_i*out_j*dout_j)

        # dout: 벡터, self.out: 벡터
        dx = self.out * dout # dx_i = sum_j(out_i * delta_ij * dout_j) (i!=j 성분 값 0이므로) => dx_i = out_i*dout_i로 요약
        
        if dout.ndim == 1:
            sumdx = np.sum(dx)
        elif dout.ndim == 2:
            sumdx = np.sum(dx, axis=1, keepdims=True) # 각 dx_i에서 동일하게 빼지는 공통 합 항
        dx -= self.out*sumdx # dx_i = out_i*dout_i - out_i*sum(out_j * dout_j)
        return dx
    
class SigmoidWithLoss:
    def __init__(self):
        self.params, self.grads = [], []
        self.loss = None
        self.t = None
        self.y = None

    def forward(self, x, t):
        self.t = t # 정수 라벨
        self.y = sigmoid(x)

        # # CEE(y, t) = -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size
        # self.loss = cross_entropy_error(np.c_[1 - self.y, self.y], self. t) # 2-class CEE로 BCE와 동일한 결과 얻기 위해 concat => (N, 2)

        self.loss = binary_cross_entropy(self.y, self.t)
        return self.loss
    
    def backward(self, dout=1):
        batch_size = self.t.shape[0]

        # self.t를 (N,1)로 맞춰서 뺄셈 정확히 수행
        t_reshaped = self.t.reshape(-1, 1) if self.t.ndim == 1 else self.t
        dx = (self.y - t_reshaped) * dout / batch_size
        return dx
    
class SoftmaxWithLoss:
    def __init__(self):
        self.params, self.grads = [], []
        self.t = None
        self.y = None

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)

        # 정답 레이블이 원핫 벡터일 경우 정답의 인덱스로 변환
        if self.t.size == self.y.size:
            self.t = self.t.argmax(axis=1)
        loss = cross_entropy_error(self.y, self.t)
        return loss
    
    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        
        # 원핫 벡터일 경우: self.y.shape == self.t.shape: (N, C)
        # dx = (self.y - self.t) * dout / batch_size (Jacobian 전체 × CE gradient)
        
        # 정수 라벨을 one-hot처럼 효과내서 메모리 절약, self.t.shape: (N,), self.y.shape: (N, C)
        dx = self.y.copy()
        dx[np.arange(batch_size), self.t] -= 1
        dx *= dout / batch_size
        return dx