import numpy as np
from numpy.typing import NDArray
def im2col(input_data: NDArray, filter_size: tuple[int, int], stride: int=1, pad: int=0) -> NDArray:
    N, C, H, W = input_data.shape
    filter_h, filter_w = filter_size

    out_h = (H - filter_h + 2*pad) // stride + 1
    out_w = (W - filter_w + 2*pad) // stride + 1

    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

    img = np.pad(input_data, ((0, 0), (0, 0), (pad, pad), (pad, pad)), 'constant')

    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            col[:, :, y, x, ...] = img[:, :, y:y_max:stride, x:x_max:stride]
    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N*out_h*out_w, -1)
    return col

def col2im(col: NDArray, input_size: tuple[int, int, int, int], filter_size: tuple[int, int], stride: int=1, pad: int=0) -> NDArray:
    N, C, H, W = input_size
    filter_h, filter_w = filter_size

    out_h = (H - filter_h + 2*pad) // stride + 1
    out_w = (W - filter_w + 2*pad) // stride + 1

    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)

    img = np.zeros((N, C, H + 2*pad + stride - 1, W + 2*pad + stride - 1))

    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, ...]

    img = img[:, :, pad:H + pad, pad: W + pad]
    return img

class Convolution:
    def __init__(self, W: NDArray, b: NDArray, stride: int=1, pad: int=0):
        self.W = W # (FN, C, filter_h, filter_w)
        self.b = b # (FN,)
        self.stride = stride
        self.pad = pad

        self.col = None
        self.col_W = None
        self.x = None

        self.dW = None
        self.db = None

    def forward(self, x: NDArray) -> NDArray:
        N, C, H, W = x.shape
        FN, C, filter_h, filter_w = self.W.shape

        out_h = (H - filter_h + 2*self.pad) // self.stride + 1
        out_w = (W - filter_w + 2*self.pad) // self.stride + 1

        col = im2col(x, (filter_h, filter_w),self.stride, self.pad) # (N*out_h*out_w, C*filter_h*filter_w)
        col_W = self.W.reshape(FN, -1).T # (C*filter_h*filter_w, FN)

        out = np.dot(col, col_W) + self.b # (N*out_h*out_w, FN)
        out = out.reshape(N, out_h, out_w, FN).transpose(0, 3, 1, 2) # (N, FN, out_h, out_w)

        self.col = col
        self.col_W = col_W
        self.x = x
        
        return out
    
    def backward(self, dout: NDArray) -> NDArray:
        N, FN, out_h, out_w = dout.shape
        FN, C, filter_h, filter_w = self.W.shape
        dout = dout.transpose(0, 2, 3, 1).reshape(N*out_h*out_w, -1) # (N*out_h*out_w, FN)
        
        dW = np.dot(self.col.T, dout) # (C*filter_h*filter_w, FN)
        self.dW = dW.reshape(FN, C, filter_h, filter_w) # (FN, C, filter_h, filter_w)
        self.db = dout.sum(axis=0) # (FN,)

        dcol = np.dot(dout, self.col_W.T) # (N*out_h*out_w, C*filter_h*filter_w)
        dx = col2im(dcol, self.x.shape, (filter_h, filter_w), self.stride, self.pad) # (N, C, H, W)
        return dx
    
# W = np.random.randn(1, 3, 3, 3) # (FN, C, FH, FW)
# b = np.random.randn(1)
# conv = Convolution(W, b, stride=2, pad=1)
# # print(conv.W.shape, conv.b.shape)
# img = np.arange(1, 1 + 2*3*12*12).reshape(2, 3, 12, 12) # (N, C, H, W)
# print(conv.forward(img).shape) # (N, FN, out_h, out_w)

# dout = np.random.randn(2, 1, 6, 6) # (N, FN, out_h, out_w)
# print(conv.backward(dout).shape) # (FN, C, FH, FW)

class MaxPooling2d:
    def __init__(self, pool_size:tuple[int, int], stride:int=2, pad:int=0):
        self.pool_h, self.pool_w = pool_size
        self.stride = stride
        self.pad = pad

        self.argmax= None
        self.max= None
        self.col = None
        self.x = None

    def forward(self, x):
        N, C, H, W = x.shape
        out_h = (H - self.pool_h) // self.stride + 1
        out_w = (W - self.pool_w) // self.stride + 1
        col = im2col(x, (self.pool_h, self.pool_w), self.stride, self.pad) # (N*out_h*out_w, C*pool_h*pool*w)
        self.col = col.reshape(-1, self.pool_h*self.pool_w) # (N*out_h*out_w*C, pool_h*pool*w)

        self.argmax = np.argmax(self.col, axis=1) # (N*out_h*out_w*C,)
        self.max = np.max(self.col, axis=1) # (N*out_h*out_w*C,)
        out = self.max.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2) # (N, C, out_h, out_w)

        self.x = x
        return out
    
    def backward(self, dout):
        pool_size_mul = self.pool_h * self.pool_w
        
        dout = dout.transpose(0, 2, 3, 1) # (N, out_h, out_w, C)

        dmax = np.zeros((dout.size, pool_size_mul)) # (N*out_h*out_w*C, pool_h*pool_w)
        dmax[np.arange(self.max.size), self.argmax] = dout.flatten() # (N*out_h*out_w*C, pool_h*pool_w)

        dcol = dmax.reshape(dout.shape[0]*dout.shape[1]*dout.shape[2], -1) # (N*out_h*out_w, C*pool_h*pool_w)

        dx = col2im(dcol, self.x.shape, (self.pool_h, self.pool_w), self.stride, self.pad) # (N, C, H, W)
        return dx


# pool = MaxPooling2d((2, 2), stride=2)

# x = np.arange(1, 1+1*3*4*4).reshape(1, 3, 4, 4)
# print(x.shape)
# print(pool.forward(x).shape)

# dout = np.random.randn(1, 3, 2, 2)
# print(pool.backward(dout))