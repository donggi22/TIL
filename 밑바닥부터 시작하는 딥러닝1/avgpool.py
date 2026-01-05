import numpy as np
from conv_practice import col2im, im2col

class AveragePooling:
    def __init__(self, pool_size: tuple[int, int], stride: int=1, pad: int=0):
        self.pool_h, self.pool_w = pool_size
        self.stride = stride
        self.pad = pad

        self.x_shape = None

    def forward(self, x):
        N, C, H, W = x.shape
        self.x_shape = x.shape

        out_h = (H - self.pool_h + 2 * self.pad) // self.stride + 1
        out_w = (W - self.pool_w + 2 * self.pad) // self.stride + 1

        col = im2col(
            x,
            (self.pool_h, self.pool_w),
            self.stride,
            self.pad
        )  # (N*out_h*out_w, C*pool_h*pool_w)

        col = col.reshape(-1, self.pool_h * self.pool_w)
        out = col.mean(axis=1)

        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)
        return out

    def backward(self, dout):
        N, C, out_h, out_w = dout.shape

        pool_size_mul = self.pool_h * self.pool_w
        dout = dout.transpose(0, 2, 3, 1).reshape(-1, 1)

        dcol = np.ones((dout.shape[0], pool_size_mul)) * dout / pool_size_mul
        dcol = dcol.reshape(N * out_h * out_w, -1)

        dx = col2im(
            dcol,
            self.x_shape,
            (self.pool_h, self.pool_w),
            self.stride,
            self.pad
        )
        return dx
    
avgpool = AveragePooling((2, 2), stride=2)

x = np.arange(1, 1+1*3*6*6).reshape(1, 3, 6, 6)
print(avgpool.forward(x))

# dout = np.arange(1, 1+1*3*3*3).reshape(1, 3, 3, 3)
# print(avgpool.backward(dout))
print(avgpool.backward(avgpool.forward(x)))