import numpy as np

def im2col(input_data, filter_size, stride=1, pad=0):
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
            col[:, :, y, x] = img[:, :, y:y_max:stride, x:x_max:stride]

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N*out_h*out_w, -1)
    return col

def col2im(col, input_size, filter_size, stride=1, pad=0):
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

    img = img[:, :, pad:H + pad, pad:W + pad]
    return img

class Pooling:
    def __init__(self, pool_size, stride):
        self.pool_h, self.pool_w = pool_size
        self.stride = stride

    def forward(self, x):
        N, C, H, W = x.shape
        out_h = (H - self.pool_h) // self.stride + 1
        out_w = (W - self.pool_w) // self.stride + 1

        col = im2col(x, (self.pool_h, self.pool_w), self.stride) # (N*out_h*out_w, C*pool_h*pool_w)
        col = col.reshape(-1, self.pool_h*self.pool_w) # (N*out_h*out_w*C, self.pool_h*self.pool_w)

        arg_max = np.argmax(col, axis=1) # (N*out_h*out_w*C)
        out = np.max(col, axis=1) # (N*out_h*out_w*C,)
        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2) # (N, C, out_h, out_w)

        self.x = x
        self.arg_max = arg_max
        return out
    
    def backward(self, dout):
        dout = dout.transpose(0, 2, 3, 1) # (N, out_h, out_w, C)

        pool_size = self.pool_h*self.pool_w
        dmax = np.zeros((dout.size, pool_size)) # (N*out_h*out_c*C, self.pool_h*self.pool_w)
        dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] = dout.flatten()
        dmax = dmax.reshape(dout.shape + (pool_size,)) # (N, out_h, out_w, C, self.pool_h*self.pool_w)

        dcol = dmax.reshape(dmax.shape[0]*dmax.shape[1]*dmax.shape[2], -1) # (N*out_h*out_w, C*self.pool_h*self.pool_w)
        dx = col2im(dcol, self.x.shape, (self.pool_h, self.pool_w), self.stride)
        return dx

# dout = np.array([[1, 2],[3, 4], [5, 6]])
# arg_max = np.array([[5, 1],[2, 3],[4, 6]])
# dmax = np.zeros((dout.size, 3*3))
# dmax[np.arange(arg_max.size), arg_max.flatten()]= dout.flatten()
# print(dout.flatten())
# print(dmax)


# x = np.arange(1*1*6*6).reshape(1, 1, 6, 6)
# pool = Pooling((3, 3), stride=2)
# print(x)
# print()
# print(pool.forward(x))
# print()

# dout = np.random.randn(1, 1, 2, 2)
# print(pool.backward(dout)) 

# forward에서 max였던 위치로만 gradient 절달되고 나머지 위치의 gradient는 0. 
# maxpooling의 역전파는 gradient가 매우 희소함.


# ---
# reshape / transpose 실수 시 즉시 shape 에러
# def backward(self, dout):
#     N, C, H, W = self.x.shape
#     out_h = (H - self.pool_h) // self.stride + 1
#     out_w = (W - self.pool_w) // self.stride + 1

#     # (N, C, out_h, out_w) → (N, out_h, out_w, C)
#     dout = dout.transpose(0, 2, 3, 1)

#     pool_size = self.pool_h * self.pool_w

#     # 🔑 핵심: (N, out_h, out_w, C, pool_size)
#     dmax = np.zeros((N, out_h, out_w, C, pool_size))

#     # arg_max를 (N, out_h, out_w, C)로 복원
#     arg_max = self.arg_max.reshape(N, out_h, out_w, C)

#     # 채널 의미가 살아 있는 scatter
#     n, oh, ow, c = np.indices((N, out_h, out_w, C))
#     dmax[n, oh, ow, c, arg_max] = dout

#     # (N*out_h*out_w, C*pool_size)
#     dcol = dmax.reshape(N*out_h*out_w, C * pool_size)

#     dx = col2im(dcol, self.x.shape,
#                 (self.pool_h, self.pool_w),
#                 self.stride)
#     return dx

# print(np.indices((4, 2, 3)))