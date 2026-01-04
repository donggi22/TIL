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

# img = np.arange(1, 1 + 2*3*4*4).reshape(2, 3, 4, 4)
# filter_size = (3, 3)
# print(im2col(img, filter_size))

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
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x]

    img = img[:, :, pad:H + pad, pad:W + pad]
    return img

# input_size = (1, 3, 4, 4)
# filter_size = (3, 3)
# col = np.arange(1, 1 + 1*4*4*3*3*3).reshape(1*4*4, 3*3*3)
# stride = 1
# pad = 1
# print(col2im(col, input_size, filter_size, stride, pad))

class Convolution:
    def __init__(self, in_channel, out_channel, filter_size, stride=1, pad=0):
        filter_h, filter_w = filter_size
        self.W = np.random.randn(out_channel, in_channel, filter_h, filter_w)
        self.b = np.random.randn(out_channel)
        self.stride = stride
        self.pad = pad

        # 중간 데이터 (backward 시 사용)
        self.x = None
        self.col = None
        self.col_W = None

        # 가중치와 편향 매개변수의 기울기
        self.dW = None
        self.db = None

    def forward(self, x):
        FN, C, FH, FW = self.W.shape
        N, C, H, W = x.shape
        out_h = (H - FH + 2*self.pad) // self.stride + 1
        out_w = (W - FW + 2*self.pad) // self.stride + 1

        col = im2col(x, (FH, FW), self.stride, self.pad) # (N*out_h*out_w, C*FH*FW)
        col_W = self.W.reshape(FN, -1).T # (C*FH*FW, FN)

        out = np.dot(col, col_W) + self.b # (N*out_h*out_w, FN)
        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2) # (N, FN, out_h, out_w)

        self.x = x
        self.col = col
        self.col_W = col_W
        return out

    def backward(self, dout):
        FN, C, FH, FW = self.W.shape
        dout = dout.transpose(0, 2, 3, 1).reshape(-1, FN) # (N*out_h*out_w, FN)

        self.db = np.sum(dout, axis=0) # (FN,)
        self.dW = np.dot(self.col.T, dout) # (C*FH*FW, FN)
        self.dW = self.dW.T.reshape(FN, C, FH, FW) # (FN, C, FH, FW)
        dcol = np.dot(dout, self.col_W.T) # (N*out_h*out_w, C*FH*FW)

        dx = col2im(dcol, self.x.shape, (FH, FW), self.stride, self.pad)
        return dx
    
# conv = Convolution(3, 1, (3, 3))

# x = np.random.randn(3, 3, 28, 28)
# print(conv.forward(x).shape)
# print(conv.forward(x)[0][0].shape)
# img1 = conv.forward(x)[0][0]
# img2 = conv.forward(x)[1][0]
# img3 = conv.forward(x)[2][0]

# import matplotlib.pyplot as plt

# fig, axs = plt.subplots(1, 3)
# axs[0].imshow(img1)
# axs[1].imshow(img2)
# axs[2].imshow(img3)
# plt.show()