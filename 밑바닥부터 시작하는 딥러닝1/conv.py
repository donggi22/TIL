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

def col2im(input_size, filter_size, col, stride=1, pad=0):
    N, C, H, W = input_size
    filter_h, filter_w = filter_size
    
    out_h = (H - filter_h + 2*pad) // stride + 1
    out_w = (W - filter_w + 2*pad) // stride + 1

    col = col.transpose(0, 3, 4, 5, 1, 2)
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
# col = np.arange(1, 1 + 1*4*4*3*4*4).reshape(1, 4, 4, 3, 4, 4)
# stride = 1
# pad = 1
# print(col2im(input_size, filter_size, col, stride, pad))