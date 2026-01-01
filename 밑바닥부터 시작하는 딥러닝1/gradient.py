import numpy as np

# def numerical_gradient(f, x):
#     h = 1e-4
#     grad = np.zeros_like(x)
    
#     if x.ndim == 1:
#         for idx in range(x.size):
#             tmp_val = x[idx]

#             # f(x+h)
#             x[idx] = tmp_val + h
#             fxh1 = f(x)

#             # f(x-h)
#             x[idx] = tmp_val - h
#             fxh2 = f(x)

#             grad = (fxh1 - fxh2) / (2*h)
#         return grad


#     elif x.ndim == 2:
#         for idx0 in range(x.shape[0]):
#             for idx1 in range(x.shape[1]):
#                 tmp_val = x[idx0, idx1]
            
#                 # f(x+h)
#                 x[idx0, idx1] = tmp_val + h
#                 fxh1 = f(x)
                
#                 # f(x-h)
#                 x[idx0, idx1] = tmp_val -h
#                 fxh2 = f(x)
                
#                 grad[idx0, idx1] = (fxh1 - fxh2) / (2*h)
#                 x[idx0, idx1] = tmp_val # 값 복원
#         return grad

def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)

    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index

        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x)

        x[idx] = float(tmp_val) - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2 * h)
        x[idx] = tmp_val

        it.iternext()

    return grad