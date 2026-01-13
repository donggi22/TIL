from np import *
from layers import Embedding, Affine, SoftmaxWithLoss
from functions import softmax, sigmoid

class RNN:
    def __init__(self, Wx, Wh, b):
        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        self.cache = None

    def forward(self, x, h_prev):
        Wx, Wh, b = self.params
        t = np.matmul(h_prev, Wh) + np.matmul(x, Wx) + b
        h_next = np.tanh(t)

        self.cache = x, h_prev, h_next
        return h_next
    
    def backward(self, dh_next):
        x, h_prev, h_next = self.cache
        Wx, Wh, b = self.params
        
        dt = dh_next*(1 - h_next**2)
        dh_prev = np.matmul(dt, Wh.T)
        dWh = np.matmul(h_prev.T, dt)
        dx = np.matmul(dt, Wx.T)
        dWx = np.matmul(x.T, dt)
        db = np.sum(dt, axis=0) # (hidden_dim,), axis=1 -> (batch_size,)

        self.grads[0][...] = dWx
        self.grads[1][...] = dWh
        self.grads[2][...] = db
        return dx, dh_prev
    
class TimeRNN:
    def __init__(self, Wx, Wh, b, stateful=False):
        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        self.layers = None

        self.h, self.dh = None, None
        self.stateful = stateful

    def set_state(self, h):
        self.h = h
    
    def reset_state(self):
        self.h = None

    def forward(self, xs):
        Wx, Wh, b = self.params
        N, T, D = xs.shape
        D, H = Wx.shape

        self.layers = []
        hs = np.empty((N, T, H), dtype='f')

        if not self.stateful or self.h is None:
            self.h = np.zeros((N, H), dtype='f') # 영행렬로 초기화

        for t in range(T):
            layer = RNN(*self.params)
            self.h = layer.forward(xs[:, t, :], self.h)
            hs[:, t, :] = self.h # x_t와 h_t는 각 샘플 데이터를 행 방향에 저장
            self.layers.append(layer)
        return hs
    
    def backward(self, dhs):
        Wx, Wh, b = self.params
        N, T, H = dhs.shape
        D, H = Wx.shape

        dxs = np.empty((N, T, D), dtype='f')
        dh = 0 # 마지막 RNN계층의 h_next는 없으므로 0
        # grads = [0, 0, 0]
        grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)] # 로컬 누적 버퍼
        for t in reversed(range(T)):
            layer = self.layers[t]
            dx, dh = layer.backward(dhs[:, t, :] + dh) # 합산된 기울기, repeat(특수한 분기)처럼 더하기
            dxs[:, t, :] = dx # x_t와 h_t는 각 샘플 데이터를 행 방향에 저장

            for i, grad in enumerate(layer.grads):
                grads[i] += grad # time 방향 누적
        
        for i, grad in enumerate(grads):
            self.grads[i][...] = grad # 한 번에 반영, torch의 optimizer.zero_grad()후 backward하는 것과 같은 효과
        self.dh = dh # seq2seq에서 필요
        return dxs
    
class TimeEmbedding:
    def __init__(self, W):
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.layers = None
        self.W = W

    def forward(self, xs):
        N, T = xs.shape
        V, D = self.W.shape

        out = np.empty((N, T, D), dtype='f')
        self.layers = []

        for t in range(T):
            layer = Embedding(self.W)
            out[:, t, :] = layer.forward(xs[:, t])
            self.layers.append(layer)
        return out
    
    def backward(self, dout):
        N, T, D = dout.shape

        grad = np.zeros_like(self.W)
        for t in range(T):
            layer = self.layers[t]
            layer.backward(dout[:, t, :])
            grad += layer.grads[0]
        self.grads[0][...] = grad
        return None

class TimeAffine:
    def __init__(self, W, b):
        self.params = [W, b]
        self.grads = [np.zeros_like(W), np.zeros_like(b)]
        self.x = None

    def forward(self, x):
        N, T, D = x.shape
        W, b = self.params

        rx = x.reshape(N*T, -1)
        out = np.dot(rx, W) + b # 모든 시점과 배치를 한 번에 계산 (벡터화로 성능 최적화)

        # # reshape 안 할 경우 for loop로 직접 계산 (Python 레벨 반복문이라 느림)
        # out = np.zeros((N, T, M))
        # for t in range(T):
        #     out[:, t, :] = np.dot(x[:, t, :], W) + b
        self.x = x
        return out.reshape(N, T, -1)
    
    def backward(self, dout):
        x = self.x
        N, T, D = x.shape
        W, b = self.params

        dout = dout.reshape(N*T, -1)
        rx = x.reshape(N*T, -1)

        db = np.sum(dout, axis=0)
        dW = np.dot(rx.T, dout)
        dx = np.dot(dout, W.T)
        dx = dx.reshape(*x.shape)

        self.grads[0][...] = dW
        self.grads[1][...] = db
        return dx
    
class TimeSoftmaxWithLoss:
    def __init__(self):
        self.params, self.grads = [], []
        self.cache = None
        self.ignore_label = -1

    def forward(self, xs, ts):
        N, T, V = xs.shape

        if ts.ndim == 3: # 정답 레이블이 원핫 벡터인 경우
            ts = ts.argmax(axis=2)
        
        mask = (ts != self.ignore_label)

        # 배치용과 시계열용을 정리(reshape)
        xs = xs.reshape(N*T, V)
        ts = ts.reshape(N*T)
        mask = mask.reshape(N*T)

        ys = softmax(xs)
        ls = np.log(ys[np.arange(N*T), ts])
        ls *= mask # ignore_label에 해당하는 데이터는 손실을 0으로 설정
        loss = -np.sum(ls)
        loss /= mask.sum()

        self.cache = (ts, ys, mask, (N, T, V))
        return loss
    
    def backward(self, dout=1):
        ts, ys, mask, (N, T, V) = self.cache

        dx = ys.copy() # softmax 출력 확률을 gradient 계산용으로 재사용
        dx[np.arange(N*T), ts] -= 1 # target 위치에 1을 빼서 softmax + cross entropy의 미분 결과를 구현
        dx *= dout # 체인룰
        dx /= mask.sum() # 유효 토큰 수로 나누어 loss 평균과 일관되게 맞춤
        dx *= mask[:, np.newaxis] # ignore_label에 해당하는 데이터는 기울기를 0으로 설정

        dx = dx.reshape((N, T, V))
        return dx
    
class LSTM:
    def __init__(self, Wx, Wh, b):
        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        self.cache = None
    
    def forward(self, x, h_prev, c_prev):
        Wx, Wh, b = self.params
        N, H = h_prev.shape

        A = np.matmul(x, Wx) + np.matmul(h_prev, Wh) + b

        # slice
        f = A[:, :H]
        g = A[:, H:2*H]
        i = A[:, 2*H:3*H]
        o = A[:, 3*H:]

        f = sigmoid(f)
        g = np.tanh(g)
        i = sigmoid(i)
        o = sigmoid(o)

        c_next = f * c_prev + g * i
        h_next = o * np.tanh(c_next)
        
        self.cache = (x, h_prev, c_prev, i, f, g, o, c_next)
        return h_next, c_next
    
    def backward(self, dh_next, dc_next):
        Wx, Wh, b = self.params
        x, h_prev, c_prev, i, f, g, o, c_next = self.cache

        tanh_c_next = np.tanh(c_next)

        '''
        c_next2 = f_next * "c_next" + g_next * i_next
        h_next = o * np.tanh("c_next")
        ds = ∂𝐿/∂𝑐_t
        d𝑐_t = dc_next = ∂𝐿/∂𝑐_{t+1}*∂𝑐_{t+1}/∂𝑐_t
        dh_t = dh_next = ∂𝐿/∂h_t
        ∂h_t/∂𝑐_t = o * (1 - tan_c_next**2)
        '''
        ds = dc_next + dh_next * o * (1 - tanh_c_next**2) # ∂𝐿/∂𝑐_t = ∂𝐿/∂𝑐_{t+1}*∂c_{t+1}/∂𝑐_t + ∂𝐿/∂h_t*∂h_t/∂𝑐_t
                                                            #                 (시간 경로)             (출력 경로)

        # c_next = f * c_prev + g * i
        dc_prev = ds * f

        # c_next = f * c_prev + g * i
        # h_next = o * tanh_c_next
        di = ds * g
        df = ds * c_prev
        do = dh_next * tanh_c_next
        dg = ds * i

        di *= i * (1 - i)
        df *= f * (1 - f)
        do *= o * (1 - o)
        dg *= (1 - g**2)

        dA = np.hstack((df, dg, di, do)) # slice한 4개의 기울기 결합

        # A = np.matmul(x, Wx) + np.matmul(h_prev, Wh) + b
        dWh = np.matmul(h_prev.T, dA) # (H, 4H)
        dWx = np.matmul(x.T, dA) # (D, 4H)
        db = np.sum(dA, axis=0) # (4H,)

        self.grads[0][...] = dWx
        self.grads[1][...] = dWh
        self.grads[2][...] = db

        dx = np.matmul(dA, Wx.T)
        dh_prev = np.matmul(dA, Wh.T)
        return dx, dh_prev, dc_prev
    
class TimeLSTM:
    def __init__(self, Wx, Wh, b, stateful=False):
        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        self.layers = None
        self.h, self.c = None, None
        self.dh = None
        self.stateful = stateful

    def forward(self, xs):
        Wx, Wh, b = self.params
        N, T, D = xs.shape
        H = Wh.shape[0]
        
        self.layers = []
        hs = np.empty((N, T, H), dtype='f')

        if not self.stateful or self.h is None: # 상태 초기화 지정
            self.h = np.zeros((N, H), dtype='f')
        if not self.stateful or self.c is None:
            self.c = np.zeros((N, H), dtype='f')

        for t in range(T):
            layer = LSTM(*self.params)
            self.h, self.c = layer.forward(xs[:, t, :], self.h, self.c)
            hs[:, t, :] = self.h
            
            self.layers.append(layer)
        return hs
        
    def backward(self, dhs):
        Wx, Wh, b = self.params
        N, T, H = dhs.shape
        D = Wx.shape[0]

        dxs = np.empty((N, T, D), dtype='f')
        dh, dc = np.zeros_like(self.h), np.zeros_like(self.c)

        grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        for t in reversed(range(T)):
            layer = self.layers[t]
            dx, dh, dc = layer.backward(dhs[:, t, :] + dh, dc)
            dxs[:, t, :] = dx
            for i, grad in enumerate(layer.grads):
                grads[i] += grad

        for i, grad in enumerate(grads):
            self.grads[i][...] = grad
        self.dh = dh # seq2seq에서 필요
        return dxs
    
    def set_state(self, h, c=None):
        self.h, self.c = h, c
    
    def reset_state(self):
        self.h, self.c = None, None