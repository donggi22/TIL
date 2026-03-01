import numpy as np

class Variable:
    def __init__(self, data):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError('{}은(는) 지원하지 않습니다.'.format(type(data)))
        self.data = data
        self.grad = None
        self.creator = None

    def set_creator(self, func):
        self.creator = func
    
    def cleargrad(self): # 토치의 optimizer.zerograd()와 유사한 듯
        self.grad = None

    def backward(self):
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        funcs = [self.creator] 
        while funcs:
            f = funcs.pop() # 함수를 가져온다.
            gys = [output.grad for output in f.outputs]
            gxs = f.backward(*gys)
            if not isinstance(gxs, tuple):
                gxs = (gxs,)

            for x, gx in zip(f.inputs, gxs):
                if x.grad is None:
                    x.grad = gx
                else: # 누적 추가
                    x.grad = x.grad + gx

                if x.creator is not None:
                    funcs.append(x.creator) # 하나 앞의 함수를 리스트에 추가한다.

def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x

class Function:
    def __call__(self, *inputs): # 입출력이 다변수일 때
        xs = [x.data for x in inputs]
        ys = self.forward(*xs) # unpack
        if not isinstance(ys, tuple): # 튜플이 아닌 경우 튜플로
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys] 

        for output in outputs:
            output.set_creator(self)
        self.inputs = inputs
        self.outputs = outputs
        return outputs if len(outputs) > 1 else outputs[0]
        
    def forward(self, x):
        raise NotImplementedError()

    def backward(self, gy):
        raise NotImplementedError()

class Add(Function):
    def forward(self, x0, x1):
        y = x0 + x1
        return y
    
    def backward(self, gy):
        return gy, gy

def add(x0, x1):
    return Add()(x0, x1)

class Square(Function):
    def forward(self, x):
        y = x ** 2
        return y
    
    def backward(self, gy):
        x = self.inputs[0].data
        gx = 2 * x * gy
        return gx
    
def square(x):
    f = Square()
    return f(x)

x = Variable(np.array(3.0))
y = add(x, x)

# 첫 번째 계산
print('y:', y.data)
y.backward()
print('x:', x.grad)

# 두 번째 계산(같은 x를 사용하여 다른 계산을 수행)
x.cleargrad() # 미분값 초기화
# x = Variable(np.array(3.0)) # 또는 x.cleargrad()
y = add(add(x, x), x)
y.backward()
print('x:', x.grad)