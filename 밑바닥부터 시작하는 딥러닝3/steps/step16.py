import numpy as np

class Variable:
    def __init__(self, data):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError('{}은(는) 지원하지 않습니다.'.format(type(data)))
        self.data = data
        self.grad = None
        self.creator = None
        self.generation = 0 # 세대 수를 기록하는 변수

    def set_creator(self, func):
        self.creator = func
        self.generation = func.generation + 1 # 세대를 기록한다(부모 세대 + 1).
    
    def cleargrad(self): # 토치의 optimizer.zerograd()와 유사한 듯
        self.grad = None

    def backward(self):
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        funcs = []
        seen_set = set()

        def add_func(f):
            if f not in seen_set: # 계산 그래프가 분기될 때 동일 함수 재방문 방지
                funcs.append(f)
                seen_set.add(f)
                funcs.sort(key=lambda x: x.generation) # 큰 세대 순으로 pop하기 위해 정렬
        
        add_func(self.creator)

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
                    add_func(x.creator) # 수정 전: funcs.append(x.creator)

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

        self.generation = max([x.generation for x in inputs])
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

x = Variable(np.array(2.0))
a = square(x) # x ** 2
y = add(square(a), square(a)) # # x ** 4 + x ** 4
y.backward()
print(y.data)
print(x.grad) # 8 * x ** 3