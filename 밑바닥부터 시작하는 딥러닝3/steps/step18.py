import weakref
import numpy as np
import contextlib

class Config:
    enable_backprop = True

# contextlib.contextmanager 데코레이터를 달면
# with 진입 시 next(generator)
# with 종료 시 정상: next(generator), 예외: generator.throw(...)
@contextlib.contextmanager
def using_config(name, value):
    old_value = getattr(Config, name)
    setattr(Config, name, value) # Config의 name 속성을 새로운 값(value)로 설정
    try:
        yield
    finally:
        setattr(Config, name, old_value) # Config의 name 속성을 원래 값(old_value)으로 복원

def no_grad(): # using_config('enable_backprop', False) 간편화
    return using_config('enable_backprop', False)

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

    def backward(self, retain_grad=False):
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
            # 수정 전: gys = [output.grad for output in f.outputs]
            gys = [output().grad for output in f.outputs]
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

            if not retain_grad:
                for y in f.outputs:
                    y().grad = None # y는 약한 참조(weakref)라서 ()로 호출

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
        
        if Config.enable_backprop: # 역전파 활성
            self.generation = max([x.generation for x in inputs])
            for output in outputs:
                output.set_creator(self)
            self.inputs = inputs
            self.outputs = [weakref.ref(output) for output in outputs]

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

x0 = Variable(np.array(1.0))
x1 = Variable(np.array(1.0))
t = add(x0, x1)
y = add(x0, t)
y.backward()

print(y.grad, t.grad)
print(x0.grad, x1.grad)

Config.enable_backprop = True
x = Variable(np.ones((100, 100, 100)))
y = square(square(square(x)))
y.backward()
# print(y.data.shape)

Config.enable_backprop = False
x = Variable(np.ones((100, 100, 100)))
y = square(square(square(x)))
# print(y.data.shape)

with using_config('enable_backprop', False):
    x = Variable(np.array(2.0))
    y = square(x)

with no_grad():
    x = Variable(np.array(2.0))
    y = square(x)