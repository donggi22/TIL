import numpy as np

class Variable:
    def __init__(self, data):
        self.data = data
        self.grad = None
        self.creator = None

    def set_creator(self, func):
        self.creator = func
    
    def backward(self):
        f = self.creator # 1. 함수를 가져온다.
        if f is not None:
            x = f.input # 2. 함수의 입력을 가져온다.
            x.grad = f.backward(self.grad) # 3. 함수의 backward 메서드를 호출한다.
            x.backward() # 하나 앞 변수의 backward 메서드를 호출한다(재귀).

class Function:
    def __call__(self, input):
        x = input.data
        y = self.forward(x)
        output = Variable(y)
        output.set_creator(self) # 출력 변수에 창조를 설정한다. 여기서 self는 메서드를 호출하는 객체(함수 클래스, func에 대응되는 객체)
        self.input = input
        self.output = output # 출력도 저장한다.
        return output
    
    def forward(self, x):
        raise NotImplementedError()

    def backward(self, gy):
        raise NotImplementedError()


class Square(Function):
    def forward(self, x):
        y = x ** 2
        return y

    def backward(self, gy):
        x = self.input.data
        gx = 2 * x * gy
        return gx


class Exp(Function):
    def forward(self, x):
        y = np.exp(x)
        return y

    def backward(self, gy):
        x = self.input.data
        gx = np.exp(x) * gy
        return gx

A = Square()
B = Exp()
C = Square()

x = Variable(np.array(0.5))
a = A(x)
b = B(a)
y = C(b)

y.grad = np.array(1.0)
y.backward() # 변수의 backward(loss값의 backward같은 느낌)
print(x.grad)