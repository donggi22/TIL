if '__file__' in globals(): # __file__ 이라는 전역변수가 정의되어 있는지 확인
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..')) # 현재 파일의 부모 디렉토리를 import 검색 경로(sys.path)에 추가
    # 즉 상위 폴더에 있는 모듈을 import할 수 있음

import numpy as np
from dezero import Variable

def rosenbrock(x0, x1):
    y = 100 * (x1 - x0 ** 2) ** 2 + (1 - x0) ** 2
    return y

x0 = Variable(np.array(0.0))
x1 = Variable(np.array(2.0))
lr = 0.001 # 학습률
iters = 10000 # 반복 횟수

for i in range(iters):
    print(x0, x1)

    y = rosenbrock(x0, x1)
    
    x0.cleargrad()
    x1.cleargrad()
    y.backward()

    x0.data -= lr * x0.grad
    x1.data -= lr * x1.grad