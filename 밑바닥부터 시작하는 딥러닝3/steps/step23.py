if '__file__' in globals(): # __file__ 이라는 전역변수가 정의되어 있는지 확인
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..')) # 현재 파일의 부모 디렉토리를 import 검색 경로(sys.path)에 추가
    # 즉 상위 폴더에 있는 모듈을 import할 수 있음

import numpy as np
from dezero import Variable

x = Variable(np.array(1.0))
y = (x + 3) ** 2
y.backward()

print(y)
print(x.grad)