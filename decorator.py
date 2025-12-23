# 데코레이터 기억 되살리기
import time

def time_measure(func):
    def act(*args, **kwargs): # 여기서 인자의 갯수를 유동적으로 만들어줌.
        start = time.time()
        f = func(*args, **kwargs)
        print(time.time()-start)
        return f
    return act

import math

def is_prime(number):
    count = 0
    for i in range(1, int(math.sqrt(number))+1):
        if number % i == 0:
            count += 1
    return count == 1

# 오일러-유클리드 정리에 의해, 짝수 완전수와 메르센 소수는 일대일 대응 관계임.
@time_measure
def perfect_under(number): 
    return {p: (2**p - 1) * 2**(p - 1) for p in range(2, number+1) if is_prime(2**p - 1)} # 이 때 2**p - 1이 소수일 때 메르센 소수라 함.

print("@ 사용:", perfect_under(45))


def perfect_under2(number): 
    return {p: (2**p - 1) * 2**(p - 1) for p in range(2, number+1) if is_prime(2**p - 1)} # 이 때 2**p - 1이 소수일 때 메르센 소수라 함.

perfect_under2 = time_measure(perfect_under2)

print("@ 미사용:", perfect_under2(45))