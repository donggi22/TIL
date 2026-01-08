# Prime Number
import math

def is_prime(number):
    count = 0
    for i in range(1, int(math.sqrt(number))+1):
        if number % i == 0:
            count += 1
    return count == 1

# print(list(range(1, int(math.sqrt(2))+1)))

# print(is_prime(6))

def prime_under(number):
    num_list = [i for i in range(2, number+1) if is_prime(i)]
    return num_list

# print("100 이하의 소수들:", prime_under(100))
# print(int(math.sqrt(2)+1))

# Perfect Number: 자기 자신을 제외한 약수들의 합이 자기 자신이 되는 것 혹은 약수들의 합이 자신의 두배가 되는 수
def factor_list(number): # 이 함수는 자기 자신을 제외한 진약수만 구하는 함수
    factor_lst = []
    for i in range(1, number):
        if number % i ==0:
            factor_lst.append(i)
    return factor_lst

# print(factor_list(20))

def is_perfect(number):
    return number == sum(factor_list(number))
        
# print(is_perfect(496))

def perfect_under(number):
    return [i for i in range(2, number+1) if is_perfect(i)]
# print(perfect_under(10000))

# 오일러-유클리드 정리에 의해, 짝수 완전수와 메르센 소수는 일대일 대응 관계임.
def perfect_under2(number): 
    return {p: (2**p - 1) * 2**(p - 1) for p in range(2, number+1) if is_prime(2**p - 1)} # 이 때 2**p - 1이 소수일 때 메르센 소수라 함.

# print(perfect_under2(45)) # {2: 6, 3: 28, 5: 496, 7: 8128, 13: 33550336, 17: 8589869056, 19: 137438691328, 31: 2305843008139952128}

import matplotlib.pyplot as plt

n = 1000
# plt.xlim(0, 2*n)
# plt.ylim(0, 2*n)
# plt.plot(range(n+1), range(n+1), 'r', '-')
# plt.bar(range(1, n+1), [sum(factor_list(i)) for i in range(1, n+1)])
# # plt.hist([sum(factor_list(i)) for i in range(1, n+1)])
# plt.show()

print(is_prime(2**67 - 1))