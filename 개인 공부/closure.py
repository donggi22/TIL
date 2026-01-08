# 클로저 기억 되살리기

def dul(a):
    b = 10
    def func():
        return a**2 + b
    return func()

s = dul(3) # func를 리턴하는 게 아니라 함수가 즉시 실행 => 결과값만 반환, 함수가 바깥으로 나가지 않음
print(s)


def sam(a):
    b = 10
    def func():
        return a**2 + b
    return func

s = sam(3) # sam이 끝나도 func는 a=3, b=10을 기억함
print(s())