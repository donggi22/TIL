import cv2

data = cv2.imread('opencv/dog.jpg', cv2.IMREAD_COLOR)
cv2.namedWindow('m') # 없으면 만들고 있으면 갱신, 가능하면 만들어라
cv2.imshow('m', data)

rgb_img = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
cv2.imshow('m1', rgb_img)

gray_img = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
cv2.imshow('m2', gray_img)

print(type(data)) # <class 'numpy.ndarray'>

print(cv2.waitKey(1000)) # 1000ms 대기 후 종료

cv2.destroyAllWindows()