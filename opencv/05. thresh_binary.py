import cv2

gray_img = cv2.imread('opencv/dog.jpg', cv2.IMREAD_GRAYSCALE)

print('변환 후')
ret, th_img = cv2.threshold(src=gray_img, thresh=150, maxval=255, type=cv2.THRESH_BINARY) # 인자에 minval은 없음 최솟값은 항상 0
# th_img[th_img == 0] = 50  # numpy를 이용하여 원하는 min값으로 바꾸기
# OpenCV threshold는 “두 값 (0 vs maxval)”만 사용하는 이진화 함수

print('원본')
print(gray_img.shape)
print(gray_img[0, 0])
print(gray_img[100, 100])

print('변환 결과')
print(th_img.shape)
print(th_img[0, 0])
print(th_img[100, 100])

cv2.imshow('img', gray_img)
cv2.imshow('th', th_img)
key = cv2.waitKey(0)
print(key)
cv2.destroyAllWindows()