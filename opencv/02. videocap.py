import cv2

# print(cv2.COLOR_BGR2RGB) # 4
# print(cv2.COLOR_RGB2BGR) # 4 위와 동일, B와 R swap만 해주면 됨
# print(cv2.COLOR_GRAY2BGR)
# print(cv2.COLOR_GRAY2RGB)
# print(cv2.COLOR_RGB2GRAY)
# print(cv2.COLOR_BGR2GRAY)
# print(cv2.IMREAD_COLOR)

# cnt = 0
# for v in dir(cv2):
#     attr = getattr(cv2, v)
#     # print(v, ':', attr)
#     if isinstance(attr, int):
#         cnt += 1
# print(cnt) # 1680

# with open('opencv/cv2_member_int.txt', 'w', encoding='utf-8') as f:
#     for v in dir(cv2):
#         attr = getattr(cv2, v)

#         if isinstance(attr, int):
#             f.write(f"{v} : {attr}\n")

# 카메라 열기
cam = cv2.VideoCapture(0) # 0: 첫번째 카메라 사용

if cam.isOpened() == False: # 카메라나 동영상 파일로 올바르게 초기화되었는지 확인
    print('연결 불가')
    exit(1)

while True:
    ret, img = cam.read() # 프레임 읽기
    if ret == False:
        print('캡쳐 불가')
        break

    cv2.imshow('cam', img)

    key = cv2.waitKey(100) # 100ms
    # print(key)
    if key == 27: # 아스키로 esc는 27
        break

# 카메라 닫기
cam.release() # 카메라 리소스 해제
cv2.destroyAllWindows() # 영상 출력 창까지 닫기