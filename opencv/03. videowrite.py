import cv2

# 캠 연결
cam = cv2.VideoCapture(0)

# 연결 확인
if not cam.isOpened():
    print('연결 불가')
    exit(1)

# 프레임 캡쳐
ret, img = cam.read()

# 캡쳐 확인
if not ret:
    print('캡쳐 불가')
    exit(1)

# 영상 저장을 위한 코덱 설정
codec = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
# codec = cv2.VideoWriter_fourcc(*'XVID') # 이렇게 asterisk로 unpacking 해도 됨.

fps = 30
h, w = img.shape[:2]
m_v = cv2.VideoWriter('opencv/m_v.avi', codec, fps, (w, h))

if not m_v.isOpened():
    print('동영상 생성 불가')
    exit(1)

while True:
    ret, img = cam.read()

    if not ret:
        print('캡쳐 불가')
        break
    
    m_v.write(img)

    cv2.imshow('frame', img)
    
    key = cv2.waitKey(int(1000 // fps))

    if key & 0xFF == 27:
        print(key)
        break

cam.release()
m_v.release()
cv2.destroyAllWindows()