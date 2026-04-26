import cv2

cap = cv2.VideoCapture('opencv/m_v.avi')

if not cap.isOpened():
    print('실행 불가')
    exit(1)

ret, img = cap.read()

if not ret:
    print('캡쳐 불가')
    exit(1)

fps = cap.get(cv2.CAP_PROP_FPS)

while True:
    ret, img = cap.read()

    if not ret:
        print('영상 종료')
        break

    cv2.imshow('frame', img)

    key = cv2.waitKey(int(1000 / fps))
    
    if key & 0xFF == 27:
        print('key')
        break

cap.release()
cv2.destroyAllWindows()