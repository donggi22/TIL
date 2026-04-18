import cv2
import numpy as np

# =========================
# 1. 체커보드 설정
# =========================
chessboard_size = (9, 6)

# 3D 좌표 생성 (z=0 평면)
objp = np.zeros((chessboard_size[0]*chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

# =========================
# 2. 이미지 불러오기
# =========================
img = cv2.imread('chessboard.jpg')  # ← 체커보드 이미지 필요
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# =========================
# 3. 코너 검출
# =========================
ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

if not ret:
    print("체커보드 코너 못 찾음")
    exit()

# =========================
# 4. 캘리브레이션 (1장으로도 가능)
# =========================
objpoints = [objp]
imgpoints = [corners]

ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, gray.shape[::-1], None, None
)

# =========================
# 5. solvePnP
# =========================
retval, rvec, tvec = cv2.solvePnP(
    objp,       # 3D 좌표
    corners,    # 2D 좌표
    camera_matrix,
    dist_coeffs
)

print("=== Camera Matrix ===")
print(camera_matrix)

print("\n=== Distortion ===")
print(dist_coeffs)

print("\n=== rvec (회전) ===")
print(rvec)

print("\n=== tvec (위치) ===")
print(tvec)