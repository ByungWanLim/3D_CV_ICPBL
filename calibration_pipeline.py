import cv2
import numpy as np
import glob
import os

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
board = cv2.aruco.CharucoBoard_create(7, 12, 25, 20, aruco_dict)  # 사각형/마커 크기 단위: mm

def load_charuco_images(folder):
    all_corners = []
    all_ids = []
    image_size = None

    for path in glob.glob(os.path.join(folder, '*.jpg')):
        img = cv2.imread(path)
        if img is None:
            print(f"❌ 이미지 로드 실패: {path}")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if image_size is None:
            image_size = gray.shape[::-1]

        corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict)

        if ids is not None:
            ret, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(corners, ids, gray, board)
            if ret > 0:
                all_corners.append(charuco_corners)
                all_ids.append(charuco_ids)
                print(f"✅ 검출 성공: {os.path.basename(path)}")
            else:
                print(f"⚠️ ChArUco 보간 실패: {os.path.basename(path)}")
        else:
            print(f"❌ ArUco 마커 검출 실패: {os.path.basename(path)}")

    return all_corners, all_ids, image_size

def calibrate_charuco(all_corners, all_ids, image_size):
    print("🎯 ChArUco 기반 카메라 캘리브레이션...")
    ret, mtx, dist, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
        charucoCorners=all_corners,
        charucoIds=all_ids,
        board=board,
        imageSize=image_size,
        cameraMatrix=None,
        distCoeffs=None
    )
    print("✅ 캘리브레이션 완료")
    print("📷 카메라 매트릭스:\n", mtx)
    print("🔍 왜곡 계수:\n", dist.ravel())
    return mtx, dist, rvecs, tvecs

if __name__ == '__main__':
    folder = './camera_image'
    corners, ids, img_size = load_charuco_images(folder)
    if len(corners) >= 3:
        calibrate_charuco(corners, ids, img_size)
    else:
        print("❌ 유효한 이미지가 3장 이상 필요합니다.")