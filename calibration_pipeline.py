import cv2
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def load_images(image_folder, pattern_size):
    objpoints = []  # 3D points in real world space
    imgpoints = []  # 2D points in image plane

    # 체스보드의 3D 기준 좌표 생성
    objp = np.zeros((pattern_size[0]*pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
    objp *= 25.0  # mm 단위

    image_paths = glob.glob(os.path.join(image_folder, "*.jpg"))
    image_size = None

    for path in sorted(image_paths):
        img = cv2.imread(path)
        if img is None:
            print(f"❌ 이미지 로드 실패: {path}")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if image_size is None:
            image_size = gray.shape[::-1]

        # OpenCV 고급 검출기 (SB 알고리즘 사용)
        ret, corners = cv2.findChessboardCornersSB(gray, pattern_size, flags=cv2.CALIB_CB_NORMALIZE_IMAGE)

        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)
            print(f"✅ 검출 성공: {os.path.basename(path)}")
        else:
            print(f"❌ 검출 실패: {os.path.basename(path)}")

    return objpoints, imgpoints, image_size


def calibrate(objpoints, imgpoints, image_size):
    print(f"🎯 카메라 캘리브레이션 시작...")

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, image_size, None, None
    )

    print("✅ 캘리브레이션 완료")
    print("📷 카메라 매트릭스:\n", mtx)
    print("🔍 왜곡 계수:\n", dist.ravel())

    return mtx, dist, rvecs, tvecs


def visualize_camera_poses(rvecs, tvecs, pattern_size, square_size):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for i in range(len(rvecs)):
        R, _ = cv2.Rodrigues(rvecs[i])
        cam_pos = -R.T @ tvecs[i].reshape(3)

        ax.scatter(cam_pos[0], cam_pos[1], cam_pos[2], marker='o', label=f"Cam {i+1}")
        ax.text(cam_pos[0], cam_pos[1], cam_pos[2], f"{i+1}", fontsize=8)

    # 체스보드 기준 평면 그리기
    board_w = (pattern_size[0]-1) * square_size
    board_h = (pattern_size[1]-1) * square_size
    xx, yy = np.meshgrid(np.linspace(0, board_w, 2), np.linspace(0, board_h, 2))
    zz = np.zeros_like(xx)
    ax.plot_surface(xx, yy, zz, alpha=0.2, color='gray')

    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    ax.set_title("📸 Camera poses")
    plt.legend()
    plt.show()


def run_pipeline(image_folder, pattern_size=(7,12), square_size=25.0):
    objpoints, imgpoints, image_size = load_images(image_folder, pattern_size)

    if len(objpoints) < 3:
        print("❌ 유효한 이미지가 3장 이상 필요합니다.")
        return

    mtx, dist, rvecs, tvecs = calibrate(objpoints, imgpoints, image_size)
    visualize_camera_poses(rvecs, tvecs, pattern_size, square_size)


if __name__ == "__main__":
    # 사용자 체스보드 기반 캘리브레이션 실행
    run_pipeline("./camera_image", pattern_size=(7, 12), square_size=25.0)
