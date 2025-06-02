import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import glob
import os
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class CameraCalibrationPipeline:
    """
    IC-PBL+ 카메라 캘리브레이션 파이프라인 (OpenCV 기반)
    향후 단계별로 직접 구현으로 대체할 예정
    """
    
    def __init__(self, pattern_size=(7, 12), square_size_mm=25.0):
        """
        Args:
            pattern_size: 내부 코너 개수 (height-1, width-1) = (7, 12)
            square_size_mm: 체스보드 사각형 실제 크기 (mm)
        """
        self.pattern_size = pattern_size  # (7, 12) - 내부 코너
        self.square_size_mm = square_size_mm
        
        print(f"🎯 캘리브레이션 설정:")
        print(f"   📐 내부 코너 개수: {pattern_size[0]} x {pattern_size[1]} = {pattern_size[0] * pattern_size[1]}개")
        print(f"   📏 사각형 크기: {square_size_mm}mm")
        
        # 결과 저장용
        self.image_points = []      # 2D 이미지 좌표
        self.object_points = []     # 3D 월드 좌표
        self.image_paths = []       # 성공한 이미지 경로들
        self.image_size = None      # 이미지 크기
        
        # 캘리브레이션 결과
        self.camera_matrix = None
        self.dist_coeffs = None
        self.rvecs = None
        self.tvecs = None
        self.reprojection_error = None
        
        # 3D 월드 좌표 생성 (z=0 평면)
        self.objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
        self.objp *= square_size_mm
        
        print(f"   📍 월드 좌표 범위: (0,0,0) ~ ({(pattern_size[0]-1)*square_size_mm}, {(pattern_size[1]-1)*square_size_mm}, 0)")
    
    def load_images(self, image_folder="./camera_image/"):
        """
        캘리브레이션 이미지들 로드
        """
        print(f"\n📁 이미지 로딩: {image_folder}")
        
        # 지원되는 이미지 형식
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
        image_files = []
        
        for ext in extensions:
            image_files.extend(glob.glob(os.path.join(image_folder, ext)))
            image_files.extend(glob.glob(os.path.join(image_folder, ext.upper())))
        
        if len(image_files) == 0:
            print(f"❌ {image_folder}에서 이미지를 찾을 수 없습니다!")
            print("   다음 형식을 지원합니다: jpg, jpeg, png, bmp, tiff")
            return []
        
        # 파일명으로 정렬
        image_files.sort()
        
        print(f"✅ {len(image_files)}개 이미지 발견:")
        for i, img_path in enumerate(image_files[:5]):  # 처음 5개만 표시
            print(f"   {i+1:2d}. {os.path.basename(img_path)}")
        if len(image_files) > 5:
            print(f"   ... 및 {len(image_files)-5}개 더")
        
        return image_files
    
    def detect_chessboard_corners(self, image_path, show_detection=False):
        """
        체스보드 코너 검출 (OpenCV 사용)
        향후 직접 구현으로 대체 예정
        """
        # 이미지 로드
        img = cv2.imread(image_path)
        if img is None:
            print(f"❌ 이미지 로드 실패: {image_path}")
            return None, None
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 이미지 크기 저장 (처음 이미지 기준)
        if self.image_size is None:
            self.image_size = gray.shape[::-1]  # (width, height)
            print(f"   📏 이미지 크기: {self.image_size[0]} x {self.image_size[1]}")
        
        # 체스보드 코너 검출
        ret, corners = cv2.findChessboardCorners(
            gray, 
            self.pattern_size, 
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH + 
                  cv2.CALIB_CB_NORMALIZE_IMAGE + 
                  cv2.CALIB_CB_FAST_CHECK
        )
        
        if ret:
            # 서브픽셀 정확도로 코너 위치 개선
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            
            # 검출 결과 시각화 (선택사항)
            if show_detection:
                img_with_corners = img.copy()
                cv2.drawChessboardCorners(img_with_corners, self.pattern_size, corners, ret)
                
                plt.figure(figsize=(10, 8))
                plt.imshow(cv2.cvtColor(img_with_corners, cv2.COLOR_BGR2RGB))
                plt.title(f'체스보드 검출: {os.path.basename(image_path)}')
                plt.axis('off')
                plt.show()
            
            return corners.reshape(-1, 2), gray
        
        return None, gray
    
    def process_all_images(self, image_folder="./camera_image/", show_progress=True):
        """
        모든 이미지에서 체스보드 코너 검출
        """
        print(f"\n🔍 체스보드 코너 검출 시작...")
        
        image_files = self.load_images(image_folder)
        if len(image_files) == 0:
            return False
        
        success_count = 0
        
        for i, image_path in enumerate(image_files):
            if show_progress:
                print(f"   처리 중 ({i+1:2d}/{len(image_files)}): {os.path.basename(image_path)}", end=" ")
            
            corners, gray = self.detect_chessboard_corners(image_path)
            
            if corners is not None:
                self.object_points.append(self.objp)
                self.image_points.append(corners)
                self.image_paths.append(image_path)
                success_count += 1
                
                if show_progress:
                    print("✅")
            else:
                if show_progress:
                    print("❌")
        
        print(f"\n📊 검출 결과:")
        print(f"   ✅ 성공: {success_count}개")
        print(f"   ❌ 실패: {len(image_files) - success_count}개")
        print(f"   📈 성공률: {success_count/len(image_files)*100:.1f}%")
        
        if success_count < 3:
            print(f"❌ 캘리브레이션을 위해 최소 3개의 성공한 이미지가 필요합니다!")
            return False
        
        return True
    
    def calibrate_camera(self):
        """
        카메라 캘리브레이션 수행 (OpenCV 사용)
        """
        if len(self.object_points) < 3:
            print("❌ 충분한 이미지가 없습니다. 최소 3개 필요.")
            return False
        
        print(f"\n🎯 카메라 캘리브레이션 수행...")
        print(f"   📊 사용 이미지: {len(self.object_points)}개")
        
        # OpenCV 캘리브레이션
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            self.object_points, 
            self.image_points, 
            self.image_size, 
            None, 
            None,
            flags=cv2.CALIB_FIX_K3  # K3 왜곡 계수 고정 (간단한 모델)
        )
        
        if not ret:
            print("❌ 캘리브레이션 실패!")
            return False
        
        # 결과 저장
        self.camera_matrix = mtx
        self.dist_coeffs = dist
        self.rvecs = rvecs
        self.tvecs = tvecs
        
        # 재투영 오차 계산
        total_error = 0
        for i in range(len(self.object_points)):
            projected_points, _ = cv2.projectPoints(
                self.object_points[i], rvecs[i], tvecs[i], mtx, dist
            )
            error = cv2.norm(self.image_points[i], projected_points.reshape(-1, 2), cv2.NORM_L2)
            total_error += error**2
        
        self.reprojection_error = np.sqrt(total_error / (len(self.object_points) * len(self.object_points[0])))
        
        print(f"✅ 캘리브레이션 완료!")
        return True
    
    def print_results(self):
        """
        캘리브레이션 결과 출력
        """
        if self.camera_matrix is None:
            print("❌ 캘리브레이션이 수행되지 않았습니다.")
            return
        
        print(f"\n📊 캘리브레이션 결과:")
        print(f"{'='*50}")
        
        # 내부 매개변수
        fx, fy = self.camera_matrix[0,0], self.camera_matrix[1,1]
        cx, cy = self.camera_matrix[0,2], self.camera_matrix[1,2]
        
        print(f"🎯 내부 매개변수 (Camera Matrix):")
        print(f"   초점거리: fx = {fx:.2f}, fy = {fy:.2f}")
        print(f"   주점좌표: cx = {cx:.2f}, cy = {cy:.2f}")
        print(f"   종횡비: {fy/fx:.4f}")
        
        print(f"\n📐 왜곡 계수:")
        k1, k2, p1, p2, k3 = self.dist_coeffs.ravel()
        print(f"   방사왜곡: k1 = {k1:.6f}, k2 = {k2:.6f}, k3 = {k3:.6f}")
        print(f"   접선왜곡: p1 = {p1:.6f}, p2 = {p2:.6f}")
        
        print(f"\n🎯 정확도:")
        print(f"   재투영 오차: {self.reprojection_error:.4f} pixels")
        
        # 카메라 정보 추정
        print(f"\n📱 카메라 정보 (추정):")
        sensor_width_mm = 36  # 추정값 (35mm 필름 기준)
        focal_length_mm = fx * sensor_width_mm / self.image_size[0]
        print(f"   추정 초점거리: {focal_length_mm:.1f}mm")
        print(f"   이미지 크기: {self.image_size[0]} x {self.image_size[1]}")
    
    def visualize_camera_poses(self):
        """
        카메라 포즈 3D 시각화
        """
        if self.rvecs is None:
            print("❌ 캘리브레이션 결과가 없습니다.")
            return
        
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')
        
        # 체스보드 (월드 좌표계 원점)
        board_width = (self.pattern_size[1] - 1) * self.square_size_mm
        board_height = (self.pattern_size[0] - 1) * self.square_size_mm
        
        # 체스보드 코너
        board_corners = np.array([
            [0, 0, 0],
            [board_width, 0, 0],
            [board_width, board_height, 0],
            [0, board_height, 0],
            [0, 0, 0]
        ])
        
        ax.plot(board_corners[:, 0], board_corners[:, 1], board_corners[:, 2], 
               'k-', linewidth=3, label='체스보드')
        
        # 체스보드 평면
        xx, yy = np.meshgrid(np.linspace(0, board_width, 10), 
                            np.linspace(0, board_height, 10))
        zz = np.zeros_like(xx)
        ax.plot_surface(xx, yy, zz, alpha=0.2, color='gray')
        
        # 월드 좌표계 축
        axis_length = max(board_width, board_height) * 0.3
        ax.quiver(0, 0, 0, axis_length, 0, 0, color='red', arrow_length_ratio=0.1, label='X축')
        ax.quiver(0, 0, 0, 0, axis_length, 0, color='green', arrow_length_ratio=0.1, label='Y축')
        ax.quiver(0, 0, 0, 0, 0, axis_length, color='blue', arrow_length_ratio=0.1, label='Z축')
        
        # 각 카메라 위치
        colors = plt.cm.tab10(np.linspace(0, 1, len(self.rvecs)))
        
        for i, (rvec, tvec) in enumerate(zip(self.rvecs, self.tvecs)):
            # 회전 행렬 변환
            R, _ = cv2.Rodrigues(rvec)
            
            # 카메라 중심 계산 (월드 좌표계에서)
            camera_center = -R.T @ tvec.flatten()
            
            ax.scatter(camera_center[0], camera_center[1], camera_center[2], 
                      s=100, c=[colors[i]], label=f'카메라 {i+1}')
            
            # 카메라 방향 표시
            camera_axis_length = axis_length * 0.5
            camera_axes = np.array([[camera_axis_length, 0, 0],
                                   [0, camera_axis_length, 0],
                                   [0, 0, camera_axis_length]])
            
            # 카메라 좌표계 축을 월드 좌표계로 변환
            axes_world = R.T @ camera_axes.T + camera_center.reshape(-1, 1)
            
            # 카메라 축 그리기
            cam_colors = ['darkred', 'darkgreen', 'darkblue']
            for j, color in enumerate(cam_colors):
                ax.plot([camera_center[0], axes_world[0, j]],
                       [camera_center[1], axes_world[1, j]],
                       [camera_center[2], axes_world[2, j]], 
                       color=color, alpha=0.6, linewidth=2)
        
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_zlabel('Z (mm)')
        ax.set_title('카메라 캘리브레이션 - 3D 포즈 시각화')
        
        # 범례
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.show()
    
    def save_results(self, output_file="calibration_results_opencv.json"):
        """
        캘리브레이션 결과 저장
        """
        if self.camera_matrix is None:
            print("❌ 저장할 결과가 없습니다.")
            return
        
        # JSON 직렬화 가능한 형태로 변환
        results = {
            'timestamp': datetime.now().isoformat(),
            'pattern_size': self.pattern_size,
            'square_size_mm': self.square_size_mm,
            'image_size': self.image_size,
            'num_images': len(self.image_points),
            'successful_images': [os.path.basename(path) for path in self.image_paths],
            'camera_matrix': self.camera_matrix.tolist(),
            'distortion_coefficients': self.dist_coeffs.tolist(),
            'reprojection_error': float(self.reprojection_error),
            'rotation_vectors': [rvec.tolist() for rvec in self.rvecs],
            'translation_vectors': [tvec.tolist() for tvec in self.tvecs],
            'method': 'OpenCV cv2.calibrateCamera'
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"💾 결과 저장: {output_file}")
    
    def run_full_pipeline(self, image_folder="./camera_image/"):
        """
        전체 캘리브레이션 파이프라인 실행
        """
        print("🚀 IC-PBL+ 카메라 캘리브레이션 파이프라인 시작")
        print("=" * 60)
        
        # 1단계: 이미지 처리
        if not self.process_all_images(image_folder):
            return False
        
        # 2단계: 캘리브레이션
        if not self.calibrate_camera():
            return False
        
        # 3단계: 결과 출력
        self.print_results()
        
        # 4단계: 시각화
        self.visualize_camera_poses()
        
        # 5단계: 결과 저장
        self.save_results()
        
        print("\n" + "=" * 60)
        print("✅ OpenCV 기반 캘리브레이션 완료!")
        print("📋 다음 단계: 직접 구현으로 단계별 대체")
        
        return True

def test_with_sample_data():
    """
    샘플 데이터로 파이프라인 테스트 (실제 이미지가 없을 때)
    """
    print("⚠️  실제 이미지가 없어 테스트 모드로 실행합니다.")
    print("   실제 사용 시에는 ./camera_image/ 폴더에 체스보드 사진을 저장하세요.")

def main():
    """
    메인 실행 함수
    """
    print("🎯 IC-PBL+ 카메라 캘리브레이션 프로젝트")
    print("=" * 50)
    
    # 폴더 구조 확인 및 생성
    checkerboard_folder = "./checkerboard/"
    image_folder = "./camera_image/"
    
    # 체스보드 폴더 확인
    if not os.path.exists(checkerboard_folder):
        print(f"📁 {checkerboard_folder} 폴더가 없습니다. 생성합니다...")
        os.makedirs(checkerboard_folder)
        print(f"✅ 체스보드 폴더 생성 완료!")
    
    # 이미지 폴더 확인
    if not os.path.exists(image_folder):
        print(f"📁 {image_folder} 폴더가 없습니다. 생성합니다...")
        os.makedirs(image_folder)
        print(f"✅ 이미지 폴더 생성 완료!")
        
    # 폴더 구조 표시
    print(f"\n📂 현재 폴더 구조:")
    print(f"   📁 {checkerboard_folder} ← 체스보드 PDF/PNG 파일")
    print(f"   📁 {image_folder} ← 촬영한 체스보드 사진들")
    
    # 체스보드 파일 확인
    checkerboard_files = glob.glob(os.path.join(checkerboard_folder, "*.pdf")) + \
                        glob.glob(os.path.join(checkerboard_folder, "*.png"))
    
    if checkerboard_files:
        print(f"\n✅ 체스보드 파일 발견:")
        for file in checkerboard_files:
            print(f"   📄 {os.path.basename(file)}")
    else:
        print(f"\n⚠️  {checkerboard_folder}에 체스보드 파일이 없습니다.")
        print(f"   체스보드 생성 코드를 먼저 실행하세요.")
    
    # 캘리브레이션 파이프라인 생성
    # 주의: 우리 체스보드는 8x13이므로 내부 코너는 7x12
    calibrator = CameraCalibrationPipeline(
        pattern_size=(7, 12),  # 내부 코너 개수
        square_size_mm=25.0    # 실제 측정값으로 수정 필요
    )
    
    # 이미지 있는지 확인
    image_files = calibrator.load_images(image_folder)
    
    if len(image_files) == 0:
        print(f"\n📸 촬영 가이드:")
        print(f"   1. {checkerboard_folder}의 PDF를 A4 용지에 출력")
        print(f"   2. 평평한 보드에 부착")
        print(f"   3. 15-20장 촬영")
        print(f"   4. 촬영한 사진들을 {image_folder}에 저장")
        print(f"   5. 다시 이 코드 실행")
        print(f"\n📋 촬영 조건:")
        print(f"   • 다양한 각도 (15°, 30°, 45° 기울임)")
        print(f"   • 다양한 거리 (가까이, 중간, 멀리)")
        print(f"   • 체스보드 전체가 화면에 들어오게")
        print(f"   • 선명한 초점 유지")
        print(f"   • 4개 모서리 마커(빨강,파랑,초록,오렌지) 모두 보이게")
        return
    
    # 전체 파이프라인 실행
    success = calibrator.run_full_pipeline(image_folder)
    
    if success:
        print("\n🎯 다음 단계 준비:")
        print("   1. OpenCV 결과 확인 및 검증")
        print("   2. 직접 구현으로 단계별 대체 시작")
        print("   3. Harris corner detector 구현")
        print("   4. 호모그래피 추정 직접 구현")
        print("   5. Zhang's 방법 직접 구현")

if __name__ == "__main__":
    main()