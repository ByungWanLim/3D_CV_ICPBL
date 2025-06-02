import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_pdf import PdfPages
import os

def create_optimal_calibration_chessboard(square_size_mm=25):
    """
    IC-PBL+ 프로젝트를 위한 최적화된 캘리브레이션 체스보드
    - 황금비율 8x13 그리드
    - 비대칭성으로 방향성 제공
    - 서브픽셀 정확도를 위한 십자가 마커
    - 방향성 그래디언트
    """
    
    # 황금비율 기반 크기
    rows, cols = 8, 13
    
    print(f"🎯 최적화된 캘리브레이션 체스보드 생성")
    print(f"📐 크기: {rows} x {cols} (내부 코너: {rows-1} x {cols-1} = {(rows-1)*(cols-1)}개)")
    print(f"📏 사각형 크기: {square_size_mm}mm x {square_size_mm}mm")
    print(f"📋 전체 크기: {cols * square_size_mm}mm x {rows * square_size_mm}mm")
    
    # Figure 생성
    fig_width = cols * square_size_mm / 25.4  # mm to inch
    fig_height = rows * square_size_mm / 25.4
    
    fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_height))
    
    # 1. 기본 체스보드 패턴 (의도적 패턴 깨기 포함)
    for i in range(rows):
        for j in range(cols):
            # 특정 위치에서 패턴 깨기 (비대칭성 제공)
            if (i == 2 and j == 5) or (i == 5 and j == 8):
                color = 'gray'  # 의도적 패턴 깨기
            elif (i + j) % 2 == 0:
                color = 'white'
            else:
                color = 'black'
            
            rect = patches.Rectangle(
                (j, rows - 1 - i), 1, 1,
                linewidth=0, edgecolor='none', facecolor=color
            )
            ax.add_patch(rect)
    
    # 2. 서브픽셀 정확도를 위한 십자가 마커 (주요 교차점에)
    cross_size = 0.08
    for i in range(2, rows-1, 2):  # 격자로 배치
        for j in range(2, cols-1, 2):
            # 십자가 마커: 세로선
            line_v = patches.Rectangle(
                (j - cross_size/2, rows - 1 - i - cross_size), 
                cross_size, 2*cross_size,
                facecolor='red', alpha=0.9, linewidth=0
            )
            # 십자가 마커: 가로선  
            line_h = patches.Rectangle(
                (j - cross_size, rows - 1 - i - cross_size/2), 
                2*cross_size, cross_size,
                facecolor='red', alpha=0.9, linewidth=0
            )
            ax.add_patch(line_v)
            ax.add_patch(line_h)
    
    # 3. 방향성 마커 (4개 모서리)
    marker_size = 0.25
    
    # 좌상단: 빨간 사각형
    top_left = patches.Rectangle(
        (0.1, rows - 0.35), marker_size, marker_size,
        facecolor='red', edgecolor='black', linewidth=1
    )
    ax.add_patch(top_left)
    
    # 우상단: 파란 원
    top_right = patches.Circle(
        (cols - 0.25, rows - 0.25), 0.15,
        facecolor='blue', edgecolor='black', linewidth=1
    )
    ax.add_patch(top_right)
    
    # 좌하단: 녹색 삼각형
    triangle_points = np.array([[0.1, 0.1], [0.35, 0.1], [0.225, 0.35]])
    triangle = patches.Polygon(triangle_points, facecolor='green', 
                             edgecolor='black', linewidth=1)
    ax.add_patch(triangle)
    
    # 우하단: 오렌지 다이아몬드
    diamond_points = np.array([[cols-0.25, 0.1], [cols-0.1, 0.25], 
                              [cols-0.25, 0.4], [cols-0.4, 0.25]])
    diamond = patches.Polygon(diamond_points, facecolor='orange',
                            edgecolor='black', linewidth=1)
    ax.add_patch(diamond)
    
    # 4. 방향성 그래디언트 (왼쪽 경계)
    for i in range(rows):
        gradient_intensity = 0.3
        rect = patches.Rectangle(
            (-0.3, rows - 1 - i), 0.3, 1,
            facecolor='black', alpha=gradient_intensity
        )
        ax.add_patch(rect)
    
    # 5. ID 번호 (중앙 하단)
    ax.text(cols/2, 0.5, 'CAM-CAL-01', fontsize=8, ha='center', va='center',
            color='purple', fontweight='bold', 
            bbox=dict(boxstyle="round,pad=0.1", facecolor='white', alpha=0.8))
    
    # 축 설정
    ax.set_xlim(-0.5, cols + 0.5)
    ax.set_ylim(-0.5, rows + 0.5)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # 여백 제거
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    
    return fig, rows, cols

def save_chessboard_files(fig, rows, cols, square_size_mm=25):
    """
    체스보드를 PDF와 PNG로 저장
    """
    base_dir = "./checkerboard"
    os.makedirs(base_dir, exist_ok=True)  # ✅ 디렉터리 생성 코드 추가
    
    base_filename = f"./checkerboard/optimal_chessboard_{rows}x{cols}_{square_size_mm}mm"
    
    # PDF 저장 (고품질 출력용)
    pdf_filename = f"{base_filename}.pdf"
    fig.savefig(pdf_filename, format='pdf', dpi=300, bbox_inches='tight', 
               pad_inches=0, facecolor='white')
    
    # PNG 저장 (미리보기용)
    png_filename = f"{base_filename}.png"
    fig.savefig(png_filename, format='png', dpi=300, bbox_inches='tight', 
               pad_inches=0, facecolor='white')
    
    print(f"✅ PDF 파일 저장: {pdf_filename}")
    print(f"✅ PNG 파일 저장: {png_filename}")
    
    return pdf_filename, png_filename

def print_usage_instructions():
    """
    사용 지침 출력
    """
    instructions = """
🎯 최적화된 캘리브레이션 체스보드 사용법

📋 1단계: 출력
  • 생성된 PDF 파일을 A4 용지에 출력
  • 프린터 설정: "실제 크기" 또는 "크기 조정 없음"
  • 레이저 프린터 사용 권장 (더 선명함)

📏 2단계: 크기 측정
  • 디지털 캘리퍼로 사각형 하나의 실제 크기 측정
  • 여러 사각형 측정하여 평균값 계산
  • 측정값을 코드에 입력 (예: 24.8mm)

🔧 3단계: 고정
  • 평평한 보드나 아크릴판에 부착
  • 구겨지거나 휘어지지 않게 주의
  • 조명이 균일한 곳에서 사용

📸 4단계: 촬영 (15-20장)
  • 다양한 각도: 15°, 30°, 45° 기울임
  • 다양한 거리: 가까이, 중간, 멀리
  • 체스보드 전체가 화면에 들어오게
  • 선명한 초점 유지

🎯 특별 기능:
  • 4개 모서리 마커로 방향 인식
  • 빨간 십자가 마커로 정밀한 코너 검출
  • 회색 사각형으로 180도 회전 구분
  • 좌측 그래디언트로 추가 방향성 제공
"""
    print(instructions)

def main():
    """
    메인 실행 함수
    """
    print("🎯 IC-PBL+ 카메라 캘리브레이션 프로젝트")
    print("최적화된 체스보드 생성")
    print("=" * 50)
    
    # 최적 체스보드 생성
    fig, rows, cols = create_optimal_calibration_chessboard(square_size_mm=25)
    
    # 파일 저장
    pdf_file, png_file = save_chessboard_files(fig, rows, cols)
    
    # 시각화
    plt.show()
    
    # 사용 지침
    print_usage_instructions()
    
    print("\n" + "=" * 50)
    print("✅ 최적화된 체스보드 생성 완료!")
    print(f"📁 PDF 파일: {pdf_file}")
    print(f"📁 PNG 파일: {png_file}")
    print("\n📋 다음 단계:")
    print("1. PDF 파일을 A4 용지에 출력")
    print("2. 실제 사각형 크기 측정")
    print("3. 평평한 보드에 부착")
    print("4. 15-20장 사진 촬영")
    print("5. OpenCV 파이프라인 코드 실행")

if __name__ == "__main__":
    main()