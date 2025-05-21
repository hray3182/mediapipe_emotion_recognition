import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os

# 設定路徑
DATASET_ROOT_DIR = '/home/ray/program/mediapipe_emotion_recognition'
MODEL_PATH = os.path.join(DATASET_ROOT_DIR, 'face_landmarker.task')

def draw_mesh(image, landmarks, connections):
    """繪製臉部網格"""
    h, w = image.shape[:2]
    for connection in connections:
        start_idx = connection[0]
        end_idx = connection[1]
        
        # 獲取連接點的座標
        start_point = (int(landmarks[start_idx].x * w), int(landmarks[start_idx].y * h))
        end_point = (int(landmarks[end_idx].x * w), int(landmarks[end_idx].y * h))
        
        # 繪製線段
        cv2.line(image, start_point, end_point, (0, 255, 0), 1)

def visualize_mesh(image_path):
    """可視化單張圖像的臉部網格和 blendshapes"""
    # 初始化 MediaPipe Face Landmarker
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True
    )
    landmarker = vision.FaceLandmarker.create_from_options(options)

    # 讀取圖像
    image = cv2.imread(image_path)
    if image is None:
        print(f"無法讀取圖像：{image_path}")
        return

    # 調整圖像大小
    image = cv2.resize(image, (192, 192))
    
    # 轉換為 RGB
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 創建 MediaPipe Image 物件
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
    
    # 進行檢測
    detection_result = landmarker.detect(mp_image)
    
    # 如果檢測到人臉
    if detection_result.face_landmarks:
        # 獲取第一個臉的標記點
        landmarks = detection_result.face_landmarks[0]
        
        # 獲取臉部網格的連接
        connections = mp.solutions.face_mesh.FACEMESH_TESSELATION
        
        # 繪製網格
        draw_mesh(image, landmarks, connections)
        
        # 如果有 blendshapes，顯示前 5 個最顯著的表情
        if detection_result.face_blendshapes:
            blendshapes = detection_result.face_blendshapes[0]
            # 按分數排序
            sorted_blendshapes = sorted(blendshapes, key=lambda x: x.score, reverse=True)
            
            # 顯示前 5 個最顯著的表情
            print("\n前 5 個最顯著的表情：")
            for i, bs in enumerate(sorted_blendshapes[:5]):
                print(f"{bs.category_name}: {bs.score:.3f}")
        
        # 顯示結果
        cv2.imshow('Face Mesh', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("未檢測到人臉")

def main():
    # 測試一張圖片
    test_image_path = os.path.join(DATASET_ROOT_DIR, 'dataset/train/Angry/22830.png')
    visualize_mesh(test_image_path)

if __name__ == "__main__":
    main() 