import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os
from sklearn.preprocessing import StandardScaler

# --- 設定區塊 ---
DATASET_ROOT_DIR = '/home/ray/program/mediapipe_emotion_recognition'
MODEL_PATH = os.path.join(DATASET_ROOT_DIR, 'face_landmarker.task')
EMOTION_MODEL_PATH = os.path.join(DATASET_ROOT_DIR, 'face_mesh_models', 'best_emotion_landmarks_mlp_model.h5')

# 情緒標籤
EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

def main():
    # 檢查模型檔案是否存在
    if not os.path.exists(MODEL_PATH):
        print(f"錯誤：MediaPipe 模型檔案未找到：{MODEL_PATH}")
        return
    if not os.path.exists(EMOTION_MODEL_PATH):
        print(f"錯誤：情緒識別模型檔案未找到：{EMOTION_MODEL_PATH}")
        return

    # 載入情緒識別模型
    try:
        model = keras.models.load_model(EMOTION_MODEL_PATH)
    except Exception as e:
        print(f"載入情緒識別模型時發生錯誤：{str(e)}")
        return

    # 初始化 MediaPipe Face Landmarker
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False
    )
    landmarker = vision.FaceLandmarker.create_from_options(options)

    # 初始化 StandardScaler
    scaler = StandardScaler()

    # 初始化網路攝影機
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("錯誤：無法開啟網路攝影機")
        return

    # 設定視窗大小
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("成功開啟網路攝影機！按 'q' 鍵退出程式")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("無法讀取網路攝影機畫面")
            break

        # 轉換為 RGB 格式
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 創建 MediaPipe Image 物件
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # 使用 MediaPipe 檢測臉部
        detection_result = landmarker.detect(mp_image)
        
        if detection_result.face_landmarks:
            # 獲取第一個臉的 landmarks
            landmarks = detection_result.face_landmarks[0]
            
            # 提取所有點位的 x, y, z 座標
            mesh_points = []
            for landmark in landmarks:
                mesh_points.extend([landmark.x, landmark.y, landmark.z])
            
            # 將特徵轉換為 numpy 陣列並重塑
            features = np.array(mesh_points).reshape(1, -1)
            
            # 標準化特徵
            features_scaled = scaler.fit_transform(features)
            
            # 預測情緒
            prediction = model.predict(features_scaled, verbose=0)
            emotion_idx = np.argmax(prediction[0])
            emotion = EMOTION_LABELS[emotion_idx]
            confidence = prediction[0][emotion_idx]
            
            # 在畫面上顯示結果
            cv2.putText(frame, f"情緒: {emotion}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"信心度: {confidence:.2f}", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # 繪製臉部網格
            h, w = frame.shape[:2]
            for landmark in landmarks:
                x, y = int(landmark.x * w), int(landmark.y * h)
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
        
        # 顯示畫面
        cv2.imshow('情緒識別', frame)
        
        # 按 'q' 退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 釋放資源
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 