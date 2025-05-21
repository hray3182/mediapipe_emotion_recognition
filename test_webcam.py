import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import mediapipe as mp
from sklearn.preprocessing import StandardScaler
import os
import sys

# 載入模型和必要的組件
MODEL_PATH = os.path.join('models', 'best_emotion_model.h5')
if not os.path.exists(MODEL_PATH):
    print(f"錯誤：找不到模型文件 {MODEL_PATH}")
    sys.exit(1)

try:
    model = keras.models.load_model(MODEL_PATH)
except Exception as e:
    print(f"載入模型時發生錯誤：{str(e)}")
    sys.exit(1)

# 載入 StandardScaler
scaler = StandardScaler()

# MediaPipe Face Mesh 初始化
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# 情緒標籤
EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# 初始化 webcam
print("正在嘗試開啟 webcam...")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("錯誤：無法開啟 webcam")
    print("請確認：")
    print("1. webcam 已正確連接")
    print("2. 在 WSL 中已正確設定 webcam 權限")
    print("3. 嘗試在 Windows 中運行此程式")
    sys.exit(1)

print("成功開啟 webcam！按 'q' 鍵退出程式")

while True:
    ret, frame = cap.read()
    if not ret:
        print("無法讀取 webcam 畫面")
        break

    # 轉換為 RGB 格式
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # 使用 MediaPipe 檢測臉部
    results = face_mesh.process(rgb_frame)
    
    if results.multi_face_landmarks:
        # 獲取 blendshapes
        blendshapes = results.multi_face_landmarks[0].blendshapes
        
        if blendshapes:
            # 提取 blendshape 值
            blendshape_values = np.array([blendshape.score for blendshape in blendshapes])
            
            # 標準化數據
            blendshape_values = scaler.fit_transform(blendshape_values.reshape(1, -1))
            
            # 預測情緒
            prediction = model.predict(blendshape_values, verbose=0)
            emotion_idx = np.argmax(prediction[0])
            emotion = EMOTION_LABELS[emotion_idx]
            confidence = prediction[0][emotion_idx]
            
            # 在畫面上顯示結果
            cv2.putText(frame, f"Emotion: {emotion}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Confidence: {confidence:.2f}", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # 顯示畫面
    cv2.imshow('Emotion Recognition', frame)
    
    # 按 'q' 退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 釋放資源
cap.release()
cv2.destroyAllWindows() 