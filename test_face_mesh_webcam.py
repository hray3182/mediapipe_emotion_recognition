import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os

# --- 設定區塊 ---
DATASET_ROOT_DIR: str = '/home/ray/program/mediapipe_emotion_recognition'
MODEL_PATH: str = os.path.join(DATASET_ROOT_DIR, 'face_landmarker.task')

def main():
    # 檢查模型檔案是否存在
    if not os.path.exists(MODEL_PATH):
        print(f"錯誤：模型檔案未找到：{MODEL_PATH}")
        print("請確認已下載 'face_landmarker_v2_with_blendshapes.task' 並放置於正確路徑。")
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

    # 初始化網路攝影機
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("錯誤：無法開啟網路攝影機")
        return

    # 設定視窗大小
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # 用於計算 FPS
    frame_count = 0
    start_time = cv2.getTickCount()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("錯誤：無法讀取攝影機畫面")
                break

            # 水平翻轉畫面（鏡像）
            frame = cv2.flip(frame, 1)

            # 轉換為 RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            # 處理當前幀
            detection_result = landmarker.detect_for_video(mp_image, frame_count)

            # 如果檢測到人臉
            if detection_result.face_landmarks:
                landmarks = detection_result.face_landmarks[0]
                
                # 繪製所有臉部特徵點
                for landmark in landmarks:
                    # 將相對座標轉換為像素座標
                    x = int(landmark.x * frame.shape[1])
                    y = int(landmark.y * frame.shape[0])
                    
                    # 繪製點
                    cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

            # 計算並顯示 FPS
            frame_count += 1
            if frame_count % 30 == 0:  # 每 30 幀更新一次 FPS
                end_time = cv2.getTickCount()
                fps = 30 * cv2.getTickFrequency() / (end_time - start_time)
                start_time = cv2.getTickCount()
                cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # 顯示畫面
            cv2.imshow('Face Mesh Test', frame)

            # 按 'q' 退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        # 清理資源
        cap.release()
        cv2.destroyAllWindows()
        landmarker.close()

if __name__ == "__main__":
    main() 