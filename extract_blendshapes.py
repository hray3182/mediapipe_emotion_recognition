import pandas as pd
import numpy as np
import cv2
import os
import glob
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple, Union

# 導入新的 MediaPipe Tasks API
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# --- 設定區塊 ---
DATASET_ROOT_DIR: str = '/home/ray/program/mediapipe_emotion_recognition'

# 輸出 Blendshape CSV 檔案的目錄
OUTPUT_CSV_DIR: str = os.path.join(DATASET_ROOT_DIR, 'extracted_blendshapes')
os.makedirs(OUTPUT_CSV_DIR, exist_ok=True) # 確保輸出目錄存在

# 定義情緒名稱到數字標籤的對應 (FER2013 標準)
# 確保這個對應與您模型訓練時的標籤順序一致
EMOTION_TO_LABEL: Dict[str, int] = {
    'Angry': 0,
    'Disgust': 1,
    'Fear': 2,
    'Happy': 3,
    'Neutral': 4,
    'Sad': 5,
    'Surprise': 6
}
LABEL_TO_EMOTION: Dict[int, str] = {v: k for k, v in EMOTION_TO_LABEL.items()} # 用於驗證

# MediaPipe Face Landmarker 模型檔案路徑
# 請確保這個檔案已經下載並放在 DATASET_ROOT_DIR 中
MODEL_PATH: str = os.path.join(DATASET_ROOT_DIR, 'face_landmarker.task')

# MediaPipe Face Landmarker 提供的 52 個 Blendshape 名稱
# 此列表來自 MediaPipe 官方文件，確保其準確性和順序
BLENDSHAPE_NAMES: List[str] = [
    "browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft", "browOuterUpRight",
    "cheekPuff", "cheekSquintLeft", "cheekSquintRight", "eyeBlinkLeft", "eyeBlinkRight",
    "eyeLookDownLeft", "eyeLookDownRight", "eyeLookInLeft", "eyeLookInRight", "eyeLookOutLeft",
    "eyeLookOutRight", "eyeLookUpLeft", "eyeLookUpRight", "eyeSquintLeft", "eyeSquintRight",
    "eyeWideLeft", "eyeWideRight", "jawForward", "jawLeft", "jawOpen", "jawRight",
    "mouthClose", "mouthDimpleLeft", "mouthDimpleRight", "mouthFrownLeft", "mouthFrownRight",
    "mouthFunnel", "mouthLeft", "mouthLowerDownLeft", "mouthLowerDownRight", "mouthPressLeft",
    "mouthPressRight", "mouthPucker", "mouthRight", "mouthRollLower", "mouthRollUpper",
    "mouthShrugLower", "mouthShrugUpper", "mouthSmileLeft", "mouthSmileRight", "mouthStretchLeft",
    "mouthStretchRight", "mouthUpperUpLeft", "mouthUpperUpRight", "noseSneerLeft", "noseSneerRight",
    "tongueOut"
]

# --- MediaPipe Face Landmarker 初始化 ---
# 創建一個 BaseOptions 物件來指定模型路徑
base_options: python.BaseOptions = python.BaseOptions(model_asset_path=MODEL_PATH)

# 創建 FaceLandmarkerOptions 物件來配置任務
face_landmarker_options: vision.FaceLandmarkerOptions = vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE, # 對於靜態圖像處理，設定為 IMAGE
    num_faces=1,                          # 預期偵測的最大人臉數量
    min_face_detection_confidence=0.5,    # 最小人臉偵測置信度
    min_face_presence_confidence=0.5,     # 最小人臉存在置信度（新 API 新增）
    min_tracking_confidence=0.5,          # 最小人臉追蹤置信度
    output_face_blendshapes=True,         # 啟用 Blendshape 輸出
    output_facial_transformation_matrixes=False # 如果不需要姿態轉換矩陣，可設定為 False
)

# 使用新的 API 初始化 FaceLandmarker
face_landmarker: vision.FaceLandmarker = vision.FaceLandmarker.create_from_options(face_landmarker_options)

# --- 輔助函數 ---
def extract_blendshapes_from_image(image_path: str, landmarker_processor: vision.FaceLandmarker) -> Optional[List[float]]:
    """
    從圖像檔案中提取 MediaPipe Blendshape。
    
    Args:
        image_path: 圖像檔案的路徑
        landmarker_processor: MediaPipe Face Landmarker 處理器
        
    Returns:
        Optional[List[float]]: 提取到的 Blendshape 分數列表，如果處理失敗則返回 None
    """
    try:
        # 讀取灰階圖像
        gray_image: Optional[np.ndarray] = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if gray_image is None:
            # print(f"警告：無法讀取圖像檔案或檔案損壞：{image_path}") # 若警告過多可註解掉
            return None

        # 將圖像調整為 192x192 大小
        gray_image = cv2.resize(gray_image, (192, 192), interpolation=cv2.INTER_AREA)

        # 將灰階圖轉換為 RGB (MediaPipe 期望 RGB 輸入)
        rgb_image_np: np.ndarray = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2RGB)
        
        # 將 NumPy 陣列轉換為 MediaPipe Image 物件
        # 圖像尺寸調整通常交給 MediaPipe 內部處理或模型預處理，或在模型選項中配置
        mp_image: mp.Image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image_np)
        
        detection_result: vision.FaceLandmarkerResult = landmarker_processor.detect(mp_image)

        # 檢查是否偵測到人臉地標點且有 Blendshape 輸出
        if detection_result.face_landmarks and detection_result.face_blendshapes:
            # face_blendshapes 是一個列表，每個元素對應一張人臉的 BlendshapeResult 物件
            # 我們假設每張圖只有一張臉，取第一個
            current_blendshapes_result = detection_result.face_blendshapes[0] 
            
            # 直接從 blendshapes 列表中獲取分數
            blendshape_dict: Dict[str, float] = {}
            for bs in current_blendshapes_result:
                blendshape_dict[bs.category_name] = bs.score
            
            # 確保按預定義的順序獲取分數，對於缺失的 Blendshape（理論上不應該有），填充 0.0
            extracted_scores: List[float] = [blendshape_dict.get(name, 0.0) for name in BLENDSHAPE_NAMES]
            
            # 簡單校驗提取到的 Blendshape 數量是否正確
            if len(extracted_scores) != len(BLENDSHAPE_NAMES):
                print(f"警告：{image_path} 的 Blendshape 數量不匹配。預期 {len(BLENDSHAPE_NAMES)}，實際 {len(extracted_scores)}。")
                return None
                
            return extracted_scores
        else:
            # 如果沒有偵測到人臉地標點或沒有 Blendshape 輸出
            # 記錄到 missing_blendshapes.txt 中
            with open('missing_blendshapes.txt', 'a') as f:
                f.write(f"{image_path}\n")
            return None
    except Exception as e:
        print(f"處理圖像 {image_path} 時發生錯誤：{e}")
        return None

def process_subset(subset_name: str, landmarker_processor: vision.FaceLandmarker) -> Optional[str]:
    """
    處理一個子集 (train/test/val) 的所有圖片並提取 Blendshape。
    
    Args:
        subset_name: 子集名稱 ('train', 'test', 或 'val')
        landmarker_processor: MediaPipe Face Landmarker 處理器
        
    Returns:
        Optional[str]: 輸出 CSV 檔案的路徑，如果處理成功則返回 None
    """
    subset_path: str = os.path.join(DATASET_ROOT_DIR, 'dataset', subset_name)
    
    blendshape_features_list: List[List[float]] = []
    labels_list: List[int] = []
    skipped_count: int = 0
    total_images_in_subset: int = 0 # 紀錄此子集總共有多少張圖片

    print(f"\n--- 開始處理 {subset_name} 資料集 ---")

    # 遍歷每個情緒資料夾
    for emotion_name in sorted(os.listdir(subset_path)):
        if emotion_name not in EMOTION_TO_LABEL:
            print(f"警告：未知情緒資料夾 '{emotion_name}' 在 {subset_name} 中，已跳過。")
            continue
        
        emotion_label: int = EMOTION_TO_LABEL[emotion_name]
        emotion_folder_path: str = os.path.join(subset_path, emotion_name)
        
        # 獲取所有圖片檔案 (支援 .jpg 和 .png)
        image_paths: List[str] = glob.glob(os.path.join(emotion_folder_path, '*.jpg')) + \
                                 glob.glob(os.path.join(emotion_folder_path, '*.png'))
        
        total_images_in_subset += len(image_paths) # 累計該子集所有圖片數量

        # 測試模式：每個情緒類別只取前 10 張圖片
        image_paths = image_paths[:] # 每個情緒類別只取前 10 張圖片進行測試
        print(f"處理 {emotion_name} 類別的前 {len(image_paths)} 張圖片") # 測試模式下的提示

        # 使用 tqdm 顯示每個情緒資料夾的進度條
        for image_path in tqdm(image_paths, desc=f"  {subset_name}/{emotion_name}"):
            extracted_scores = extract_blendshapes_from_image(image_path, landmarker_processor)

            if extracted_scores is not None:
                blendshape_features_list.append(extracted_scores)
                labels_list.append(emotion_label)
            else:
                skipped_count += 1
    
    if not blendshape_features_list:
        print(f"沒有為 {subset_name} 提取到任何 Blendshape 資料。請檢查圖像品質和 MediaPipe 檢測置信度。")
        return None

    blendshape_df: pd.DataFrame = pd.DataFrame(blendshape_features_list, columns=BLENDSHAPE_NAMES)
    blendshape_df['emotion'] = labels_list
    
    output_csv_path: str = os.path.join(OUTPUT_CSV_DIR, f'{subset_name}_blendshapes.csv')
    blendshape_df.to_csv(output_csv_path, index=False)

    print(f"\n成功從 {total_images_in_subset} 張圖片中提取了 {len(blendshape_features_list)} 張圖片的 Blendshape 資料用於 {subset_name}。")
    print(f"由於 MediaPipe 未偵測到人臉或處理失敗，跳過了 {skipped_count} 張圖片。")
    print(f"{subset_name} Blendshape 資料已儲存到：{output_csv_path}")
    return output_csv_path

def main() -> None:
    """
    主程式入口點。
    """
    print(f"資料集根目錄：{DATASET_ROOT_DIR}")
    print(f"輸出 CSV 目錄：{OUTPUT_CSV_DIR}")
    
    # 檢查模型檔案是否存在
    if not os.path.exists(MODEL_PATH):
        print(f"錯誤：模型檔案未找到：{MODEL_PATH}")
        print("請確認已下載 'face_landmarker_v2_with_blendshapes.task' 並放置於正確路徑。")
        print("您可以從以下連結下載：https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task")
        return

    subsets_to_process: List[str] = ['train', 'test', 'val'] # 根據您的資料夾結構調整
    
    output_files: Dict[str, str] = {}
    for subset in subsets_to_process:
        output_path = process_subset(subset, face_landmarker) 
        if output_path:
            output_files[subset] = output_path
            
    # 在所有處理完成後，可以選擇關閉 Landmarker。
    # 對於批次處理圖像，通常會保持開啟直到所有處理結束。
    # face_landmarker.close() 

    print("\n--- 所有資料提取完成 ---")
    for subset, path in output_files.items():
        print(f"  {subset} 資料儲存在：{path}")

if __name__ == "__main__":
    main()