# MediaPipe 情緒識別系統

這是一個基於 MediaPipe 和深度學習的情緒識別系統，可以通過網路攝影機即時識別人臉表情和情緒。本專案主要用於學術研究目的。

## 功能特點

- 即時人臉檢測和追蹤
- 情緒識別（支援多種情緒類別）
- 使用 MediaPipe 進行人臉特徵點提取
- 基於深度學習的情緒分類模型
- 即時視覺化顯示

## 系統需求

- Python 3.8 或更高版本
- CUDA 支援的 GPU（推薦）
- 網路攝影機

## 安裝步驟

1. 克隆專案：
```bash
git clone [repository_url]
cd mediapipe_emotion_recognition
```

2. 建立並啟動虛擬環境：
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# 或
.venv\Scripts\activate  # Windows
```

3. 安裝依賴套件：
```bash
pip install -r requirements.txt
```

4. 下載 MediaPipe 模型：
從 [MediaPipe Face Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker) 下載所需的模型文件，並放置在專案根目錄下。

## 資料處理與訓練流程

1. 資料預處理：
首先使用 `extract_blendshapes.py` 將原始資料集轉換為 blendshape 特徵數據：
```bash
python extract_blendshapes.py
```
這將在 `extracted_blendshapes/` 目錄下生成處理後的特徵數據。

2. 模型訓練：
使用處理後的 blendshape 數據進行模型訓練：
```bash
python train_emotion_model.py
```
訓練好的模型將保存在 `models/` 目錄下。

## 測試與驗證

1. 測試網路攝影機：
```bash
python test_webcam.py
```

2. 測試網格視覺化：
```bash
python test_mesh_visualization.py
```

## 專案結構

- `extract_blendshapes.py`: 將原始資料集轉換為 blendshape 特徵數據
- `train_emotion_model.py`: 使用 blendshape 特徵訓練情緒識別模型
- `test_webcam.py`: 網路攝影機測試程式
- `test_mesh_visualization.py`: 網格視覺化測試
- `models/`: 存放訓練好的模型
- `dataset/`: 原始訓練資料集
- `extracted_blendshapes/`: 提取的特徵資料
- `face_landmarker.task`: MediaPipe 人臉特徵點模型文件

## 相關資源

- [MediaPipe Face Landmarker 模型](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker)
- [MediaPipe Face Landmarker 模型說明文件](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker#models)

## 注意事項

- 確保您的系統已正確安裝 CUDA 和 cuDNN（如果使用 GPU）
- 建議使用虛擬環境來避免套件衝突
- 首次運行時需要下載 MediaPipe 模型文件
- 資料處理過程可能需要較長時間，請確保有足夠的磁碟空間
