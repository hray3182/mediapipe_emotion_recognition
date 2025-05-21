import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, regularizers

# --- 設定 ---
EXTRACTED_BLENDSHAPES_DIR = '/home/ray/program/mediapipe_emotion_recognition/extracted_blendshapes'
DATASET_ROOT_DIR = '/home/ray/program/mediapipe_emotion_recognition'

TRAIN_CSV = os.path.join(EXTRACTED_BLENDSHAPES_DIR, 'train_blendshapes.csv')
TEST_CSV = os.path.join(EXTRACTED_BLENDSHAPES_DIR, 'test_blendshapes.csv')
VAL_CSV = os.path.join(EXTRACTED_BLENDSHAPES_DIR, 'val_blendshapes.csv')

# 輸出模型檔案的目錄
MODEL_SAVE_DIR = os.path.join(DATASET_ROOT_DIR, 'models')
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# 定義情緒名稱到數字標籤的對應 (與提取時保持一致)
EMOTION_TO_LABEL = {
    'Angry': 0, 'Disgust': 1, 'Fear': 2, 'Happy': 3,
    'Neutral': 4, 'Sad': 5, 'Surprise': 6
}
NUM_CLASSES = len(EMOTION_TO_LABEL) # 總共有 7 種情緒

# MediaPipe Face Landmarker 提供的 52 個 Blendshape 名稱
BLENDSHAPE_NAMES = [
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

# --- 載入數據集 ---
def load_and_preprocess_data(file_path):
    print(f"正在載入數據: {file_path}")
    df = pd.read_csv(file_path)
    X = df[BLENDSHAPE_NAMES].values # 特徵 (Blendshape 數值)
    y = df['emotion'].values        # 標籤 (情緒數字)
    return X, y

# 載入所有數據集
X_train, y_train = load_and_preprocess_data(TRAIN_CSV)
X_test, y_test = load_and_preprocess_data(TEST_CSV)
X_val, y_val = load_and_preprocess_data(VAL_CSV)

print(f"訓練集形狀: X={X_train.shape}, y={y_train.shape}")
print(f"測試集形狀: X={X_test.shape}, y={y_test.shape}")
print(f"驗證集形狀: X={X_val.shape}, y={y_val.shape}")

# --- 數據預處理 ---
# 1. 特徵標準化 (使用 StandardScaler)
# StandardScaler 在訓練集上 fit，然後 transform 所有數據集
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_val_scaled = scaler.transform(X_val)

# 2. 標籤 One-Hot 編碼 (使用 OneHotEncoder)
# OneHotEncoder 在所有可能的類別上 fit，然後 transform 所有數據集
encoder = OneHotEncoder(sparse_output=False, categories='auto')
# 由於情緒標籤是數字 0-6，我們可以先將它們合併並 fit 一次確保所有類別都被識別
all_labels = np.concatenate((y_train, y_test, y_val)).reshape(-1, 1)
encoder.fit(all_labels)

y_train_encoded = encoder.transform(y_train.reshape(-1, 1))
y_test_encoded = encoder.transform(y_test.reshape(-1, 1))
y_val_encoded = encoder.transform(y_val.reshape(-1, 1))

print(f"One-Hot 編碼後訓練集標籤形狀: {y_train_encoded.shape}")

# --- 構建模型 ---
# 這裡我們構建一個簡單的全連接神經網絡 (MLP)
# 輸入層的維度是 Blendshape 的數量 (52)
# 輸出層的維度是情緒類別的數量 (7)
input_shape = (X_train_scaled.shape[1],) # 52

# 設定學習率
learning_rate = 0.001  # Adam 優化器的預設學習率
print(f"\n使用的學習率: {learning_rate}")

model = models.Sequential([
    layers.Input(shape=input_shape),
    layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001)), # 添加 L2 正則化
    layers.Dropout(0.3), # 添加 Dropout 防止過擬合
    layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    layers.Dropout(0.3),
    layers.Dense(NUM_CLASSES, activation='softmax') # 分類問題使用 softmax 激活函數
])

# 編譯模型
# optimizer: 'adam' 是常用的選擇
# loss: 'categorical_crossentropy' 用於多分類 One-Hot 編碼標籤
# metrics: 'accuracy' 用於評估準確度
model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# --- 模型訓練 ---
# 定義回調函數
callbacks = [
    keras.callbacks.EarlyStopping(patience=10, monitor='val_loss', restore_best_weights=True), # 早停，如果驗證損失連續 10 個 epoch 沒有改善，則停止
    keras.callbacks.ModelCheckpoint(filepath=os.path.join(MODEL_SAVE_DIR, 'best_emotion_model.h5'),
                                    monitor='val_loss',
                                    save_best_only=True) # 保存最佳模型
]

print("\n--- 開始訓練模型 ---")
history = model.fit(
    X_train_scaled, y_train_encoded,
    epochs=100, # 可以設定更多 epochs，早停會控制訓練時長
    batch_size=64, # 調整批次大小
    validation_data=(X_val_scaled, y_val_encoded),
    callbacks=callbacks,
    verbose=1
)

# --- 模型評估 ---
print("\n--- 評估模型性能 ---")
# 載入最佳模型進行評估
best_model = models.load_model(os.path.join(MODEL_SAVE_DIR, 'best_emotion_model.h5'))

loss, accuracy = best_model.evaluate(X_test_scaled, y_test_encoded, verbose=0)
print(f"測試集損失: {loss:.4f}")
print(f"測試集準確度: {accuracy:.4f}")

# 生成更詳細的評估報告
y_pred_probs = best_model.predict(X_test_scaled)
y_pred = np.argmax(y_pred_probs, axis=1) # 轉換為類別索引
y_true = np.argmax(y_test_encoded, axis=1) # 轉換為類別索引

print("\n分類報告:")
print(classification_report(y_true, y_pred, target_names=list(EMOTION_TO_LABEL.keys())))

print("\n混淆矩陣:")
print(confusion_matrix(y_true, y_pred))

# --- 模型保存 ---
# 最佳模型已經通過 ModelCheckpoint 保存了
print(f"\n最佳模型已保存到: {os.path.join(MODEL_SAVE_DIR, 'best_emotion_model.h5')}")

# 你也可以保存最終模型 (如果不需要最佳模型)
# model.save(os.path.join(MODEL_SAVE_DIR, 'final_emotion_model.h5'))
# print(f"最終模型已保存到: {os.path.join(MODEL_SAVE_DIR, 'final_emotion_model.h5')}")