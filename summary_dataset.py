import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # 用於3D繪圖

# --- 設定 (請根據你的實際路徑更新) ---
EXTRACTED_LANDMARKS_DIR = '/home/ray/program/mediapipe_emotion_recognition/extracted_face_mesh' # 你的數據目錄
TRAIN_CSV = os.path.join(EXTRACTED_LANDMARKS_DIR, 'train_face_mesh.csv')

# MediaPipe Face Mesh 提供的 478 個 3D 關鍵點坐標名稱
FEATURE_COLUMN_NAMES = [f"{coord}_{i}" for i in range(478) for coord in ['x', 'y', 'z']]

# 定義用於自定義歸一化的關鍵點索引
# 這些索引根據 MediaPipe Face Mesh 的標準定義圖
NOSE_TIP_IDX = 1          # 鼻尖
LEFT_EYE_OUTER_IDX = 133  # 左眼外眼角
RIGHT_EYE_OUTER_IDX = 362 # 右眼外眼角

# --- 自定義數據一致化函數 (與你訓練腳本中的相同) ---
def custom_normalize_landmarks(X_raw):
    """
    對原始關鍵點數據進行平移和尺度歸一化。
    X_raw: numpy array, 原始關鍵點數據，形狀為 (樣本數, 478 * 3)
    """
    X_normalized = np.zeros_like(X_raw, dtype=np.float32)
    num_landmarks = 478

    for i in range(X_raw.shape[0]):
        landmarks_flat = X_raw[i]
        landmarks_3d = landmarks_flat.reshape(num_landmarks, 3)

        # 獲取鼻尖坐標
        try:
            nose_tip = landmarks_3d[NOSE_TIP_IDX]
            nose_tip_x, nose_tip_y, nose_tip_z = nose_tip[0], nose_tip[1], nose_tip[2]
        except IndexError:
            # 確保索引存在
            # 如果索引無效，將該樣本數據歸零並跳過
            X_normalized[i] = np.zeros_like(landmarks_flat)
            continue

        # 獲取兩眼外眼角坐標
        try:
            left_eye_outer = landmarks_3d[LEFT_EYE_OUTER_IDX]
            right_eye_outer = landmarks_3d[RIGHT_EYE_OUTER_IDX]
            
            D_face = np.sqrt(
                (left_eye_outer[0] - right_eye_outer[0])**2 +
                (left_eye_outer[1] - right_eye_outer[1])**2 +
                (left_eye_outer[2] - right_eye_outer[2])**2
            )

            if D_face < 1e-6: # 避免除以零或過小的數
                # 如果距離過小，將該樣本數據歸零並跳過
                X_normalized[i] = np.zeros_like(landmarks_flat)
                continue

        except IndexError:
            # 如果索引無效，將該樣本數據歸零並跳過
            X_normalized[i] = np.zeros_like(landmarks_flat)
            continue

        # 對所有關鍵點進行平移和尺度歸一化
        for j in range(num_landmarks):
            translated_x = landmarks_3d[j, 0] - nose_tip_x
            translated_y = landmarks_3d[j, 1] - nose_tip_y
            translated_z = landmarks_3d[j, 2] - nose_tip_z

            X_normalized[i, j*3 + 0] = translated_x / D_face
            X_normalized[i, j*3 + 1] = translated_y / D_face
            X_normalized[i, j*3 + 2] = translated_z / D_face
            
    return X_normalized

# --- 載入數據集並進行自定義歸一化 ---
def load_data_with_custom_normalization(file_path):
    print(f"正在載入數據: {file_path}")
    df = pd.read_csv(file_path)
    X_raw = df[FEATURE_COLUMN_NAMES].values
    y = df['emotion'].values
    
    X_processed = custom_normalize_landmarks(X_raw)

    # 過濾掉由於異常檢測導致的歸零樣本
    valid_indices = ~np.all(X_processed == 0, axis=1)
    if not np.all(valid_indices):
        print(f"從 {file_path} 中移除了 {np.sum(~valid_indices)} 個無效樣本。")
        X_processed = X_processed[valid_indices]
        y = y[valid_indices]
        X_raw = X_raw[valid_indices] # 同步過濾原始數據

    return X_processed, y, X_raw # 返回原始數據（已過濾無效樣本）以便比較

# 載入並處理訓練集數據
X_train_processed, y_train, X_train_raw_filtered = load_data_with_custom_normalization(TRAIN_CSV)

print(f"處理後訓練集形狀: X={X_train_processed.shape}, y={y_train.shape}")


# --- 視覺化歸一化後的結果 ---

def plot_landmarks(landmarks_3d, title, ax=None, color='blue'):
    """
    繪製面部關鍵點。
    landmarks_3d: 形狀為 (478, 3) 的關鍵點坐標
    """
    if ax is None:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        show_plot = True
    else:
        show_plot = False

    x = landmarks_3d[:, 0]
    y = landmarks_3d[:, 1]
    z = landmarks_3d[:, 2]

    ax.scatter(x, y, z, s=5, c=color) # s=5 是點的大小
    ax.set_title(title)
    ax.set_xlabel("X (Width)")
    ax.set_ylabel("Y (Height)")
    ax.set_zlabel("Z (Depth)")
    ax.set_aspect('equal', adjustable='box') # 保持坐標軸比例一致

    if show_plot:
        plt.show()

def plot_2d_projections(landmarks_3d, title, fig, subplot_base, color='blue'):
    """
    繪製面部關鍵點的 2D 投影。
    """
    x = landmarks_3d[:, 0]
    y = landmarks_3d[:, 1]
    z = landmarks_3d[:, 2]

    # X-Y 平面投影
    ax_xy = fig.add_subplot(subplot_base + 1)
    ax_xy.scatter(x, y, s=5, c=color)
    ax_xy.set_title(f"{title} (X-Y Projection)")
    ax_xy.set_xlabel("X")
    ax_xy.set_ylabel("Y")
    ax_xy.set_aspect('equal', adjustable='box')

    # X-Z 平面投影
    ax_xz = fig.add_subplot(subplot_base + 2)
    ax_xz.scatter(x, z, s=5, c=color)
    ax_xz.set_title(f"{title} (X-Z Projection)")
    ax_xz.set_xlabel("X")
    ax_xz.set_ylabel("Z")
    ax_xz.set_aspect('equal', adjustable='box')

    # Y-Z 平面投影
    ax_yz = fig.add_subplot(subplot_base + 3)
    ax_yz.scatter(y, z, s=5, c=color)
    ax_yz.set_title(f"{title} (Y-Z Projection)")
    ax_yz.set_xlabel("Y")
    ax_yz.set_ylabel("Z")
    ax_yz.set_aspect('equal', adjustable='box')


# 選擇第一個樣本進行視覺化
sample_idx = 0
if X_train_processed.shape[0] > sample_idx:
    # 獲取原始樣本 (過濾後的)
    raw_sample_3d = X_train_raw_filtered[sample_idx].reshape(478, 3)
    # 獲取處理後的樣本
    processed_sample_3d = X_train_processed[sample_idx].reshape(478, 3)
    
    print(f"\n--- 視覺化樣本 {sample_idx} ---")
    print(f"原始樣本 X, Y, Z 範圍: X=[{raw_sample_3d[:,0].min():.4f}, {raw_sample_3d[:,0].max():.4f}], "
          f"Y=[{raw_sample_3d[:,1].min():.4f}, {raw_sample_3d[:,1].max():.4f}], "
          f"Z=[{raw_sample_3d[:,2].min():.4f}, {raw_sample_3d[:,2].max():.4f}]")
    print(f"處理後樣本 X, Y, Z 範圍: X=[{processed_sample_3d[:,0].min():.4f}, {processed_sample_3d[:,0].max():.4f}], "
          f"Y=[{processed_sample_3d[:,1].min():.4f}, {processed_sample_3d[:,1].max():.4f}], "
          f"Z=[{processed_sample_3d[:,2].min():.4f}, {processed_sample_3d[:,2].max():.4f}]")


    # 繪製原始和歸一化數據的 2D 投影進行對比
    fig_compare = plt.figure(figsize=(15, 10))
    fig_compare.suptitle(f'樣本 {sample_idx}: 原始 vs. 歸一化關鍵點投影')

    # 原始數據投影
    plot_2d_projections(raw_sample_3d, '原始', fig_compare, 240, color='red') # 2行4列，從第1個開始
    
    # 歸一化數據投影
    plot_2d_projections(processed_sample_3d, '歸一化', fig_compare, 244, color='blue') # 2行4列，從第5個開始

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # 調整佈局，防止標題重疊
    plt.show()

else:
    print("數據集為空或樣本索引超出範圍，無法視覺化。")
