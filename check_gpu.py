import tensorflow as tf

def check_gpu():
    print("TensorFlow 版本:", tf.__version__)
    
    # 使用 tf.config.list_physical_devices 檢查 GPU
    gpus = tf.config.list_physical_devices('GPU')
    
    if gpus:
        print("\n找到以下 GPU 設備:")
        for gpu in gpus:
            print(f"- {gpu}")
    else:
        print("\n未找到可用的 GPU 設備")

if __name__ == "__main__":
    check_gpu() 