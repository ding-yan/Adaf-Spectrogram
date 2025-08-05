import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import librosa
import librosa.display
import os
from PIL import Image

# 定義轉換函數
def convert_txt_to_spectrogram(txt_path, output_path, sr=100, n_mels=64, n_fft=128, hop_length=64, size=(128, 128)):
    try:
        # 讀取數據
        with open(txt_path, 'r') as file:
            lines = file.readlines()

        # 提取數據
        data = [float(line.strip()) for line in lines]
        data = np.array(data)


        # 計算 Mel 頻譜圖
        S = librosa.feature.melspectrogram(y=data, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)
        S_dB = librosa.power_to_db(S, ref=np.max)

        # 顯示 Mel 頻譜圖
        fig, ax = plt.subplots(figsize=(size[0] / 100, size[1] / 100), dpi=100)
        img = librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr, fmax=sr / 2, ax=ax )
        # 移除坐標軸
        ax.axis('off')
        # 儲存圖像為 PNG 文件
        plt.savefig('temp.png', bbox_inches='tight', pad_inches=0, dpi=100)
        plt.close()
        # 讀取圖像並調整大小
        with Image.open('temp.png') as img:
            img = img.resize(size, Image.LANCZOS)
            img.save(output_path)

        # 刪除臨時文件
        os.remove('temp.png')
    except Exception as e:
        print(f"Error processing {txt_path}: {e}")

# 設定資料夾路徑
base_folder = 'raw'
output_folder = 'MicSigV1_Mel_Spectrograms_64'

# 創建輸出資料夾（如果不存在）
os.makedirs(output_folder, exist_ok=True)

# 遍歷資料夾中的所有 TXT 文件
for file_name in os.listdir(base_folder):
    if file_name.endswith('.txt'):
        txt_path = os.path.join(base_folder, file_name)
        output_path = os.path.join(output_folder, f"{os.path.splitext(file_name)[0]}.png")
        convert_txt_to_spectrogram(txt_path, output_path)

print("Conversion completed.")
