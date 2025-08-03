import os
import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft
from PIL import Image
from scipy.io import wavfile
from scipy.signal import spectrogram
import librosa
import librosa.display

# 設定資料夾路徑
folder_path = 'C:\\Users\\User\\Desktop\\Dataset\\ESC-50-master'

# 設定參數
sampling_rate = 44100  # 每秒44100個樣本
nperseg = 2048        # 每個窗口的長度
noverlap = 1536       # 窗口重疊數，75% 重疊2048-512=1536
output_size = (128, 128)  # 期望輸出的圖片大小

# 建立輸出資料夾
output_folder = os.path.join(folder_path, 'ESC50_Spectrograms(Power)')
os.makedirs(output_folder, exist_ok=True)

#
for subfolder in os.listdir(folder_path):
    subfolder_path = os.path.join(folder_path, subfolder)
    
    if os.path.isdir(subfolder_path):
        for filename in os.listdir(subfolder_path):
            if filename.endswith('.wav'):
                file_path = os.path.join(subfolder_path, filename)
                
                try:
                    # 讀取音訊數據
                    samplerate,audio_data = wavfile.read(file_path)

                    # 計算 STFT
                    frequencies, times, spectrogram = stft(audio_data, fs=sampling_rate, nperseg=nperseg, noverlap=noverlap)

                    # S_db = librosa.amplitude_to_db(np.abs(spectrogram), ref=np.max)
                    
                    # 畫圖並保存
                    plt.figure()

                    # plt.imshow(np.abs(spectrogram), aspect='auto', origin='lower', cmap='viridis')
                    plt.imshow(np.abs(spectrogram)** 2, aspect='auto', origin='lower', cmap='viridis')
                    # plt.imshow(S_db, aspect='auto', origin='lower', cmap='viridis')
                    plt.axis('off')  # 移除所有軸和標籤
                    
                    # 暫時保存圖片
                    temp_file = os.path.join(output_folder, f'{os.path.splitext(filename)[0]}_temp.png')
                    plt.savefig(temp_file, bbox_inches='tight', pad_inches=0)
                    plt.close()
                    
                    # 重新調整圖片大小
                    img = Image.open(temp_file)
                    img_resized = img.resize(output_size, Image.LANCZOS)
                    output_file = os.path.join(output_folder, f'{os.path.splitext(filename)[0]}.png')
                    img_resized.save(output_file)
                    
                    # 刪除暫時圖片
                    os.remove(temp_file)
                except Exception as e:
                    print(f"Error processing file {filename}: {e}")

print('Spectrograms 生成完畢！')
