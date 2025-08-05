import os
import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft
from PIL import Image

# 設定資料夾路徑
folder_path = 'C:\\Users\\User\\Desktop\\Dataset\\MicSigV1'

# 設定參數
sampling_rate = 100     # 每秒100個樣本
nperseg = 128           # 每個窗口的長度
noverlap = 64           # 窗口重疊數，50% 重疊
output_size = (128, 128)  # 期望輸出的圖片大小

# 建立輸出資料夾
output_folder = os.path.join(folder_path, 'MicSigV1_Spectrograms(Power)')
os.makedirs(output_folder, exist_ok=True)

#
for subfolder in os.listdir(folder_path):
    subfolder_path = os.path.join(folder_path, subfolder)
    
    if os.path.isdir(subfolder_path):
        for filename in os.listdir(subfolder_path):
            if filename.endswith('.txt'):
                file_path = os.path.join(subfolder_path, filename)
                
                try:
                    # 讀取數據
                    with open(file_path, 'r') as file:
                        lines = file.readlines()
                        
                    data = [float(line.strip()) for line in lines] 
                    
                    # 計算 STFT
                    frequencies, times, spectrogram = stft(data, fs=sampling_rate, nperseg=nperseg, noverlap=noverlap)
                    
                    # 畫圖並保存
                    plt.figure()
        
                    # plt.imshow(np.abs(spectrogram), aspect='auto', origin='lower', cmap='viridis')        #(Amplitude)
                    plt.imshow(np.abs(spectrogram)**2, aspect='auto', origin='lower', cmap='viridis')       #(Power)
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
