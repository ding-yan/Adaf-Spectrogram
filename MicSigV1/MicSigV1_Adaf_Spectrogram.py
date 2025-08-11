import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.table as tbl
from scipy.signal import stft
import os
from PIL import Image


# 資料夾路徑
folder_path = 'C:\\Users\\User\\Desktop\\Dataset\\MicSigV1\\raw'
output_folder = os.path.join(folder_path, 'MicSigV1_Adaf_Spectrograms_128(Amplitude)')
# 如果儲存圖片的資料夾不存在，則創建
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

accumulated_frequency_energy = None
sampling_rate = 100  # 每秒100個樣本
nperseg = 128  # 每個窗口的長度
noverlap = 64  # 窗口重疊數
size=(128, 128)

try:
    # 遍歷資料夾內的所有 txt 檔案
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            filepath = os.path.join(folder_path, filename)
            
            try:
                # 讀取數據
                with open(filepath, 'r') as file:
                    lines = file.readlines()
                
                data = [float(line.strip()) for line in lines]
                
                # 計算 STFT
                frequencies, times, spectrogram = stft(data, fs=sampling_rate, nperseg=nperseg, noverlap=noverlap)
                
                # 計算頻率能量
                frequency_energy = np.sum(np.abs(spectrogram), axis=1)      # Amplitude
                
                # 累加頻率能量
                if accumulated_frequency_energy is None:
                    accumulated_frequency_energy = frequency_energy
                else:
                    accumulated_frequency_energy += frequency_energy

            except Exception as e:
                print(f"Error processing {filepath}: {e}")
                continue


    print(frequencies)


    # 根據頻率能量分佈劃分區段
    total_energy = np.sum(accumulated_frequency_energy)
    energy_proportions =  accumulated_frequency_energy / total_energy  # 每個頻率的能量占比
    cumulative_energy = np.cumsum(energy_proportions)


    frequency_energy_list = [(frequencies[i],  accumulated_frequency_energy[i]) for i in range(len(frequencies))]

    # 顯示每個頻率及其能量占比
    for freq, energy in frequency_energy_list:
        proportion = (energy / total_energy) * 100
        print(f'頻率 {freq:.2f} Hz, 能量: {energy:.2f}, 能量占比: {proportion:.2f}%')
    

    # 設定區段數量
    num_sections = 128 
    section_indices = np.linspace(0, 1, num_sections + 1)

    frequency_ranges = []

    for i in range(num_sections):
        lower_bound = np.searchsorted(cumulative_energy, section_indices[i])
        upper_bound = np.searchsorted(cumulative_energy, section_indices[i + 1])
        
        # 確保 upper_bound 不小於 lower_bound，避免重複
        if upper_bound == lower_bound:
            upper_bound = lower_bound + 1
        
        frequency_ranges.append((frequencies[lower_bound], frequencies[upper_bound - 1]))

    # 確保最後一個區段能夠包含所有剩餘的頻率
    frequency_ranges[-1] = (frequency_ranges[-1][0], frequencies[-1])

    # 遍歷資料夾內的所有 txt 檔案，根據相同的頻率範圍生成自適應的頻譜圖
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            filepath = os.path.join(folder_path, filename)
            
            try:
                # 讀取數據
                with open(filepath, 'r') as file:
                    lines = file.readlines()

                data = [float(line.strip()) for line in lines]
                
                # 計算 STFT
                frequencies, times, spectrogram = stft(data, fs=sampling_rate, nperseg=nperseg, noverlap=noverlap)

                # 建立自適應頻譜圖和計算每個區段的總能量
                adaptive_spectrogram = np.zeros((num_sections, len(times)))
                section_energy_totals = np.zeros(num_sections)

                for i in range(num_sections):
                    lower_freq, upper_freq = frequency_ranges[i]
                    freq_indices = np.where((frequencies >= lower_freq) & (frequencies <= upper_freq))[0]
                    
                    if len(freq_indices) > 0:
                        section_energy_totals[i] = np.sum(accumulated_frequency_energy [freq_indices])
                        adaptive_spectrogram[i] = np.sum(np.abs(spectrogram[freq_indices, :]), axis=0)      # Amplitude-Sum
                        # adaptive_spectrogram[i] = np.mean(np.abs(spectrogram[freq_indices, :]), axis=0)   # Amplitude-Mean
                        # adaptive_spectrogram[i] = np.max(np.abs(spectrogram[freq_indices, :]), axis=0)    # Amplitude-Max
                
                # 繪製自適應縮放後的百分位頻譜圖
                plt.figure(figsize=(1.28, 1.28), dpi=100)
                

                plt.imshow(adaptive_spectrogram[::-1], aspect='auto', cmap='viridis', extent=[times.min(), times.max(), 0, num_sections])
                plt.axis('off')

                # 儲存圖片
                save_path = os.path.join(output_folder, f'{filename[:-4]}_temp.png')
                plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=100)
                plt.close()  

                # 重新調整圖片大小
                img = Image.open(save_path)
                img_resized = img.resize(size, Image.LANCZOS)
                output_file = os.path.join(output_folder, f'{filename[:-4]}.png')
                img_resized.save(output_file)      
                # 刪除暫時圖片
                os.remove(save_path)
                # 清除當前圖形，釋放記憶體
                plt.clf()
            
            except Exception as e:
                print(f"Error processing {filepath}: {e}")
                continue




    #計算每個區段的能量百分比
    section_energy_percentages = (section_energy_totals / total_energy) * 100

    # 顯示每個區段的 Hz 範圍和總能量
    for i, (low, high) in enumerate(frequency_ranges):
        print(f'區段 {i + 1}: {low:.2f} Hz - {high:.2f} Hz, 總能量: {section_energy_totals[i]:.2f}, 百分比: {section_energy_percentages[i]:.2f}%')
    
    # 儲存資訊到 TXT
    output_path = f'Adaf_frequency_section_{num_sections}_info.txt'
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(f"sampling_rate: {sampling_rate}\n")
        f.write(f"nperseg: {nperseg}\n")
        f.write(f"noverlap: {noverlap}\n") 
        # 先寫入 frequencies 頻率軸
        f.write("=== Frequencies ===\n")
        for freq in frequencies:
            f.write(f"{freq:} Hz\n")

        f.write("\n=== 各區段頻率範圍 ===\n")
        for i, (low, high) in enumerate(frequency_ranges):
            f.write(f"區段 {i + 1}: {low:} Hz - {high:} Hz\n")

    print(f"已儲存至 {output_path}")


    # # 獲取桌面路徑
    # desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
    # # 繪製頻率能量分佈的條形圖
    # plt.figure(figsize=(10, 4))
    # plt.bar(frequencies, accumulated_frequency_energy, width=0.25, color='b')
    # plt.title('MicSigV1 Dataset Frequency Distribution')
    # plt.xlabel('Frequency Bin [Hz]')
    # plt.ylabel('Total Energy')
    # # 保存到桌面
    # output_path = os.path.join(desktop_path, 'MicSigV1 Frequency Distribution.pdf')
    # plt.savefig(output_path, format='pdf')
    # plt.close()
    # plt.show()
except Exception as e:
    print(f"Error processing {filepath}: {e}")

print("completed")