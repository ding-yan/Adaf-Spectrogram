import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import stft
import os
from PIL import Image
from scipy.io import wavfile
import librosa
import librosa.display

# 資料夾路徑
folder_path = 'C:\\Users\\User\\Desktop\\Dataset\\bird song data set\\selected_wavfiles_label'
output_folder = os.path.join(folder_path, 'BSC5_Adaf_Spectrograms_128(Power)')
# 如果儲存圖片的資料夾不存在，則創建
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

accumulated_frequency_energy = None
sampling_rate = 22050  # 每秒22050個樣本
nperseg = 2048  # 每個窗口的長度
noverlap = 1536  # 窗口重疊數 2048-512=1536
size=(128, 128)

try:
    # 遍歷資料夾內的所有 wav 檔案
    for filename in os.listdir(folder_path):
        if filename.endswith(".wav"):
            filepath = os.path.join(folder_path, filename)
            
            try:
                # 讀取音訊數據
                samplerate,audio_data = wavfile.read(filepath)
                
                # 計算 STFT
                frequencies, times, spectrogram = stft(audio_data, fs=sampling_rate, nperseg=nperseg, noverlap=noverlap)
                
                # 計算頻率能量
                frequency_energy = np.sum(np.abs(spectrogram)**2, axis=1)
                
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
    # print(cumulative_energy)

    # 設定區段數量
    num_sections = 128 
    section_indices = np.linspace(0, 1, num_sections + 1)

    frequency_ranges = []

    for i in range(num_sections):
        lower_bound = np.searchsorted(cumulative_energy, section_indices[i])
        upper_bound = np.searchsorted(cumulative_energy, section_indices[i + 1])
        
        # 確保 upper_bound 不小於 lower_bound，避免重複
        if upper_bound <= lower_bound:
            upper_bound = lower_bound + 1
        
        frequency_ranges.append((frequencies[lower_bound], frequencies[upper_bound - 1]))

    # 確保最後一個區段能夠包含所有剩餘的頻率
    frequency_ranges[-1] = (frequency_ranges[-1][0], frequencies[-1])

    # 根據相同的頻率範圍生成自適應的頻譜圖
    for filename in os.listdir(folder_path):
        if filename.endswith(".wav"):
            filepath = os.path.join(folder_path, filename)
            
            try:
                # 讀取音訊數據
                samplerate,audio_data = wavfile.read(filepath)

                # 計算 STFT
                frequencies, times, spectrogram = stft(audio_data, fs=sampling_rate, nperseg=nperseg, noverlap=noverlap)

                # 建立自適應頻譜圖和計算每個區段的總能量
                adaptive_spectrogram = np.zeros((num_sections, len(times)))
                section_energy_totals = np.zeros(num_sections)

                for i in range(num_sections):
                    lower_freq, upper_freq = frequency_ranges[i]
                    freq_indices = np.where((frequencies >= lower_freq) & (frequencies <= upper_freq))[0]
                    
                    if len(freq_indices) > 0:
                        section_energy_totals[i] = np.sum(accumulated_frequency_energy [freq_indices])
                        adaptive_spectrogram[i] = np.sum(np.abs(spectrogram[freq_indices, :])**2, axis=0)

                # S_db_1 = librosa.amplitude_to_db(adaptive_spectrogram, ref=np.max)

                # 畫
                plt.figure(figsize=(1.28, 1.28), dpi=100)

                plt.imshow(adaptive_spectrogram[::-1], aspect='auto', cmap='viridis', extent=[times.min(), times.max(), 0, num_sections])
                # plt.imshow(S_db_1[::-1], aspect='auto', cmap='viridis', extent=[times.min(), times.max(), 0, num_sections])
                plt.axis('off')

                save_path = os.path.join(output_folder, f'{filename[:-4]}_temp.png')
                plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=100)
                plt.close()  

                
                img = Image.open(save_path)
                img_resized = img.resize(size, Image.LANCZOS)
                output_file = os.path.join(output_folder, f'{filename[:-4]}.png')
                img_resized.save(output_file)      
               
                os.remove(save_path)
                
                plt.clf()
            
            except Exception as e:
                print(f"Error processing {filepath}: {e}")
                continue




    # 計算每個區段的能量百分比
    section_energy_percentages = (section_energy_totals / total_energy) * 100

    # 顯示每個區段的 Hz 範圍和總能量
    for i, (low, high) in enumerate(frequency_ranges):
        print(f'區段 {i + 1}: {low:.2f} Hz - {high:.2f} Hz, 總能量: {section_energy_totals[i]:.2f}, 百分比: {section_energy_percentages[i]:.2f}%')

    # 儲存資訊到 TXT
    output_path = f'Adaf_frequency_section_{num_sections}_info_power.txt'
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


    # 獲取桌面路徑
    # desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
    # # 頻率能量分佈的條形圖
    # plt.figure(figsize=(40, 4))
    # plt.bar(frequencies, accumulated_frequency_energy, width=0.25, color='b')#0.25
    # plt.title('Frequency Distribution')
    # plt.xlabel('Frequency [Hz]')
    # plt.ylabel('Total Energy')
    # # 保存到桌面
    # output_path = os.path.join(desktop_path, 'sound.png')
    # plt.savefig(output_path)
    # plt.close()
    # plt.show()

except Exception as e:
    print(f"Error processing {filepath}: {e}")

print("completed")