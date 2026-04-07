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

# data paths
#folder_path = 'C:\\Users\\User\\Desktop\\Dataset\\bird song data set\\selected_wavfiles_label'
folder_path = './data/selected_wavfiles_label'

# parameters
sampling_rate = 22050  # 22050 samples per second
nperseg = 2048        # length of each window
noverlap = 1536       # window overlap, 75% overlap - 2048-512=1536
output_size = (128, 128)  # expected output image size

# create output directory
output_folder = os.path.join(folder_path, 'BSC5_Spectrograms(Power)')
os.makedirs(output_folder, exist_ok=True)

# process each file
for subfolder in os.listdir(folder_path):
    subfolder_path = os.path.join(folder_path, subfolder)
    
    if os.path.isdir(subfolder_path):
        for filename in os.listdir(subfolder_path):
            if filename.endswith('.wav'):
                file_path = os.path.join(subfolder_path, filename)
                
                try:
                    # read audio data
                    samplerate,audio_data = wavfile.read(file_path)

                    # calculate STFT
                    frequencies, times, spectrogram = stft(audio_data, fs=sampling_rate, nperseg=nperseg, noverlap=noverlap)

                    # S_db_1 = librosa.amplitude_to_db(np.abs(spectrogram), ref=np.max)     #(dB)
                    
                    # plot and save
                    plt.figure()

                    # plt.pcolormesh(times, frequencies, np.abs(spectrogram), shading='gouraud')            #(Amplitude)
                    plt.imshow(np.abs(spectrogram)**2, aspect='auto', origin='lower', cmap='viridis')       #(Power)
                    # plt.imshow(S_db_1, aspect='auto', origin='lower', cmap='viridis')                     #(dB)
                    plt.axis('off')  # remove all axes and labels
                    
                    # temporarily save image
                    temp_file = os.path.join(output_folder, f'{os.path.splitext(filename)[0]}_temp.png')
                    plt.savefig(temp_file, bbox_inches='tight', pad_inches=0)
                    plt.close()
                    
                    # resize image
                    img = Image.open(temp_file)
                    img_resized = img.resize(output_size, Image.LANCZOS)
                    output_file = os.path.join(output_folder, f'{os.path.splitext(filename)[0]}.png')
                    img_resized.save(output_file)
                    
                    # delete temporary image
                    os.remove(temp_file)
                except Exception as e:
                    print(f"Error processing file {filename}: {e}")

print('Spectrograms generated!')
