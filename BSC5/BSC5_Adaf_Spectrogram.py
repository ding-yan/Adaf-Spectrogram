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

# data paths
#folder_path = 'C:\\Users\\User\\Desktop\\Dataset\\bird song data set\\selected_wavfiles_label'
folder_path = './data/selected_wavfiles_label'
output_folder = os.path.join(folder_path, 'BSC5_Adaf_Spectrograms_128(Power)')
# create output directory
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

accumulated_frequency_energy = None
sampling_rate = 22050  # 22050 samples per second
nperseg = 2048  # length of each window
noverlap = 1536  # window overlap, 2048-512=1536
size=(128, 128)

try:
    # process each file
    for filename in os.listdir(folder_path):
        if filename.endswith(".wav"):
            filepath = os.path.join(folder_path, filename)
            
            try:
                # read audio data
                samplerate,audio_data = wavfile.read(filepath)

                if samplerate != sampling_rate:
                    print(f"Warning: {filename} sample rate is {samplerate}, expected {sampling_rate}. Skipping.")
                    continue

                # calculate STFT
                frequencies, times, spectrogram = stft(audio_data, fs=sampling_rate, nperseg=nperseg, noverlap=noverlap)

                # calculate frequency energy
                frequency_energy = np.sum(np.abs(spectrogram)**2, axis=1)
                
                # accumulate frequency energy
                if accumulated_frequency_energy is None:
                    accumulated_frequency_energy = frequency_energy
                else:
                    accumulated_frequency_energy += frequency_energy

            except Exception as e:
                print(f"Error processing {filepath}: {e}")
                continue


    print(frequencies)

    # divide into sections based on frequency energy distribution
    total_energy = np.sum(accumulated_frequency_energy)
    energy_proportions =  accumulated_frequency_energy / total_energy  # energy proportion of each frequency
    cumulative_energy = np.cumsum(energy_proportions)

    frequency_energy_list = [(frequencies[i],  accumulated_frequency_energy[i]) for i in range(len(frequencies))]

    # display each frequency and its energy proportion
    for freq, energy in frequency_energy_list:
        proportion = (energy / total_energy) * 100
        print(f'frequency {freq:.2f} Hz, energy: {energy:.2f}, energy proportion: {proportion:.2f}%')
    # print(cumulative_energy)

    # set number of sections
    num_sections = 128 
    section_indices = np.linspace(0, 1, num_sections + 1)

    frequency_ranges = []

    for i in range(num_sections):
        lower_bound = np.searchsorted(cumulative_energy, section_indices[i])
        upper_bound = np.searchsorted(cumulative_energy, section_indices[i + 1])
        
        # ensure upper_bound is not less than lower_bound, avoid duplicate
        if upper_bound <= lower_bound:
            upper_bound = lower_bound + 1
        
        frequency_ranges.append((frequencies[lower_bound], frequencies[upper_bound - 1]))

    # ensure the last section can contain all remaining frequencies
    frequency_ranges[-1] = (frequency_ranges[-1][0], frequencies[-1])

    # generate adaptive spectrogram based on the same frequency range
    for filename in os.listdir(folder_path):
        if filename.endswith(".wav"):
            filepath = os.path.join(folder_path, filename)
            
            try:
                # read audio data
                samplerate,audio_data = wavfile.read(filepath)

                if samplerate != sampling_rate:
                    print(f"Warning: {filename} sample rate is {samplerate}, expected {sampling_rate}. Skipping.")
                    continue

                # calculate STFT
                frequencies, times, spectrogram = stft(audio_data, fs=sampling_rate, nperseg=nperseg, noverlap=noverlap)

                # create adaptive spectrogram and calculate total energy of each section
                adaptive_spectrogram = np.zeros((num_sections, len(times)))
                section_energy_totals = np.zeros(num_sections)

                for i in range(num_sections):
                    lower_freq, upper_freq = frequency_ranges[i]
                    freq_indices = np.where((frequencies >= lower_freq) & (frequencies <= upper_freq))[0]
                    
                    if len(freq_indices) > 0:
                        section_energy_totals[i] = np.sum(accumulated_frequency_energy [freq_indices])
                        adaptive_spectrogram[i] = np.sum(np.abs(spectrogram[freq_indices, :])**2, axis=0)

                # S_db_1 = librosa.amplitude_to_db(adaptive_spectrogram, ref=np.max)

                # plot
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




    # calculate energy percentage of each section
    section_energy_percentages = (section_energy_totals / total_energy) * 100

    # display each section's Hz range and total energy
    for i, (low, high) in enumerate(frequency_ranges):
        print(f'section {i + 1}: {low:.2f} Hz - {high:.2f} Hz, total energy: {section_energy_totals[i]:.2f}, percentage: {section_energy_percentages[i]:.2f}%')

    # save information to TXT
    output_path = f'Adaf_frequency_section_{num_sections}_info_power.txt'
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(f"sampling_rate: {sampling_rate}\n")
        f.write(f"nperseg: {nperseg}\n")
        f.write(f"noverlap: {noverlap}\n") 
        # write frequencies axis first
        f.write("=== Frequencies ===\n")
        for freq in frequencies:
            f.write(f"{freq:} Hz\n")

        f.write("\n=== frequency ranges of each section ===\n")
        for i, (low, high) in enumerate(frequency_ranges):
            f.write(f"section {i + 1}: {low:} Hz - {high:} Hz\n")

    print(f"saved to {output_path}")

except Exception as e:
    print(f"Error processing {filepath}: {e}")

print("completed")