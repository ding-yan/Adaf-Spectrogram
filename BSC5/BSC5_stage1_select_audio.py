import pandas as pd
import shutil
import os
import sys

# path settings
#csv_path = 'C:\\Users\\User\\Desktop\\Dataset\\bird song data set\\bird_songs_metadata.csv'
csv_path = './data/bird_songs_metadata.csv'
#wav_src_folder = 'C:\\Users\\User\Desktop\\Dataset\\bird song data set\\wavfiles'
wav_src_folder = './data/wavfiles'
#wav_dst_folder = 'C:\\Users\\User\Desktop\\Dataset\\bird song data set\\selected_wavfiles'
wav_dst_folder = './data/selected_wavfiles'

# create target directory
os.makedirs(wav_dst_folder, exist_ok=True)

# read csv and obtain the 'filename' field
df = pd.read_csv(csv_path)
filenames = df['filename'].dropna().unique()

# copy files
for fname in filenames:
    src_file = os.path.join(wav_src_folder, fname)
    dst_file = os.path.join(wav_dst_folder, fname)
    if os.path.exists(src_file):
        shutil.copy2(src_file, dst_file)
    else:
        print(f"Cannot find file：{fname}")
        sys.exit(1)

print("File copy completed")
