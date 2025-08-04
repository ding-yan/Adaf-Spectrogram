import pandas as pd
import shutil
import os

# 路徑設定
csv_path = 'C:\\Users\\User\\Desktop\\Dataset\\bird song data set\\bird_songs_metadata.csv'
wav_src_folder = 'C:\\Users\\User\Desktop\\Dataset\\bird song data set\\wavfiles'
wav_dst_folder = 'C:\\Users\\User\Desktop\\Dataset\\bird song data set\\selected_wavfiles'

# 建立目標資料夾（如果不存在）
os.makedirs(wav_dst_folder, exist_ok=True)

# 讀取 CSV 並取得 filename 欄位
df = pd.read_csv(csv_path)
filenames = df['filename'].dropna().unique()

# 複製檔案
for fname in filenames:
    src_file = os.path.join(wav_src_folder, fname)
    dst_file = os.path.join(wav_dst_folder, fname)
    if os.path.exists(src_file):
        shutil.copy2(src_file, dst_file)
    else:
        print(f"找不到檔案：{fname}")

print("檔案複製完成。")