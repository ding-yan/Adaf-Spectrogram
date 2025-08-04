import pandas as pd
import os
import shutil

# 路徑設定
csv_path = 'C:\\Users\\User\\Desktop\\Dataset\\bird song data set\\bird_songs_metadata.csv'
selected_folder = 'C:\\Users\\User\\Desktop\\Dataset\\bird song data set\\selected_wavfiles'
output_folder = 'C:\\Users\\User\\Desktop\\Dataset\\bird song data set\\selected_wavfiles_label'

# 建立輸出資料夾（如果不存在）
os.makedirs(output_folder, exist_ok=True)

# 定義鳥種對應的編號
label_map = {
    "Bewick's Wren": 0,
    "Northern Cardinal": 1,
    "American Robin": 2,
    "Song Sparrow": 3,
    "Northern Mockingbird": 4
}

# 讀取 CSV
df = pd.read_csv(csv_path)

# 過濾只包含五種鳥的資料
df = df[df['name'].isin(label_map.keys())]

# 對每個檔案重新命名
for _, row in df.iterrows():
    original_filename = row['filename']
    bird_name = row['name']
    label = label_map[bird_name]
    
    old_path = os.path.join(selected_folder, original_filename)
    new_filename = f"{label}_{original_filename}"
    new_path = os.path.join(output_folder, new_filename)
    
    if os.path.exists(old_path):
        shutil.copy(old_path, new_path)
    else:
        print(f"找不到檔案：{original_filename}")

print("重新命名並複製完成。")