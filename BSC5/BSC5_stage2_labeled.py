import pandas as pd
import os
import shutil

# data paths
csv_path = './data/bird_songs_metadata.csv'
#csv_path = 'C:\\Users\\User\\Desktop\\Dataset\\bird song data set\\bird_songs_metadata.csv'
selected_folder = './data/selected_wavfiles'
#selected_folder = 'C:\\Users\\User\\Desktop\\Dataset\\bird song data set\\selected_wavfiles'
output_folder = './data/selected_wavfiles_label'
#output_folder = 'C:\\Users\\User\\Desktop\\Dataset\\bird song data set\\selected_wavfiles_label'

# create output directory
os.makedirs(output_folder, exist_ok=True)

# define bird species mapping
label_map = {
    "Bewick's Wren": 0,
    "Northern Cardinal": 1,
    "American Robin": 2,
    "Song Sparrow": 3,
    "Northern Mockingbird": 4
}

# read CSV
df = pd.read_csv(csv_path)

# filter only the data of five bird species
df = df[df['name'].isin(label_map.keys())]

# rename and copy each file
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
        print(f"Cannot find file：{original_filename}")

print("Rename and copy completed.")
