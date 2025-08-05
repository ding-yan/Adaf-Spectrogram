import pandas as pd
import os
#import os
#basePath = os.path.dirname(os.path.abspath(__file__))

# 確保資料夾 raw 存在
output_folder = 'raw'
os.makedirs(output_folder, exist_ok=True)

MicSigV1=pd.read_json ('./MicSigV1_v1_1.json')
# print(MicSigV1.head())  # 查看前幾行數據

print('----------------------------------')
# 統計 Type 欄位中每個類別的數量
type_counts = MicSigV1['Type'].value_counts()
print(type_counts)
print('----------------------------------')
# 統計 Station 欄位中每個類別的數量
station_counts = MicSigV1['Station'].value_counts()
print(station_counts)
print('----------------------------------')
# 計算每一行的 Data 欄位資料長度
MicSigV1['Data_length'] = MicSigV1['Data'].apply(len)
print(MicSigV1[['Data', 'Data_length']].head())
print('----------------------------------')


# # 篩選 Station 為 BREF，並選擇 Type 和 Data 欄位
# bref_data = MicSigV1[MicSigV1['Station'] == 'BREF'][['Type', 'Data']]
# print(bref_data)
#---------------------------------------------------------------------------------------------------------------------

# 篩選 Station 為 BREF，Type 為 LP 或 VT，並選擇 Type 和 Data 欄位
# filtered_data = MicSigV1[(MicSigV1['Station'] == 'BREF') & (MicSigV1['Type']== 'LP')][['Type', 'Data']]
filtered_data = MicSigV1[(MicSigV1['Station'] == 'BREF') & (MicSigV1['Type'].isin(['LP', 'VT']))][['Type', 'Data']]
print('BREF--(LP & VT):',filtered_data)
print('----------------------------------')
print('BREF--(LP & VT):',len(filtered_data))
print('----------------------------------')
type_counts = filtered_data['Type'].value_counts()
print(type_counts)
print('----------------------------------')
# 計算 Data 欄位的長度
# filtered_data['Data_length'] = filtered_data['Data'].apply(len)
# print(filtered_data)
# print('----------------------------------')


filtered_data_txt = MicSigV1[(MicSigV1['Station'] == 'BREF') & (MicSigV1['Type'].isin(['LP', 'VT']))]

# 記錄 LP 和 VT 的計數器
counters = {'LP': 1, 'VT': 1}

# 依據條件生成 TXT 檔案
for index, row in filtered_data_txt.iterrows():
    # 獲取 Type 和 Data
    type_label = row['Type']
    data_series = row['Data']
    
    # 構建檔案名
    filename = os.path.join(output_folder, f"{type_label}{counters[type_label]}.txt")
    counters[type_label] += 1
    
    # 內容為 Data 並換行
    content = "\n".join(map(str, data_series))
    
    # 寫入 TXT 檔案
    with open(filename, 'w') as file:
        file.write(content)

print("所有檔案生成完成！")