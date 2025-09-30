import os
import pandas as pd
from scipy.stats import kendalltau, spearmanr
import pdb
from itertools import permutations
import numpy as np

result_folder = './result/'
output_folder = './result/merged/'
os.makedirs(output_folder, exist_ok=True)
csv_files = [f for f in os.listdir(result_folder) if f.endswith('.csv')]

file_groups = {}
for file in csv_files:
    key = file.split("',")[0] + "'"
    if key not in file_groups:
        file_groups[key] = []
    file_groups[key].append(file)

avg_kendall_corr = 0
num_groups = 0

for group_key, files in file_groups.items():
    merged_df = pd.DataFrame()
    df_list = []
    file_name = []
    order = []
    num = 0

    for file in files:
        file_path = os.path.join(result_folder, file)
        df = pd.read_csv(file_path)
        file_name.append(file.split('.')[0])
        df_list.append(df)
        order.append(df['order'].values[0])

    order = [sorted(order).index(x) for x in order]
    for i in range(len(order)):
        idx = order.index(i)
        df = df_list[idx]

        df = df[~(df.iloc[:, :-1] == 0).all(axis=1)] #保留有效node
        df = df.copy()
        df['pred'] = df['pred'].rank(method='dense').astype(int) - 1 + num  # 重排
        num += df.shape[0]
        pdb.set_trace()

        if merged_df.empty:
            merged_df = df
        else:
            merged_df = pd.concat([merged_df, df], ignore_index=True)
        
    target = np.arange(len(merged_df))
    
    kendall_corr, _ = kendalltau(merged_df['order'].values, merged_df['pred'].values)#  merged_df['order'].values
    print(f"Single Kendall: {kendall_corr:.4f}")
    avg_kendall_corr += kendall_corr
    num_groups += 1
    
    
    merged_file_name = group_key.strip("'") + '\').csv'
    merged_file_path = os.path.join(output_folder, merged_file_name)
    merged_df.to_csv(merged_file_path, index=False)
    
print(f"Average Kendall : {avg_kendall_corr / num_groups:.4f}")