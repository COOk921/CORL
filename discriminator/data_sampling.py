import pickle
import pandas as pd
import numpy as np
import torch
import pdb



def read_pkl(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

def load_and_process_data(file_path, num_samples=200):
 
    data = read_pkl(file_path)
    data = {tuple(key) if isinstance(key, np.ndarray) else key: value for key, value in data.items()}
    
    # 合并所有的DataFrame
    dfs = list(data.values())
    # 移除需要忽略的特征列
    dfs = [df.drop(columns=['Unit Nbr', 'Time Completed'], errors='ignore') for df in dfs]
    merged_df = pd.concat(dfs, ignore_index=True)

    data = merged_df.head(num_samples).values.astype(np.float32)
    
    return data

# data = load_and_process_data()

