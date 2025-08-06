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
    # keys = list(data.keys())

    # 合并所有的DataFrame
    dfs = list((data).values())
    # 移除需要忽略的特征列
    dfs = [df.drop(columns=['Unit Nbr', 'Time Completed'], errors='ignore') for df in dfs]
    merged_df = pd.concat(dfs, ignore_index=True)

    # 定义一个数组，size为key的数量+1，记录每个key对应数据在合并后DataFrame中的起始和终止位置
    positions = [0]
    cumulative_length = 0
    for df in dfs:
        cumulative_length += len(df)
        positions.append(cumulative_length)

    data = merged_df.head(num_samples).values.astype(np.float32)

    return data,positions

# data = load_and_process_data()

