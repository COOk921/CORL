import pickle
import pandas as pd
import numpy as np
import random

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

    
    dfs = [df['data'].drop(columns=['Unit Nbr', 'Time Completed'], errors='ignore') for df in dfs]
    merged_df = pd.concat(dfs, ignore_index=True)

    # 定义一个数组，size为key的数量+1，记录每个key对应数据在合并后DataFrame中的起始和终止位置
    positions = [0]
    cumulative_length = 0
    for df in dfs:
        cumulative_length += len(df)
        positions.append(cumulative_length)

    data = merged_df.head(num_samples).values.astype(np.float32)

    return data,positions



def generate_sample_pairs(all_features, num_pos_samples, neighbor_range=20, seed=42):
    """
    生成固定数量的正负样本对
    参数:
        all_features: np.ndarray, shape=(num_nodes, dim) 集装箱特征
        num_pos_samples: int, 希望采样的正样本数量
        neighbor_range: int, 用于生成负样本时随机邻近范围
        seed: int, 随机种子
    返回:
        pairs: list of tuple(np.ndarray, np.ndarray)
        labels: list of int
    """
    np.random.seed(seed)
    num_nodes = len(all_features)
    
    # Step 1: 所有可能的正样本索引 (相邻)
    all_pos_indices = [(i, i+1) for i in range(num_nodes - 1)]
    
    # Step 2: 随机采样固定数量的正样本
    if num_pos_samples > len(all_pos_indices):
        raise ValueError("正样本数量超过可用的相邻对数！")
    pos_indices = np.random.choice(len(all_pos_indices), size=num_pos_samples, replace=False)
    pos_indices = [all_pos_indices[idx] for idx in pos_indices]
    
    pairs = []
    labels = []
    
    for i, j in pos_indices:
        # 正样本
        pairs.append((all_features[i], all_features[j]))
        labels.append(1)
        
        # 负样本 1: 倒序
        # pairs.append((all_features[j], all_features[i]))
        # labels.append(0)
        
        # 负样本 2: 其中一个 + 随机邻近集装箱（不等于另一个正样本节点）
        anchor = np.random.choice([i, j])  # 随机选正样本中的一个作为anchor
        while True:
            k = np.random.randint(max(0, anchor - neighbor_range), 
                                  min(num_nodes, anchor + neighbor_range + 1))
            if k != anchor and k != i and k != j:
                break
        pairs.append((all_features[anchor], all_features[k]))
        labels.append(0)
    

    # combined = list(zip(pairs, labels))
    # random.shuffle(combined)  # 原地打乱

    # pairs, labels = zip(*combined)  # 解压回两个变量
    # pairs = list(pairs)
    # labels = list(labels)

    return pairs, labels

