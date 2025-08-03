import logging
import pickle
import pandas as pd
import numpy as np

root_dir = "./data/container_data.pkl"


def read_pkl(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data


class ContainerDataset:
    def __init__(self):
        
        data = read_pkl(root_dir)
        self.data = {tuple(key) if isinstance(key, np.ndarray) else key: value for key, value in data.items()}
        self.keys = list(self.data.keys())
        self.rng = np.random.default_rng()  # For random sampling
        self.current_idx = 0  # For sequential evaluation

        self.selected_columns = ['from_bay', 'from_col', 'from_layer', 'to_bay', 'to_col', 'to_layer']

    def get_next_data(self, max_nodes, mode='train'):
        if mode == 'train':
            # Randomly select a key for training
            key = self.rng.choice(self.keys)
        else:
            # Sequential selection for evaluation
            if self.current_idx >= len(self.keys):
                self.current_idx = 0  # Loop back to start
            key = self.keys[self.current_idx]
            self.current_idx += 1
       
        df = self.data[tuple(key)]

        nodes = df[self.selected_columns].to_numpy()[:max_nodes]  # 限制为最大节点数
        
        if len(nodes) < max_nodes:
            # Pad with zeros if fewer nodes
            nodes = np.pad(nodes, ((0, max_nodes - len(nodes)), (0, 0)), mode='constant')
        # Normalize to [0, 1] if needed
        nodes = (nodes - nodes.min(axis=0)) / (nodes.max(axis=0) - nodes.min(axis=0) + 1e-8)

        
        return nodes


# dataloader = ContainerDataset()
# node = dataloader.get_next_data(20)





