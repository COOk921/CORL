import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class SequentialPairDataset(Dataset):
    """
    根据序列邻近性创建正负样本对的自定义数据集。
    """
    def __init__(self, features, window_size, num_neg_samples):
        """
        Args:
            features (np.array): 形状为 (num_samples, feature_dim) 的节点特征序列。
            window_size (int): 定义正样本的邻近窗口。
            num_neg_samples (int): 每个锚点节点生成的负样本数量。
        """
        super().__init__()
        self.features = features
        self.window_size = window_size
        self.num_neg_samples = num_neg_samples
        self.num_nodes = len(features)
        
        self.pairs = []
        self.labels = []
        self._generate_pairs()

    def _generate_pairs(self):
        print("Generating positive and negative pairs...")
        positive = 0
        Negative = 0  
        for i in range(self.num_nodes):
            # --- 创建正样本 ---
            # 寻找窗口内的邻居
            start = max(0, i - self.window_size)
            end = min(self.num_nodes, i + self.window_size + 1)
            for j in range(start, end):
                if i == j:
                    continue
                self.pairs.append((self.features[i], self.features[j]))
                self.labels.append(1) # 标签为 1
                positive += 1

            # --- 创建负样本 ---
            for _ in range(self.num_neg_samples):
                # 随机选择一个远离窗口的节点
                while True:
                    k = np.random.randint(0, self.num_nodes)
                    if abs(i - k) > self.window_size:
                        self.pairs.append((self.features[i], self.features[k]))
                        self.labels.append(0) # 标签为 0
                        Negative += 1
                        break
           

        print(f"Generated {len(self.pairs)} total pairs. positive: {positive}, negative: {Negative}")
       

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        node1_feat, node2_feat = self.pairs[idx]
        label = self.labels[idx]
        
        # 将Numpy数组转为Torch Tensor
        node1_feat = torch.from_numpy(node1_feat).float()
        node2_feat = torch.from_numpy(node2_feat).float()
        label = torch.tensor(label, dtype=torch.float32)
        
        return node1_feat, node2_feat, label