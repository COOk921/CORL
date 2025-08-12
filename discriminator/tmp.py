import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

# from dataset import SequentialPairDataset

from data_sampling import load_and_process_data,generate_sample_pairs
from config import Config
import random
 
# ======== 1. 数据增强函数 ========
def augment_feature(x, noise_std=0.01, drop_prob=0.1):
    """
    对单个特征向量做数据增强：
    1. 添加高斯噪声
    2. 随机丢弃部分特征
    """
    x = x.copy()
    # 高斯噪声
    x += np.random.normal(0, noise_std, size=x.shape)
    # 随机丢特征
    mask = np.random.rand(*x.shape) < drop_prob
    x[mask] = 0
    return x

# ======== 2. 正负样本对生成 ========
def generate_contrastive_pairs(features, window_size=3, num_negatives=1):
    """
    features: numpy array [N, dim]
    window_size: 多大范围算相邻
    num_negatives: 每个正样本对应几个负样本
    """
    pos_pairs = []
    neg_pairs = []
    N = len(features)

    for i in range(N - 1):
        # 正样本对：相邻集装箱
        if abs(i - (i + 1)) <= window_size:
            f1 = augment_feature(features[i])
            f2 = augment_feature(features[i+1])
            pos_pairs.append((f1, f2))

            # 负样本对
            for _ in range(num_negatives):
                j = random.randint(0, N-1)
                while abs(i - j) <= window_size:
                    j = random.randint(0, N-1)
                f_neg = augment_feature(features[j])
                neg_pairs.append((f1, f_neg))

    return pos_pairs, neg_pairs

# ======== 3. MLP 编码器 ========
class MLPEncoder(nn.Module):
    def __init__(self, input_dim, emb_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, emb_dim)
        )

    def forward(self, x):
        return self.net(x)

def nt_xent_loss(z1, z2, temperature=0.5):
    """
    z1, z2: [batch, dim] 对应正样本两侧的embedding
    """
    def cosine_similarity_matrix(x):
        # x: (2N, D)
        x = F.normalize(x, dim=1) 
        return torch.mm(x, x.t()) 

    batch_size = z1.shape[0]
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)

    representations = torch.cat([z1, z2], dim=0)  # 2N × dim
    similarity_matrix = cosine_similarity_matrix(representations)

    # similarity_matrix = F.cosine_similarity(
    #     representations.unsqueeze(1), representations.unsqueeze(0), dim=2
    # )  # 2N × 2N

    # 对角线为自身相似度，置 -inf 避免参与 softmax
    mask = torch.eye(2 * batch_size, dtype=torch.bool).to(z1.device)
    similarity_matrix.masked_fill_(mask, -9e15)

    # 构造正样本索引
    positives = torch.cat([
        torch.arange(batch_size, 2 * batch_size),
        torch.arange(0, batch_size)
    ]).to(z1.device)

    pos_sim = similarity_matrix[torch.arange(2 * batch_size), positives]
    loss = -torch.log(
        torch.exp(pos_sim / temperature) /
        torch.exp(similarity_matrix / temperature).sum(dim=1)
    )

    return loss.mean()
 
def predict_pair(encoder, f1, f2, threshold=0.8, device = "cpu"):
    """
    encoder: 训练好的编码器
    f1, f2: 两个集装箱的特征 (list / numpy array / tensor)
    threshold: 相似度阈值
    """
    f1 = torch.tensor(f1, dtype=torch.float32).unsqueeze(0).to(device)  # shape: [1, dim]
    f2 = torch.tensor(f2, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        z1 = F.normalize(encoder(f1), dim=1)
        z2 = F.normalize(encoder(f2), dim=1)
        sim = F.cosine_similarity(z1, z2).item()

    label = 1 if sim > threshold else 0
    return label, sim

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # --- 1. 生成模拟数据 ---
    print("Load data...")

    all_features,positions = load_and_process_data(Config.root_path, num_samples = Config.num_samples)
    print("all_features:", all_features.shape)
    
    N = all_features.shape[0]
    dim = all_features.shape[1]

    pos_pairs, neg_pairs = generate_contrastive_pairs(all_features)

    all_pairs = pos_pairs + neg_pairs
    labels = [1] * len(pos_pairs) + [0] * len(neg_pairs)

    x1 = torch.tensor([p[0] for p in all_pairs], dtype=torch.float32).to(device)
    x2 = torch.tensor([p[1] for p in all_pairs], dtype=torch.float32).to(device)

    encoder = MLPEncoder(input_dim=dim, emb_dim=64).to(device)
    x1 = x1.to(device)
    x2 = x2.to(device)

    optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-3)

    for epoch in range(10):
        optimizer.zero_grad()
        z1 = encoder(x1)
        z2 = encoder(x2)
        loss = nt_xent_loss(z1, z2)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
    torch.save(encoder.state_dict(), "encoder.pth")

    encoder.eval()

    data,_ = load_and_process_data(Config.root_path, num_samples=500)

    nodes_1 = data[11]
    nodes_2 = data[12]
    label, sim = predict_pair(encoder,nodes_1,nodes_2,threshold=0.85,device = device)
    print(f"预测类别: {label}, 相似度: {sim:.4f}")

    nodes_2 = data[3]
    label, sim = predict_pair(encoder,nodes_1,nodes_2,threshold=0.85,device = device)
    print(f"预测类别: {label}, 相似度: {sim:.4f}")
