import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split

from model import Discriminator
from dataset import SequentialPairDataset
from train import train, evaluate
from data_sampling import load_and_process_data

import pdb

CONFIG = {
    "input_dim": 12,         # 每个节点的特征维度
    "hidden_dim": 256,         # 模型隐藏层维度
    "heuristic_dim": 10,       # (可选) 启发式方法产生的特征维度
    "learning_rate": 0.001,
    "epochs": 10,
    "batch_size": 64,
    "window_size": 3,          # 定义“邻近”的窗口大小
    "num_neg_samples": 3,      # 每个正样本对应生成的负样本数量
    "test_size": 0.2,          # 划分训练集和验证集的比例

    "root_path": "./data/processed_container_data.pkl",
    "save_path": "./discriminator/model/discriminator.pth"
}


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model_for_inference = Discriminator(
    input_dim=CONFIG["input_dim"],
    hidden_dim=CONFIG["hidden_dim"],
).to(device)

model_for_inference.load_state_dict(torch.load(CONFIG["save_path"]))


model_for_inference.eval()
print("Model loaded successfully!")

with torch.no_grad():
    
    data,_ = load_and_process_data(CONFIG["root_path"], num_samples=500)
    
    nodes_1 = torch.from_numpy(data[0:20]).float().to(device) #[0,10]
    nodes_2 = torch.from_numpy(data[20:21]).float().to(device) #[10,20]

    node_0_feat = torch.from_numpy(data[10]).float().view(-1, CONFIG["input_dim"]).to(device)
    node_10_feat = torch.from_numpy(data[11]).float().view(-1, CONFIG["input_dim"]).to(device)

    node_5_feat = torch.from_numpy(data[20]).float().view(-1, CONFIG["input_dim"]).to(device)
    node_15_feat = torch.from_numpy(data[21]).float().view(-1, CONFIG["input_dim"]).to(device)


   
    similarity_score = model_for_inference(nodes_1, nodes_2)
    print(similarity_score)

    # 使用加载的模型进行预测
    similarity_score = model_for_inference(node_0_feat, node_10_feat).item()
    print(f"Similarity score between 0 and 10: {similarity_score:.4f}")

    similarity_score = model_for_inference(node_5_feat, node_15_feat).item()
    print(f"Similarity score between 5 and 15: {similarity_score:.4f}")

