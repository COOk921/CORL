import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader,TensorDataset
import numpy as np
from sklearn.model_selection import train_test_split

from model import Discriminator,PairClassifier
# from dataset import SequentialPairDataset
from train import train, evaluate
from data_sampling import load_and_process_data,generate_sample_pairs
from config import Config

import pdb




if __name__ == '__main__':
 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # --- 1. 生成模拟数据 ---
    print("Load data...")

    all_features,positions = load_and_process_data(Config.root_path, num_samples = Config.num_samples)
    print("all_features:", all_features.shape)
     
    # --- 2. 生成所有的样本对 ---
    print("Step 1: Generating all pairs from the entire dataset...")
    all_pairs = []
    all_labels = []
    num_nodes = len(all_features)
    
    for i in range(num_nodes):
        # 创建正样本
        start = max(0, i - Config.window_size)
        end = min(num_nodes, i + Config.window_size + 1)
     
        for j in range(start, end):
            if i == j: continue
            all_pairs.append((all_features[i], all_features[j]))
            all_labels.append(1)
            

        # positions[idx] >= i
        idx = np.searchsorted(positions, i)

        for _ in range(Config.num_neg_samples):
            while True:        
                # k = np.random.randint(positions[idx], positions[idx+1])    
                k = np.random.randint(i-5, i+5)
                if abs(i - k) > Config.window_size:
                    k = np.random.randint(0, num_nodes) if k >= num_nodes else k
                    all_pairs.append((all_features[i], all_features[k]))
                    all_labels.append(0)
                    break
                if positions[idx+1] - positions[idx] <= Config.window_size:
                    break
    # all_pairs, all_labels = generate_sample_pairs(all_features, 90000)     

    
    print(f"Generated {len(all_pairs)} total pairs.")
    print(f"Generated {len(all_pairs)} total pairs. positive: {sum(all_labels)}, negative: {len(all_labels) - sum(all_labels)}")
    
    X = np.array([list(pair) for pair in all_pairs])
    y = np.array(all_labels)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, 
        test_size=Config.test_size, 
        shuffle=True, 
        stratify=y  
    )

    train_dataset = TensorDataset(torch.from_numpy(X_train[:, 0, :]), torch.from_numpy(X_train[:, 1, :]), torch.from_numpy(y_train).float())
    val_dataset = TensorDataset(torch.from_numpy(X_val[:, 0, :]), torch.from_numpy(X_val[:, 1, :]), torch.from_numpy(y_val).float())

    train_loader = DataLoader(train_dataset, batch_size=Config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.batch_size, shuffle=False)
   
    
    # --- 3. 初始化模型、损失函数和优化器 ---
    model = Discriminator(
        input_dim=Config.input_dim,
        hidden_dim=Config.hidden_dim,
        heuristic_dim=Config.heuristic_dim # 如果要用启发式方法，取消这行注释
    ).to(device)
    # model = PairClassifier(Config.input_dim, Config.hidden_dim).to(device)

    
    criterion = nn.BCELoss() # 二元交叉熵损失
    optimizer = optim.Adam(model.parameters(), lr=Config.learning_rate)
    
    print("\nStarting training...")
    # --- 4. 训练循环 ---
    for epoch in range(Config.epochs):
        train_loss = train(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, val_auc = evaluate(model, val_loader, criterion, device)
        
        print(f"Epoch {epoch+1}/{Config.epochs} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val Acc: {val_acc:.4f} | "
              f"Val AUC: {val_auc:.4f}")
        
        if epoch % 5 or epoch == Config.epochs - 1:
            torch.save(model.state_dict(), Config.save_path)
    

    print("\nTraining finished.")
    
    # --- 5. 使用模型进行预测 ---
    # 假设你想预测两个新节点的相似度
    model.eval()
    with torch.no_grad():
        node_A_feat = torch.randn(1, Config.input_dim).to(device)
        node_B_feat = torch.randn(1, Config.input_dim).to(device)
        similarity_score = model(node_A_feat, node_B_feat).item()
        print(f"\nSimilarity score between two random nodes: {similarity_score:.4f}")
