import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import numpy as np
from sklearn.model_selection import train_test_split

from data_sampling import load_and_process_data

# ===================================================================
# 1. 配置参数 (CONFIG) - 已更新
# ===================================================================
CONFIG = {
    "input_dim": 6,
    "embedding_dim": 64,
    "hidden_dim": 128,
    "learning_rate": 0.001,
    "epochs": 30,
    "batch_size": 128,         # InfoNCE 受益於更大的批次大小
    "window_size": 5,
    "temperature": 0.07,       # InfoNCE 的關鍵超參數
    "test_size": 0.2,

    "root_path": "./data/processed_container_data.pkl",
    "save_path": "./contrastive/encoder.pth",

    "num_samples": 20000
}

class InfoNCELoss(nn.Module):

    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, query_emb, key_emb):
        
        batch_size = query_emb.shape[0]
        
        # 正規化嵌入以計算餘弦相似度
        query_emb = F.normalize(query_emb, p=2, dim=1)
        key_emb = F.normalize(key_emb, p=2, dim=1)
        
        # 計算查詢與所有鍵的相似度矩陣
        # 相似度矩陣的 (i, j) 元素是 query_emb[i] 和 key_emb[j] 的相似度
        similarity_matrix = torch.matmul(query_emb, key_emb.T) / self.temperature
        
        # 標籤：對於第 i 個查詢，它的正樣本在相似度矩陣的第 i 行第 i 列
        # 這構成了一個多分類問題，標籤是 [0, 1, 2, ..., batch_size-1]
        labels = torch.arange(batch_size, device=query_emb.device)
        
        # 計算交叉熵損失
        # 我們將 query_emb 和 key_emb 視為對稱的，所以計算兩次損失
        loss_i = self.criterion(similarity_matrix, labels)
        loss_j = self.criterion(similarity_matrix.T, labels)
        
        loss = (loss_i + loss_j) / 2
        return loss



# ===================================================================
# 2. 数据加载器 (PositivePairDataset)
# ===================================================================
class PositivePairDataset(Dataset):
    """为 InfoNCE 创建正样本对。"""
    def __init__(self, features, window_size):
        super().__init__()
        self.features = features
        self.window_size = window_size
        self.num_nodes = len(features)
        self.pairs = self._generate_pairs()

    def _generate_pairs(self):
        print("Generating positive pairs...")
        pairs = []
        for i in range(self.num_nodes):
            possible_positives = [j for j in range(max(0, i - self.window_size), min(self.num_nodes, i + self.window_size + 1)) if i != j]
            if not possible_positives:
                continue
            positive_idx = np.random.choice(possible_positives)
            pairs.append((self.features[i], self.features[positive_idx]))
        print(f"Generated {len(pairs)} positive pairs.")
        return pairs
    
    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        node1, node2 = self.pairs[idx]
        return torch.from_numpy(node1).float(), torch.from_numpy(node2).float()



class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, embedding_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim)
        )
    def forward(self, x):
        return self.network(x)

# ===================================================================
# 3. 训练和评估函数
# ===================================================================
def train_infonce(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for node1, node2 in dataloader:
        node1, node2 = node1.to(device), node2.to(device)
        optimizer.zero_grad()
        emb1 = model(node1)
        emb2 = model(node2)
        loss = criterion(emb1, emb2)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def evaluate_infonce(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for node1, node2 in dataloader:
            node1, node2 = node1.to(device), node2.to(device)
            emb1 = model(node1)
            emb2 = model(node2)
            loss = criterion(emb1, emb2)
            total_loss += loss.item()

            # 评估 Top-1 准确率
            similarity_matrix = torch.matmul(F.normalize(emb1, p=2, dim=1), F.normalize(emb2, p=2, dim=1).T)
            # 找到每個查詢最相似的鍵的索引
            preds = torch.argmax(similarity_matrix, dim=1)
            # 正確的標籤應該是對角線上的元素
            labels = torch.arange(len(preds), device=device)
            total_correct += torch.sum(preds == labels).item()
            total_samples += len(preds)
            
    accuracy = total_correct / total_samples
    return total_loss / len(dataloader), accuracy

# ===================================================================
# 4. 主程序入口
# ===================================================================
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    all_features,positions = load_and_process_data(CONFIG["root_path"], num_samples = CONFIG["num_samples"])
    print("all_features:", all_features.shape)

    train_features, val_features = train_test_split(all_features, test_size=CONFIG["test_size"], shuffle=False)
    
    train_dataset = PositivePairDataset(train_features, CONFIG["window_size"])
    val_dataset = PositivePairDataset(val_features, CONFIG["window_size"])
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"], shuffle=False)
    
    model = Encoder(
        input_dim=CONFIG["input_dim"],
        hidden_dim=CONFIG["hidden_dim"],
        embedding_dim=CONFIG["embedding_dim"]
    ).to(device)
    
    criterion = InfoNCELoss(temperature=CONFIG["temperature"])
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])
    
    print("\nStarting InfoNCE contrastive training...")
    # for epoch in range(CONFIG["epochs"]):
    #     train_loss = train_infonce(model, train_loader, optimizer, criterion, device)
    #     val_loss, val_acc = evaluate_infonce(model, val_loader, criterion, device)
        
    #     print(f"Epoch {epoch+1}/{CONFIG['epochs']} | "
    #           f"Train Loss: {train_loss:.4f} | "
    #           f"Val Loss: {val_loss:.4f} | "
    #           f"Val Top-1 Acc: {val_acc:.4f}")
              
    # torch.save(model.state_dict(), CONFIG["save_path"])
    
    model.load_state_dict(torch.load(CONFIG["save_path"]))
    print("\n--- Using the trained encoder as a discriminator ---")
    model.eval()
    with torch.no_grad():
        data,_ = load_and_process_data(CONFIG["root_path"], num_samples=500)
    
        nodes_A = torch.from_numpy(data[3]).float().to(device) #[0,10]
        nodes_B = torch.from_numpy(data[4]).float().to(device) #[10,20]
        nodes_C = torch.from_numpy(data[25]).float().to(device) #[10,20]
        
        emb_A = model(nodes_A)
        emb_B = model(nodes_B)
        emb_C = model(nodes_C)
        
        dist_AB = F.pairwise_distance(emb_A, emb_B).item()
        dist_AC = F.pairwise_distance(emb_A, emb_C).item()
        
        print(f"Distance between similar nodes (A, B): {dist_AB:.4f}")
        print(f"Distance between dissimilar nodes (A, C): {dist_AC:.4f}")
