import torch
from sklearn.metrics import accuracy_score, roc_auc_score

def train(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for node1, node2, labels in dataloader:
        node1, node2, labels = node1.to(device), node2.to(device), labels.to(device)
        
       
        # 梯度清零
        optimizer.zero_grad()
        
        # 正向传播
        # 在这里可以传入你的启发式特征
        # heuristic_feat = calculate_heuristics(node1, node2).to(device)
        # outputs = model(node1, node2, heuristic_feat)
        outputs = model(node1, node2)
        
        # 计算损失
        loss = criterion(outputs, labels.unsqueeze(1))
        
        # 反向传播
        loss.backward()
        
        # 更新权重
        optimizer.step()
        
        total_loss += loss.item()
        
    return total_loss / len(dataloader)


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_labels = []
    all_preds = []
    
    with torch.no_grad():
        for node1, node2, labels in dataloader:
            node1, node2, labels = node1.to(device), node2.to(device), labels.to(device)
            
            outputs = model(node1, node2)
            loss = criterion(outputs, labels.unsqueeze(1))
            total_loss += loss.item()

            


            # 收集预测和标签用于计算评估指标
            preds = torch.round(outputs).squeeze()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
         

            
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    try:
        auc = roc_auc_score(all_labels, all_preds)
    except ValueError:
        auc = 0.5 # 如果标签全为一种，无法计算AUC
        
    return avg_loss, accuracy, auc