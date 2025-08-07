import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim, heuristic_dim=0):
        super().__init__()
        self.use_heuristics = heuristic_dim > 0
        
        # 计算组合后的特征维度
        # 我们使用拼接和相减两种方式来组合节点特征
        combined_dim = input_dim * 2 + heuristic_dim
        
        self.network = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid() # 输出归一化的分数
        )

    def forward(self, node1_feat, node2_feat, heuristic_feat=None):

        feature_diff = torch.abs(node1_feat - node2_feat)
        #combined = feature_diff
        feature_prod = node1_feat * node2_feat
        combined = torch.cat([feature_diff, feature_prod], dim=-1)

        # 如果使用启发式特征，在这里拼接
        if self.use_heuristics and heuristic_feat is not None:
             combined = torch.cat([combined, heuristic_feat], dim=1)

       
        return self.network(combined)
