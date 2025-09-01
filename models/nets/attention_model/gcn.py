from torch_geometric.nn import GCNConv
import torch.nn.functional as F
import torch
import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCN(nn.Module):
    def __init__(self, in_channels, embed_dim, hidden_dim, out_dim, num_layers=2, dropout=0.3):
       
        super().__init__()
        self.dropout = dropout
        self.num_layers = num_layers
       
        self.node_emb = nn.Linear(in_channels, embed_dim)

        # 构建 GCN 层
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_dim))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        if num_layers > 1:
            self.convs.append(GCNConv(hidden_dim, out_dim))
        else:  
            self.convs[0] = GCNConv(embed_dim, out_dim)

    def forward(self, batch):
        # 从 DataBatch 中获取节点特征和边索引
        node_features = batch.x
        edge_index = batch.edge_index
        
        update_node_feature = node_features  #self.node_emb(node_features)  
        init_h =  update_node_feature
       
        for i, conv in enumerate(self.convs):
            update_node_feature = conv(update_node_feature, edge_index)
            if i != len(self.convs) - 1:
                update_node_feature = F.relu(update_node_feature)
                update_node_feature = F.dropout(update_node_feature, p=self.dropout, training=self.training)


        update_node_feature = update_node_feature  + self.node_emb(init_h)
        return update_node_feature
