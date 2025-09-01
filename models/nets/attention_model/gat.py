from torch_geometric.nn import GCNConv
import torch.nn.functional as F
import torch
import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GATConv


class GAT(nn.Module):
    def __init__(self, in_channels, embed_dim, hidden_dim, out_dim, num_layers=3, dropout=0.2, heads=4):
       
        super().__init__()
        self.dropout = dropout
        self.num_layers = num_layers
        self.heads = heads
       
        self.node_emb = nn.Linear(in_channels, out_dim)

        self.convs = nn.ModuleList()
        self.convs.append(GATConv(in_channels, hidden_dim, heads=self.heads))
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_dim * self.heads, hidden_dim, heads=self.heads))
        if num_layers > 1:
            # 最后一层 GAT 不使用多头，所以 heads 设置为 1
            self.convs.append(GATConv(hidden_dim * self.heads, out_dim, heads=1))
        else:  
            self.convs[0] = GATConv(embed_dim, out_dim, heads=1)

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
