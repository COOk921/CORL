from torch_geometric.nn import GCNConv
import torch.nn.functional as F
import torch
import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCN(nn.Module):
    def __init__(self, num_nodes, embed_dim, hidden_dim, out_dim, num_layers=2, dropout=0.3):
       
        super().__init__()
        self.dropout = dropout
        self.num_layers = num_layers
        self.node_emb = nn.Linear(num_nodes, embed_dim)

        # 构建 GCN 层
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(embed_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        if num_layers > 1:
            self.convs.append(GCNConv(hidden_dim, out_dim))
        else:  
            self.convs[0] = GCNConv(embed_dim, out_dim)

    def forward(self, node_ids, edge_index):
        
        update_node_feature = self.node_emb(node_ids)  # 节点 id → embedding
        init_h =  update_node_feature

        for i, conv in enumerate(self.convs):
            update_node_feature= conv(update_node_feature, edge_index)
            if i != len(self.convs) - 1:
                update_node_feature = F.relu(update_node_feature)
                update_node_feature = F.dropout(update_node_feature, p=self.dropout, training=self.training)


        update_node_feature = update_node_feature + init_h
        return update_node_feature
