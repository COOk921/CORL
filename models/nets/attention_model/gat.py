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
    def __init__(self, in_channels, embed_dim, hidden_dim, out_dim, num_layers=2, dropout=0.3, heads=2):
       
        super().__init__()
        self.dropout = dropout
        self.num_layers = num_layers
        self.heads = heads
       
        self.node_emb = nn.Linear(in_channels, out_dim)

        self.convs = nn.ModuleList()
        self.convs.append(GATConv(out_dim, hidden_dim, heads=self.heads))
        
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_dim * self.heads, hidden_dim, heads=self.heads))
        if num_layers > 1:
            # 最后一层 GAT 不使用多头，所以 heads 设置为 1
            self.convs.append(GATConv(hidden_dim * self.heads, out_dim, heads=1))
        else:  
            self.convs[0] = GATConv(embed_dim, out_dim, heads=1)

    def forward(self, batch):

        node_features = batch.x
        edge_index = batch.edge_index
        
        update_node_feature = self.node_emb(node_features)  
        init_h =  update_node_feature
       
        for i, conv in enumerate(self.convs):
            update_node_feature = conv(update_node_feature, edge_index)
            if i != len(self.convs) - 1:
                update_node_feature = F.elu(update_node_feature)
                update_node_feature = F.dropout(update_node_feature, p=self.dropout, training=self.training)

        # update_node_feature = update_node_feature  + self.node_emb(init_h)
        return update_node_feature


# from torch_geometric.nn import GCNConv
# import torch.nn.functional as F
# import torch
# import pdb

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch_geometric.nn import GCNConv
# from torch_geometric.nn import GATConv

# class GATLayer(nn.Module):
#     def __init__(self, in_dim, out_dim, heads, dropout=0.2, feed_forward_hidden=512):
#         super().__init__()
#         # GAT Attention部分
#         self.gat_conv = GATConv(in_dim, out_dim, heads=heads, dropout=dropout)
        
#         # Add & Norm
#         self.dropout1 = nn.Dropout(dropout)
#         self.norm1 = nn.LayerNorm(out_dim * heads)
        
#         # Add & Norm
#         self.dropout2 = nn.Dropout(dropout)
#         self.norm2 = nn.LayerNorm(out_dim * heads)

#     def forward(self, x, edge_index):

#         attended_x = self.gat_conv(x, edge_index)
      
#         return x


# class GAT(nn.Module):
#     def __init__(self, in_channels, hidden_dim, out_dim, num_layers=3, dropout=0.2, heads=2, feed_forward_hidden=256):
        
#         super().__init__()
#         self.dropout = dropout
#         self.num_layers = num_layers
#         self.input_proj = nn.Linear(in_channels, hidden_dim * heads)
        
#         self.gat_layers = nn.ModuleList()
#         for _ in range(num_layers -1):
#             self.gat_layers.append(
#                 GATLayer(
#                     in_dim= hidden_dim * heads, 
#                     out_dim=hidden_dim, 
#                     heads=heads,
#                     dropout=dropout,
#                     feed_forward_hidden=feed_forward_hidden
#                 )
#             )

#         self.final_gat = GATConv(hidden_dim * heads, out_dim, heads=1, dropout=dropout)
        
#     def forward(self, batch):
#         node_features = batch.x
#         edge_index = batch.edge_index

#         h = node_features
#         h = self.input_proj(node_features) # ->hidden_dim*heads 128*2=256

#         for layer in self.gat_layers:
#             h = layer(h, edge_index)
            
#         h_out = self.final_gat(h, edge_index)
        
#         return h_out

