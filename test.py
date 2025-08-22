import torch
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv
import torch.nn.functional as F

# 假设输入
batch_size = 128
num_samples = 100     # 每个 batch 内 100 个西瓜
num_features = 6      # 每个西瓜 6 个特征
num_feature_values = 10  # 假设所有特征总共有 10 种取值（类别id范围 0~9）

# 随机生成类别型特征
# 这里每个西瓜的 6 个特征值都是 0~9 的整数
x = torch.randint(0, num_feature_values, (batch_size, num_samples, num_features))  # [128, 100, 6]
print(x[0,:3,:])

def build_graph_for_batch(sample_features, num_feature_values):
    """
    输入：sample_features [num_samples, num_features]
    输出：一个 PyG Data 图对象
    """
    num_samples = sample_features.size(0)

    # 节点：西瓜 + 特征值
    # 0 ~ num_samples-1 : 西瓜节点
    # num_samples ~ num_samples+num_feature_values-1 : 特征值节点
    total_nodes = num_samples + num_feature_values
    edge_index = []

    # 构建边
    for i in range(num_samples):
        for f in range(sample_features.size(1)):
            fv = sample_features[i, f].item()
            fv_node = num_samples + fv
            edge_index.append([i, fv_node])
            edge_index.append([fv_node, i])  # 双向

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    # 初始特征: 用 one-hot，简单起见
    x_nodes = torch.eye(total_nodes)

    return Data(x=x_nodes, edge_index=edge_index)


# --- 构建 batch ---
graphs = []
for b in range(batch_size):
    g = build_graph_for_batch(x[b], num_feature_values)
    graphs.append(g)

# 合并成一个 Batch
batch_graph = Batch.from_data_list(graphs)
print(batch_graph)
# >>> Batch(x=[128*110, 110], edge_index=[2, ...], batch=[128*110])


# --- 定义 GCN 模型 ---
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x


# 模型初始化
in_channels = batch_graph.x.size(1)  # one-hot 维度
model = GCN(in_channels, hidden_channels=64, out_channels=32)

# 前向传播
out = model(batch_graph.x, batch_graph.edge_index, batch_graph.batch)
print("所有节点 embedding:", out.shape)  # [128*110, 32]

# 按 batch 分割
watermelon_emb = []
start = 0
for b in range(batch_size):
    num_nodes = num_samples + num_feature_values
    wm_nodes = out[start : start + num_samples]  # 前100是西瓜节点
    watermelon_emb.append(wm_nodes)
    start += num_nodes
watermelon_emb = torch.stack(watermelon_emb)  # [128, 100, 32]

print("西瓜 embedding:", watermelon_emb.shape)
