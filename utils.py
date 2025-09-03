import pdb
from scipy.stats import kendalltau, spearmanr
import numpy as np
from torch_geometric.data import Data
import torch
from torch_geometric.data import Data, Batch

def _count_increasing_pairs(arr):
    """
    Count pairs (i < j) with arr[i] < arr[j].
    Uses coordinate-compression + Fenwick (BIT) -> O(len(arr) log U)
    """
    if arr.size < 2:
        return 0
    vals, inv = np.unique(arr, return_inverse=True)  # inv: 0..U-1 ranks
    U = vals.size
    bit = [0] * (U + 1)  # 1-indexed BIT

    def bit_add(i):
        i += 1
        while i <= U:
            bit[i] += 1
            i += i & -i

    def bit_sum(i):
        # sum of [0..i], i is 0-based index; if i < 0 -> 0
        if i < 0:
            return 0
        s = 0
        i += 1
        while i > 0:
            s += bit[i]
            i -= i & -i
        return s

    cnt = 0
    for rank in inv:  # iterate in original order
        if rank > 0:
            cnt += bit_sum(rank - 1)   # number of previous elements with value < current
        # else rank == 0 -> 0 previous smaller
        bit_add(rank)
    return cnt

def compute_rehandle_rate(from_layer, from_col, from_bay, from_yard, denominator='same'):
    """
    Compute '捣箱' statistics.

    Parameters
    ----------
    from_layer, from_col, from_bay, from_yard : array-like, same length N
        Positions of containers; index order is the current order.
    denominator : {'same', 'all'}
        'same' (default) -> compute rate = rehandle_pairs / (pairs sharing same yard,bay,col)
        'all' -> compute rate = rehandle_pairs / (N*(N-1)/2)

    Returns
    -------
    rehandle_pairs : int
        Number of pairs (i<j) that satisfy: same yard,bay,col AND from_layer[i] < from_layer[j]
    total_pairs_for_denominator : int
        Denominator used (either number of same-stack pairs or total possible pairs)
    rate : float
        rehandle_pairs / total_pairs_for_denominator (0.0 if denominator == 0)
    """
    # convert to numpy arrays
    from_layer = np.asarray(from_layer)
    from_col   = np.asarray(from_col)
    from_bay   = np.asarray(from_bay)
    from_yard  = np.asarray(from_yard)

    if not (from_layer.shape == from_col.shape == from_bay.shape == from_yard.shape):
        raise ValueError("All input arrays must have the same shape/length.")

    N = from_layer.size
    # build grouping key (yard, bay, col)
    keys = np.vstack((from_yard, from_bay, from_col)).T
    unique_keys, inverse = np.unique(keys, axis=0, return_inverse=True)

    rehandle_pairs = 0
    total_same_stack_pairs = 0
    # iterate each group (same yard,bay,col)
    for gid in range(unique_keys.shape[0]):
        idxs = np.nonzero(inverse == gid)[0]  # indices in original order (ascending)
        m = idxs.size
        total_same_stack_pairs += m * (m - 1) // 2
        if m < 2:
            continue
        layers = from_layer[idxs]
        rehandle_pairs += _count_increasing_pairs(layers)

    if denominator == 'same':
        denom = total_same_stack_pairs
    elif denominator == 'all':
        denom = N * (N - 1) // 2
    else:
        raise ValueError("denominator must be 'same' or 'all'")

    rate = (rehandle_pairs / denom) if denom > 0 else 0.0
    return int(rehandle_pairs), int(denom), float(rate)


def calculation_metrics(sequence, feature):
    """
    # (env,traj,step)
    # (env, node,obs_dim)
    """

    avg_rate = 0
    for i in range(sequence.shape[0]):
        max_traj = 0
        for j in range(sequence.shape[1]):
            from_layer = feature[sequence[i,j], -1]
            from_col = feature[sequence[i,j], -2]
            from_bay = feature[sequence[i,j], -3]
            from_yard = feature[sequence[i,j], -4]
            
            rehandle, denom, rate = compute_rehandle_rate(from_layer, from_col, from_bay, from_yard)
            max_traj = max(max_traj, rate) 
        avg_rate += max_traj
    avg_rate /= sequence.shape[0]

    return rate


def single_batch_graph_data(data):
    # 现在 data 的形状为 (num_nodes, dim)
    if isinstance(data, np.ndarray):
        node_features = torch.tensor(data, dtype=torch.float)
    else:
        node_features = data

    # 初始化边索引和边特征列表
    edge_index_list = []
    edge_attr_list = []
   
    # 情况1：第3列，第4列，第5列的值完全相同，按第6列从小到大建边
    # 提取第3,4,5列特征
    selected_features_1 = node_features[:, 2:5]
    unique_features_1, inverse_indices_1 = torch.unique(selected_features_1, dim=0, return_inverse=True)
    
    for group_id in range(len(unique_features_1)):
        group_indices = torch.where(inverse_indices_1 == group_id)[0]
        num_nodes_in_group = len(group_indices)
        
        if num_nodes_in_group > 1:
            # 按第6列（索引为5）的值排序
            sorted_indices = group_indices[torch.argsort(node_features[group_indices, 5])]
            # 按排序后的顺序建立边
            source_nodes = sorted_indices[:-1]
            target_nodes = sorted_indices[1:]
            edge_index_list.append(torch.stack([source_nodes, target_nodes], dim=0))

    # 情况2：第3列，第4列，第6列的值完全相同，按第5列从小到大建边
    # 提取第3,4,6列特征
    selected_features_2 = torch.cat([node_features[:, 2:4], node_features[:, 5:6]], dim=1)
    unique_features_2, inverse_indices_2 = torch.unique(selected_features_2, dim=0, return_inverse=True)
    
    for group_id in range(len(unique_features_2)):
        group_indices = torch.where(inverse_indices_2 == group_id)[0]
        num_nodes_in_group = len(group_indices)
        
        if num_nodes_in_group > 1:
            # 按第5列（索引为4）的值排序
            sorted_indices = group_indices[torch.argsort(node_features[group_indices, 4])]
            # 按排序后的顺序建立边
            source_nodes = sorted_indices[:-1]
            target_nodes = sorted_indices[1:]
            edge_index_list.append(torch.stack([source_nodes, target_nodes], dim=0))

    if edge_index_list:
        edge_index = torch.cat(edge_index_list, dim=1).long()
        # edge_attr = torch.cat(edge_attr_list, dim=0)
    else:
        # 如果没有满足条件的边，创建空的边索引和边特征
        edge_index = torch.empty((2, 0), dtype=torch.long)
        # edge_attr = torch.empty((0, 1), dtype=torch.float)

    data_graph = Data(x=node_features, edge_index=edge_index)  # edge_attr=edge_attr
    # 因为现在只有一个图，不需要再使用 Batch 包装
    return data_graph
