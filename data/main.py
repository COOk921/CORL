import pandas as pd
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data
from sklearn.utils import shuffle
from itertools import permutations
from sklearn.preprocessing import StandardScaler,OneHotEncoder
import copy

import pdb

# ==============================================================================
# 步骤 1: 数据准备 - 生成训练用的Pair
# ==============================================================================

def create_pairwise_data(df: pd.DataFrame, feature_cols: list, window_size: int):
    """
    为单个DataFrame生成所有窗口内的pair数据和标签.

    Args:
        df (pd.DataFrame): 输入的DataFrame, 原始顺序即为Ground Truth.
        feature_cols (list): 用于计算的特征列名列表.
        window_size (int): 滑动窗口的大小 (D).

    Returns:
        tuple: 包含两个元素的元组:
               - X_pairs (np.ndarray): N x (2 * num_features) 的数组, 每行为一个pair的拼接特征.
               - y_labels (np.ndarray): N x 1 的数组, 每行是一个pair的标签 (0或1).
    """
    # 记录Ground Truth顺序
    df['ground_truth_order'] = np.arange(len(df))

    # 打乱DataFrame
    df_shuffled = shuffle(df, random_state=42).reset_index(drop=True)

    # 提取特征和顺序信息
    features = df_shuffled[feature_cols].values
    orders = df_shuffled['ground_truth_order'].values


   
    X_pairs = []
    y_labels = []
    pair_original_indices = []

    # 滑动窗口生成pairs
    for i in range(len(df_shuffled) - window_size + 1):
        window_features = features[i : i + window_size]
        window_orders = orders[i : i + window_size]

        # 第一个node作为锚点
        anchor_feature = window_features[0]
        anchor_order = window_orders[0]

        # 与窗口内其他node生成pair
        for j in range(1, window_size):
            other_feature = window_features[j]
            other_order = window_orders[j]

            # 特征拼接
            pair_feature = np.concatenate([anchor_feature, other_feature])
            X_pairs.append(pair_feature)

            # 生成标签
            label = 1 if anchor_order < other_order else 0
            y_labels.append(label)

            pair_original_indices.append([anchor_order, other_order])

    if not X_pairs:
        # 如果DataFrame的行数小于窗口大小，则返回空数组
        num_features = len(feature_cols)
        return np.array([]).reshape(0, 2 * num_features), np.array([]),np.array([])

    return np.array(X_pairs), np.array(y_labels),pair_original_indices


# ==============================================================================
# 步骤 2: 建立模型
# ==============================================================================

class PairwiseRankingModel(nn.Module):
    """
    一个简单的MLP模型，用于预测pair的正确顺序概率.
    输入是一个拼接了两个node特征的向量.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        """
        Args:
            input_dim (int): 输入特征的维度 (等于 2 * num_node_features).
            hidden_dim (int): 隐藏层的维度.
        """
        super(PairwiseRankingModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  # 输出0到1之间的置信度
        )

    def forward(self, x):
        return self.network(x)


# ==============================================================================
# 步骤 3: 训练模型
# ==============================================================================

def train_model(model: nn.Module, X_train: np.ndarray, y_train: np.ndarray, epochs: int = 50, learning_rate: float = 0.001,device: torch.device = torch.device('cpu')):
    """
    训练单个模型.

    Args:
        model (nn.Module): 待训练的模型实例.
        X_train (np.ndarray): 训练数据特征.
        y_train (np.ndarray): 训练数据标签.
        epochs (int): 训练轮数.
        learning_rate (float): 学习率.
    """
    if X_train.shape[0] == 0:
        print("    警告: 没有可供训练的pair数据，跳过训练。")
        return
   
    # 转换为PyTorch Tensors
    X_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(device)
    model.to(device)

    # 定义损失函数和优化器
    criterion = nn.BCELoss()  # 二元交叉熵损失
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print(f"    开始训练... 共 {epochs} 个 epochs.")
    # 训练循环
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"    Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
    print("    训练完成.")


# ==============================================================================
# 步骤 4 & 5: 预测、生成邻接矩阵并创建PyG图
# ==============================================================================
def build_graph_from_pairs(
    model: nn.Module,
    df: pd.DataFrame,
    feature_cols_for_graph: list,
    X_pairs: np.ndarray,
    pair_original_indices: np.ndarray,
    threshold: float,
    device: torch.device = torch.device('cpu')
):
    """
    使用训练好的模型对生成过的pair进行预测，构建带权有向图.

    Args:
        model (nn.Module): 训练好的模型.
        df (pd.DataFrame): 原始DataFrame，用于提取节点特征.
        feature_cols_for_graph (list): 用于图节点特征的列名.
        X_pairs (np.ndarray): 在第一步中生成的pair特征数据.
        pair_original_indices (np.ndarray): 每个pair对应的原始节点索引 <源, 目标>.
        threshold (float): 置信度阈值P.

    Returns:
        torch_geometric.data.Data: 创建好的PyG图对象.
    """
    
    if X_pairs.shape[0] == 0:
        print("    警告: 没有pair数据，创建一个空图。")
        x = torch.tensor(df[feature_cols_for_graph].values, dtype=torch.float32)
        edge_index = torch.empty((2, 0), dtype=torch.long)
        return Data(x=x, edge_index=edge_index)

    model.to(device)
    model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X_pairs, dtype=torch.float32).to(device)
        # 批量预测所有pair的置信度
        confidences = model(X_tensor).squeeze().cpu().numpy()

    # 根据阈值筛选边
    edge_mask = confidences > threshold
    
    # 获取通过筛选的边的索引和特征
    filtered_indices = np.array(pair_original_indices)[edge_mask]
    filtered_confidences = confidences[edge_mask]
    
    # --- 创建PyG图 ---
    # 1. 节点特征 (x)
    x = torch.tensor(df[feature_cols_for_graph].values, dtype=torch.float32)

    if filtered_indices.shape[0] > 0:
        # 2. 边索引 (edge_index) - [2, num_edges]
        edge_index = torch.tensor(filtered_indices.T, dtype=torch.long)
        # 3. 边特征 (edge_attr) - [num_edges, num_edge_features]
        edge_attr = torch.tensor(filtered_confidences, dtype=torch.float32).view(-1, 1)
    else:
        # 如果没有边通过阈值，则创建空边
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 1), dtype=torch.float32)

    
    # 4. 创建Data对象
    graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    print(f"    图已创建: {graph}")
    print(f"    根据阈值 {threshold}，共添加了 {edge_index.shape[1]} 条边.")
   
    return graph


def read_data(file_path: str,continuous_features:list,categorical_features:list) -> dict:
    with open(file_path, 'rb') as f:
        data = pd.read_pickle(f)
    data = {tuple(key) if isinstance(key, np.ndarray) else key: value for key, value in data.items()}
    
    processed_data_local = {}
    for key, df in data.items():
        processed_df = pd.DataFrame(index=df.index)
        if 'Unit Nbr' in df.columns:
            processed_df['Unit Nbr'] = df['Unit Nbr']
        if 'Time Completed' in df.columns:
            processed_df['Time Completed'] = df['Time Completed']

        # 转化连续特征
        local_scaler = StandardScaler()
        scaled_continuous = local_scaler.fit_transform(df[continuous_features])
        for i, col_name in enumerate(continuous_features):
            processed_df[col_name] = scaled_continuous[:, i]
        
        local_vocab_mappings = {}
        for col in categorical_features:
            unique_categories = df[col].unique()
            local_vocab_mappings[col] = {category: i + 1 for i, category in enumerate(unique_categories)}
            local_vocab_mappings[col]['[UNK]'] = 0 # 同样为未知类别保留0
            
            processed_df[col] = df[col].map(local_vocab_mappings[col]).fillna(0).astype(int)
    
        processed_data_local[key] = processed_df

    return processed_data_local

# ==============================================================================
# 步骤 6: 主流程
# ==============================================================================

def process_data_pipeline(
    data_dict: dict,
    feature_cols_for_model: list,
    feature_cols_for_graph: list,
    window_size: int,
    threshold: float,
    epochs: int = 50,
    learning_rate: float = 0.001,
    hidden_dim: int = 64
) -> dict:
    """
    处理整个数据字典，为每个DataFrame训练模型并生成图.

    Args:
        data_dict (dict): key为ID, value为DataFrame的输入字典.
        feature_cols_for_model (list): 用于训练排序模型的特征列.
        feature_cols_for_graph (list): 用于最终图节点表示的特征列.
        window_size (int): 滑动窗口大小D.
        threshold (float): 生成邻接矩阵的置信度阈值P.
        epochs (int): 模型训练轮数.
        learning_rate (float): 学习率.
        hidden_dim (int): 模型隐藏层维度.

    Returns:
        dict: 嵌套字典, 包含每个DataFrame的原始数据和生成的图.
    """
    final_results = {}
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    idx= 0
    for key, df in data_dict.items():
        print(f"\n===== 开始处理 DataFrame: '{key}-{idx}/{len(data_dict)}' =====")
        idx+=1
        # 深拷贝以防修改原始数据
        original_df = copy.deepcopy(df)
        processing_df = copy.deepcopy(df)

        # 步骤 1: 创建训练数据
        print("步骤 1: 创建pair训练数据...")
        X_train, y_train,pair_indices = create_pairwise_data(processing_df, feature_cols_for_model, window_size)
        
        print(f"    为 '{key}' 生成了 {len(y_train)} 个训练pair.")

        # 步骤 2: 初始化模型
        print("步骤 2: 初始化模型...")
        input_dim = 2 * len(feature_cols_for_model)
        model = PairwiseRankingModel(input_dim=input_dim, hidden_dim=hidden_dim)
        print(f"    模型已创建，输入维度: {input_dim}")

        # 步骤 3: 训练模型
        print("步骤 3: 训练模型...")
        train_model(model, X_train, y_train, epochs=epochs, learning_rate=learning_rate,device=device)

        # 步骤 4 & 5: 生成邻接矩阵并创建图
        print("步骤 4 & 5: 生成邻接矩阵并创建PyG图...")
        pyg_graph = build_graph_from_pairs(
            model=model,
            df=original_df,
            feature_cols_for_graph=feature_cols_for_graph,
            X_pairs=X_train, # 用来预测的pair就是训练的pair
            pair_original_indices=pair_indices,
            threshold=threshold,
            device=device
        )

        # 步骤 6: 存储结果
        final_results[key] = {
            'data': original_df,
            'graph': pyg_graph
        }
        print(f"===== DataFrame '{key}' 处理完成 =====")

        

    return final_results


# ==============================================================================
# 示例：如何使用
# ==============================================================================
if __name__ == '__main__':
    # --- 1. 准备模拟数据 ---
    # 假设我们有两个DataFrame, 分别代表两个独立的序列
    # data_A = pd.DataFrame({
    #     'feat_1': np.linspace(0, 1, 10),
    #     'feat_2': np.random.rand(10),
    #     'feat_3': np.sin(np.linspace(0, np.pi, 10)),
    #     'text_id': [f'node_{i}' for i in range(10)] # 非计算特征
    # })

    # data_B = pd.DataFrame({
    #     'feat_1': np.linspace(1, 0, 15),
    #     'feat_2': np.random.rand(15),
    #     'feat_3': np.cos(np.linspace(0, np.pi, 15)),
    #     'text_id': [f'node_{i}' for i in range(15)]
    # })

    # # 创建输入的总字典
    # data = {
    #     'sequence_A': data_A,
    #     'sequence_B': data_B
    # }

    # --- 2. 设定超参数 ---
    # 你需要手动选择的参数

    continuous_features = ['Unit Weight (kg)']
    categorical_features = ['Unit POD', 'from_yard', 'from_bay', 'from_col', 'from_layer', ]

    # 用于模型训练的特征
    FEATURES_FOR_MODEL = ['Unit Weight (kg)','Unit POD', 'from_yard', 'from_bay', 'from_col', 'from_layer']
    # 用于最终图节点表示的特征 (可以和模型特征相同或不同)
    FEATURES_FOR_GRAPH = ['Unit Weight (kg)','Unit POD', 'from_yard', 'from_bay', 'from_col', 'from_layer']

    # 其他参数
    D_WINDOW_SIZE = 4       # 滑动窗口大小
    P_THRESHOLD = 0.5      # 置信度阈值
    EPOCHS = 40             # 训练轮数
    LEARNING_RATE = 0.005   # 学习率
    HIDDEN_DIM = 128        # 模型隐藏层大小
   
    # 路径
    READ_PATH = "./data/container_data_cluster.pkl"
    WRITE_PATH = "./data/processed_container_data_cluster.pkl"
    # 读取数据
    data = read_data(READ_PATH,continuous_features,categorical_features)

    # --- 3. 运行主流程 ---
    final_output = process_data_pipeline(
        data_dict=data,
        feature_cols_for_model=FEATURES_FOR_MODEL,
        feature_cols_for_graph=FEATURES_FOR_GRAPH,
        window_size=D_WINDOW_SIZE,
        threshold=P_THRESHOLD,
        epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        hidden_dim=HIDDEN_DIM
    )
    
    with open(WRITE_PATH, 'wb') as f:
        pickle.dump(final_output, f)

    
    # --- 4. 查看结果 ---
    # print("\n\n#################### 最终输出 ####################")
    # for key, value in final_output.items():
    #     print(f"\n--- 结果 for '{key}' ---")
    #     print("  - 'data' (原始DataFrame):")
    #     print(value['data'].head())
    #     print("\n  - 'graph' (PyG图对象):")
    #     print(value['graph'])
    #     print(f"    图是否有向: {value['graph'].is_directed()}")