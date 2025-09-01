import gym
import numpy as np
from gym import spaces
import torch

from discriminator.model import Discriminator,PairClassifier
import time
import pickle
import pdb

import torch
from torch_geometric.data import Data
from torch_geometric.data import Data
import torch

_DATA_CACHE = None
_MODEL_CACHE = None



root_dir = "data/container_data.pkl"
model_path = "./discriminator/model/discriminator.pth"


def get_data(max_nodes,data_path="./data/processed_container_data.pkl",  mode = 'train'):

    global _DATA_CACHE
    selected_columns = ['Unit Weight (kg)','Unit POD',  'from_yard', 'from_bay', 'from_col', 'from_layer']

    if _DATA_CACHE is None:
        print("--- Loading data from file (will happen only ONCE) ---")
        with open(data_path, 'rb') as f:
            data = pickle.load(f)

        data = {tuple(key) if isinstance(key, np.ndarray) else key: value for key, value in data.items()}

        _DATA_CACHE = data
        keys = list(data.keys())
    else:
        keys = list(_DATA_CACHE.keys())
    
    if mode == 'train':
        key = np.random.default_rng().choice(keys)

    df = _DATA_CACHE[tuple(key)]
    nodes = df[selected_columns].to_numpy()[:max_nodes]
    
    if len(nodes) < max_nodes:
        nodes = np.pad(nodes, ((0, max_nodes - len(nodes)), (0, 0)), mode='constant')
    
    return nodes

def get_discriminator_reward(dest_node,prev_node,input_dim, hidden_dim, device ,model_path = model_path):

    global _MODEL_CACHE
    if _MODEL_CACHE is None:
        print("--- Loading model from file (will happen only ONCE) ---")
        
        model_for_inference = Discriminator(
            input_dim = input_dim,
            hidden_dim = hidden_dim,
        ).to(device)
        # model_for_inference = PairClassifier(
        #     dim = input_dim,
        #     hidden_dim = hidden_dim,
        # ).to(device)


        model_for_inference.load_state_dict(torch.load(model_path))

        _MODEL_CACHE = model_for_inference
   
    _MODEL_CACHE.eval()

    with torch.no_grad():
        dest_node = torch.from_numpy(dest_node).float().to(device)
        prev_node = torch.from_numpy(prev_node).float().to(device)

        similarity_score = _MODEL_CACHE(dest_node, prev_node)
        # similarity_score = torch.round(similarity_score).squeeze().detach().cpu().numpy()
        similarity_score = similarity_score.squeeze().detach().cpu().numpy()
    

    return similarity_score

def similarity_reward( x, y, eps=1e-8, pad_value=0.0):
    dot_product = np.sum(x * y, axis=-1)  # Shape: [batch, n_traj]
    # 计算 x 和 y 的 L2 范数
    norm_x = np.linalg.norm(x, axis=-1)  # Shape: [batch, n_traj]
    norm_y = np.linalg.norm(y, axis=-1)  # Shape: [batch, n_traj]
    
    # 创建填充掩码，标记 norm_x 或 norm_y 为 0 的位置
    pad_mask = (norm_x == 0) | (norm_y == 0)  # Shape: [batch, n_traj]
    
    # 初始化输出数组，填充 pad_value
    sim = np.full_like(norm_x, pad_value)  # Shape: [batch, n_traj]
    
    # 有效掩码：norm_x 和 norm_y 都不为 0 的位置
    valid_mask = ~pad_mask  # Shape: [batch, n_traj]
    
    # 如果存在有效位置，计算余弦相似度
    if np.any(valid_mask):
        # 提取有效位置的点积和范数
        dot_product_valid = dot_product[valid_mask]  # Shape: [num_valid]
        norm_x_valid = norm_x[valid_mask]  # Shape: [num_valid]
        norm_y_valid = norm_y[valid_mask]  # Shape: [num_valid]
        
        # 计算余弦相似度
        cos_sim = dot_product_valid / (norm_x_valid * norm_y_valid + eps)
        cos_sim = np.clip(cos_sim, -1.0, 1.0)
        
        # 映射到 [0, 1]
        sim[valid_mask] = (cos_sim + 1) / 2
    return sim

def assign_env_config(self, kwargs):
    """
    Set self.key = value, for each key in kwargs
    """
    for key, value in kwargs.items():
        setattr(self, key, value)

def read_pkl(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

def graph_data(data):
    # 获取节点数量和特征维度
    num_nodes = data.shape[0]
    dim = data.shape[1]

    # 将数据转换为 PyTorch 张量，并确保数据类型为 float
    if isinstance(data, np.ndarray):
        node_features = torch.tensor(data, dtype=torch.float)
    else:
        node_features = data

    import time
    begin = time.time()
    source_nodes = []
    target_nodes = []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                source_nodes.append(i)
                target_nodes.append(j)
    edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)

    edge_attr_list = []
    for i, j in zip(source_nodes, target_nodes):
        dist = torch.norm(node_features[i] - node_features[j], p=2)
        edge_attr_list.append(dist)
    edge_attr = torch.tensor(edge_attr_list, dtype=torch.float).view(-1, 1)

    # 创建 PyG 的 Data 对象
    data_graph = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr)
    end = time.time()
    print(f"time: {end - begin}")

    return data_graph



class ContainerVectorEnv(gym.Env):
    def __init__(self, *args, **kwargs):
        self.max_nodes = 50
        self.n_traj = 50
        self.dim = 6  # Default feature dimension, override via kwargs
        self.hidden_dim = 256
        self.eval_data = True
        self.eval_partition = "test"
        self.eval_data_idx = 0
        assign_env_config(self, kwargs)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        obs_dict = {
            "observations": spaces.Box(low=0, high=1, shape=(self.max_nodes, self.dim)),
            "action_mask": spaces.MultiBinary([self.n_traj, self.max_nodes]),
            "first_node_idx": spaces.MultiDiscrete([self.max_nodes] * self.n_traj),
            "last_node_idx": spaces.MultiDiscrete([self.max_nodes] * self.n_traj),
            "is_initial_action": spaces.Discrete(1),
            # "graph_data": spaces.Graph(
            #     node_space=spaces.Box(low=0, high=1, shape=(self.max_nodes,)), 
            #     edge_space=None
            #     )

        }

        self.observation_space = spaces.Dict(obs_dict)
        self.action_space = spaces.MultiDiscrete([self.max_nodes] * self.n_traj)
        self.reward_space = None
        
        self.reset()

    def seed(self, seed):
        np.random.seed(seed)

    def reset(self):
        self.visited = np.zeros((self.n_traj, self.max_nodes), dtype=bool)
        self.num_steps = 0
        self.last = np.zeros(self.n_traj, dtype=int)
        self.first = np.zeros(self.n_traj, dtype=int)
        
        if self.eval_data:
            self._load_orders()
        else:
            self._generate_orders()

        # self.graph_data = graph_data(self.nodes)
        self.state = self._update_state()
        self.info = {}
        self.done = False
        return self.state

    def _load_orders(self):
        # Load container features, assuming dataset provides (max_nodes, dim) arrays
        self.nodes = get_data(max_nodes = self.max_nodes, mode='train')
        
        
    def _generate_orders(self):
        self.nodes = np.random.rand( self.max_nodes, self.dim)
        
    def step(self, action):

        self._go_to(action)
        self.num_steps += 1
        self.state = self._update_state()
        self.done = (action == self.first) & self.is_all_visited()
        self.info
        return self.state, self.reward, self.done, self.info # 

    def is_all_visited(self):
        return self.visited.all(axis=1)

    def _go_to(self, destination):
        self.dest_node = self.nodes[destination]  # (n_traj, dim)
        self.prev_node = self.nodes[self.last]  # (n_traj, dim)
      
        """ reward 在文件 syncVectorEnvPomo.py 中计算 """
        if self.num_steps != 0:
            # self.reward =  self.similarity(self.dest_node, self.prev_node) # -self.cost(dest_node, prev_node)   
            self.reward = 0  #get_discriminator_reward(self.dest_node, self.prev_node, self.dim, self.hidden_dim, self.device)
        else:
            self.reward = np.zeros(self.n_traj)
            self.first = destination

        self.last = destination
        self.visited[np.arange(self.n_traj), destination] = True
    
    def similarity(self,x, y, eps=1e-8,pad_value=0.0):
      
        dot_product = np.sum(x * y, axis=-1)  # Shape: (n_traj,)
        
        norm_x = np.linalg.norm(x, axis=-1)  # Shape: (n_traj,)
        norm_y = np.linalg.norm(y, axis=-1)  # Shape: (n_traj,)
                
        pad_mask = (norm_x == 0) | (norm_y == 0)
        sim = np.full_like(norm_x, pad_value)

        valid_mask = ~pad_mask
        if np.any(valid_mask):

            dot_product = np.sum(x[valid_mask] * y[valid_mask], axis=-1)
            cos_sim = dot_product / (norm_x[valid_mask] * norm_y[valid_mask] + eps)
            cos_sim = np.clip(cos_sim, -1.0, 1.0)
            sim[valid_mask] = (cos_sim + 1) / 2

        return sim


    def cost(self, loc1, loc2):
        # 
        # ((loc1[:, 0] - loc2[:, 0]) ** 2 + (loc1[:, 1] - loc2[:, 1]) ** 2) ** 0.5
        return np.sqrt(np.sum((loc1 - loc2) ** 2, axis=-1))

    def _update_state(self):
        obs = {
            "observations": self.nodes,
            "action_mask": self._update_mask(),
            "first_node_idx": self.first,
            "last_node_idx": self.last,
            "is_initial_action": self.num_steps == 0,
            # "graph_data": self.graph_data ,
        }
        return obs

    def _update_mask(self):
        action_mask = ~self.visited
        action_mask[np.arange(self.n_traj), self.first] |= self.is_all_visited()
        return action_mask

