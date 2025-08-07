import gym
import numpy as np
from gym import spaces
import torch

from .container_data import ContainerDataset
from discriminator.model import Discriminator
import time
import pickle
import pdb

_DATA_CACHE = None
_MODEL_CACHE = None



root_dir = "data/container_data.pkl"
model_path = "discriminator/model/discriminator.pth"

def get_data(max_nodes,data_path="data/processed_container_data.pkl",  mode = 'train'):

    global _DATA_CACHE
    selected_columns = ['Unit Weight (kg)','from_bay', 'from_col', 'from_layer',  'to_bay', 'to_col', 
    'to_layer','Unit POD', 'Unit Type Height', 'Unit Type Length', 'Unit Type ISO', 'from_yard'  ]

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
    else:
        pass

    df = _DATA_CACHE[tuple(key)]

   
    nodes = df[selected_columns].to_numpy()[:max_nodes]

    if len(nodes) < max_nodes:
        nodes = np.pad(nodes, ((0, max_nodes - len(nodes)), (0, 0)), mode='constant')
    
   
    return nodes

def get_discriminator(dest_node,prev_node,input_dim, hidden_dim,device ,model_path = model_path):

    global _MODEL_CACHE
    if _MODEL_CACHE is None:
        print("--- Loading model from file (will happen only ONCE) ---")
        
        model_for_inference = Discriminator(
            input_dim = input_dim,
            hidden_dim = hidden_dim,
        ).to(device)
        model_for_inference.load_state_dict(torch.load(model_path))

        _MODEL_CACHE = model_for_inference
   

  
    dest_node = torch.from_numpy(dest_node).float().to(device)
    prev_node = torch.from_numpy(prev_node).float().to(device)

   

    similarity_score = _MODEL_CACHE(dest_node, prev_node)
    # similarity_score = torch.round(similarity_score).squeeze().detach().cpu().numpy()
    similarity_score = similarity_score.squeeze().detach().cpu().numpy()
    

    return similarity_score




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


class ContainerVectorEnv(gym.Env):
    def __init__(self, *args, **kwargs):
        self.max_nodes = 20
        self.n_traj = 20
        self.dim = 12  # Default feature dimension, override via kwargs
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
        self.state = self._update_state()
        self.info = {}
        self.done = False
        return self.state

    def _load_orders(self):
        # Load container features, assuming dataset provides (max_nodes, dim) arrays
        #self.nodes = ContainerDataset().get_next_data(max_nodes = self.max_nodes, mode='train')
        self.nodes = get_data(max_nodes = self.max_nodes, mode='train')
        
        
    def _generate_orders(self):
        # centers = np.random.rand(2, self.dim)        
        # cluster_sizes = np.full(2, self.max_nodes // 2)
        # cluster_sizes[:self.max_nodes % 2] += 1

        # node_clusters = []
        # for i in range(2):
        #     # 使用标准差为0.1的高斯噪声，让数据集中在中心附近
        #     cluster_nodes = np.random.normal(loc=centers[i], scale=0.01, size=(cluster_sizes[i], self.dim))
        #     # 确保数据在[0, 1]范围内
        #     cluster_nodes = np.clip(cluster_nodes, 0, 1)
        #     node_clusters.append(cluster_nodes)
        
        # self.nodes = np.vstack(node_clusters)

        #self.nodes = np.random.rand(self.max_nodes, self.dim)
        #self.nodes = self.dataset.get_next_data(max_nodes = self.max_nodes, mode='train')
        
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
            #self.reward =  self.similarity(self.dest_node, self.prev_node) # -self.cost(dest_node, prev_node)   
            self.reward = 0 #get_discriminator(self.dest_node, self.prev_node, self.dim, self.hidden_dim, self.device)
        else:
            self.reward = np.zeros(self.n_traj)
            self.first = destination

        self.last = destination
        self.visited[np.arange(self.n_traj), destination] = True
    


    def similarity(self, x, y):
        """
        Compute similarity between two feature vectors: 1 - (sum of squared diffs / dim)
        Returns value in [0, 1], where 1 is identical and 0 is maximally different
        """
        diff = x - y  # (n_traj, dim)
        d2 = np.sum(diff ** 2, axis=-1)  # (n_traj,)
        sim = 1 - (d2 / self.dim)
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
        }
        return obs

    def _update_mask(self):
        action_mask = ~self.visited
        action_mask[np.arange(self.n_traj), self.first] |= self.is_all_visited()
        return action_mask

