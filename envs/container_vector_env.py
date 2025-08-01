import gym
import numpy as np
from gym import spaces

from .container_data import ContainerDataset

def assign_env_config(self, kwargs):
    """
    Set self.key = value, for each key in kwargs
    """
    for key, value in kwargs.items():
        setattr(self, key, value)

class ContainerVectorEnv(gym.Env):
    def __init__(self, *args, **kwargs):
        self.max_nodes = 50
        self.n_traj = 50
        self.dim = 10  # Default feature dimension, override via kwargs
        self.eval_data = False
        self.eval_partition = "test"
        self.eval_data_idx = 0
        assign_env_config(self, kwargs)

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
        self.nodes = np.array(ContainerDataset[self.eval_partition, self.max_nodes, self.eval_data_idx])

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

        self.nodes = np.random.rand(self.max_nodes, self.dim)

    def step(self, action):
        self._go_to(action)
        self.num_steps += 1
        self.state = self._update_state()
        self.done = (action == self.first) & self.is_all_visited()
        return self.state, self.reward, self.done, self.info

    def is_all_visited(self):
        return self.visited.all(axis=1)

    def _go_to(self, destination):
        dest_node = self.nodes[destination]  # (n_traj, dim)

        if self.num_steps != 0:
            prev_node = self.nodes[self.last]  # (n_traj, dim)
            self.reward =  -self.cost(dest_node, prev_node)    #self.similarity(dest_node, prev_node)
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
