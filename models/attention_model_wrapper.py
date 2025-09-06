import torch
import numpy as np
from .nets.attention_model.attention_model import *
from .nets.attention_model.gat import GAT
from torch_geometric.data import Data, Batch
import pdb


def graph_data(data):
    batch, num_nodes, dim = data.shape

    if isinstance(data, np.ndarray):
        node_features = torch.tensor(data, dtype=torch.float)
    else:
        node_features = data

    source_nodes = []
    target_nodes = []
    data_graphs = []
    for b in range(batch):
        batch_node_features = node_features[b]
      
        # 初始化边索引和边特征列表
        edge_index_list = []
        edge_attr_list = []

        # 情况1：第3列，第4列，第5列的值完全相同，按第6列从小到大建边
        # 提取第3,4,5列特征
        selected_features_1 = batch_node_features[:, 2:5]
        unique_features_1, inverse_indices_1 = torch.unique(selected_features_1, dim=0, return_inverse=True)
        
        for group_id in range(len(unique_features_1)):
            group_indices = torch.where(inverse_indices_1 == group_id)[0]
            num_nodes_in_group = len(group_indices)
            
            if num_nodes_in_group > 1:
                # 按第6列（索引为5）的值排序
                sorted_indices = group_indices[torch.argsort(batch_node_features[group_indices, 5])]
                # 按排序后的顺序建立边
                source_nodes = sorted_indices[:-1]
                target_nodes = sorted_indices[1:]
                edge_index_list.append(torch.stack([source_nodes, target_nodes], dim=0))

        # 情况2：第3列，第4列，第6列的值完全相同，按第5列从小到大建边
        # 提取第3,4,6列特征
        selected_features_2 = torch.cat([batch_node_features[:, 2:4], batch_node_features[:, 5:6]], dim=1)
        unique_features_2, inverse_indices_2 = torch.unique(selected_features_2, dim=0, return_inverse=True)
        
        for group_id in range(len(unique_features_2)):
            group_indices = torch.where(inverse_indices_2 == group_id)[0]
            num_nodes_in_group = len(group_indices)
            
            if num_nodes_in_group > 1:
                # 按第5列（索引为4）的值排序
                sorted_indices = group_indices[torch.argsort(batch_node_features[group_indices, 4])]
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

        data_graph = Data(x=batch_node_features, edge_index=edge_index, ) #edge_attr=edge_attr
        data_graphs.append(data_graph)
        
        
    batch_graph = Batch.from_data_list(data_graphs)

    return batch_graph




class Problem:
    def __init__(self, name):
        self.NAME = name

class Backbone(nn.Module):
    def __init__(
        self,
        embedding_dim=128,
        problem_name="tsp",
        n_encode_layers=3,
        tanh_clipping=10.0,
        n_heads=8,
        device="cpu",
    ):
        super(Backbone, self).__init__()
        self.device = device
        self.problem = Problem(problem_name)
        self.embedding = AutoEmbedding(self.problem.NAME, {"embedding_dim": embedding_dim})

        self.embedding_dim = embedding_dim
        self.encoder = GraphAttentionEncoder(
            n_heads=n_heads,
            embed_dim=embedding_dim,
            n_layers=n_encode_layers,
        )

        self.decoder = Decoder(
            embedding_dim, self.embedding.context_dim, n_heads, self.problem, tanh_clipping
        )
        self.gat = GAT(6, self.embedding_dim,self.embedding_dim, self.embedding_dim)

    def forward(self, obs):
        # state = stateWrapper(obs, device=self.device, problem=self.problem.NAME)
        # input = state.states["observations"]

        # embedding = self.embedding(input)
        # encoded_inputs, _ = self.encoder(embedding)

        # # decoding
        # cached_embeddings = self.decoder._precompute(encoded_inputs)
        # logits, glimpse = self.decoder.advance(cached_embeddings, state)

        return #logits, glimpse

    def encode(self, obs):
        state = stateWrapper(obs, device=self.device, problem=self.problem.NAME)
        input = state.states["observations"][:, :, 1:]      # [batch, num_node, dim]
        
        """图模型 """
        # b_graph = obs["graph_data"]       
        # graph = Batch.from_data_list(b_graph)
        # # graph = graph_data(input)        
        # out = self.gat(graph.to(self.device))
       
        # embedding = out.view(input.shape[0], input.shape[1], -1)
        # encoded_inputs =  embedding

        """embedding + MHA """
        embedding = self.embedding(input)
        encoded_inputs, _ = self.encoder(embedding)     # [batch,num_node,hidden_dim]
        cached_embeddings = self.decoder._precompute(encoded_inputs)  

        return cached_embeddings

    def decode(self, obs, cached_embeddings):
        state = stateWrapper(obs, device=self.device, problem=self.problem.NAME)
       
        logits, glimpse = self.decoder.advance(cached_embeddings, state)

        return logits, glimpse


class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()

    def forward(self, x):
        logits = x[0]  # .squeeze(1) # not needed for pomo
        return logits


class Critic(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Critic, self).__init__()
        hidden_size = kwargs["hidden_size"]
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, 1)
        )

    def forward(self, x):
        out = self.mlp(x[1])  # B x T x h_dim --mlp--> B x T X 1
        return out


class Agent(nn.Module):
    def __init__(self, embedding_dim=128, device="cpu", name="tsp"):
        super().__init__()
        self.backbone = Backbone(embedding_dim=embedding_dim, device=device, problem_name=name)
        self.critic = Critic(hidden_size=embedding_dim)
        self.actor = Actor()

    def forward(self, x):  # only actor
        # x = self.backbone(x)
        # logits = self.actor(x)
        # action = logits.max(2)[1]
        # return action, logits
        state = self.backbone.encode(x)
        x = self.backbone.decode(x, state)
        logits = self.actor(x)
        action = logits.max(2)[1]
        return action, logits
        

    def get_value(self, x):
        x = self.backbone(x)
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        x = self.backbone(x)
        logits = self.actor(x)
        probs = torch.distributions.Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)

    def get_value_cached(self, x, state):
        x = self.backbone.decode(x, state)
        return self.critic(x)

    def get_action_and_value_cached(self, x, action=None, state=None):
        if state is None:
            state = self.backbone.encode(x)
            x = self.backbone.decode(x, state)
        else:
            x = self.backbone.decode(x, state)
       
        logits = self.actor(x)
       
        probs = torch.distributions.Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x), state


class stateWrapper:
    """
    from dict of numpy arrays to an object that supplies function and data
    """

    def __init__(self, states, device, problem="tsp"):
        self.device = device
        
        self.states = {}
        for k, v in states.items():
            if k != 'graph_data':
                self.states[k] = torch.tensor(v, device=self.device)
            else:
                self.states[k] = v
        
        if problem == "container":
            self.is_initial_action = self.states["is_initial_action"].to(torch.bool)
            self.first_a = self.states["first_node_idx"]
        elif problem == "tsp":
            self.is_initial_action = self.states["is_initial_action"].to(torch.bool)
            self.first_a = self.states["first_node_idx"]
        elif problem == "cvrp":
            input = {
                "loc": self.states["observations"],
                "depot": self.states["depot"].squeeze(-1),
                "demand": self.states["demand"],
            }
            self.states["observations"] = input
            self.VEHICLE_CAPACITY = 0
            self.used_capacity = -self.states["current_load"]

    def get_current_node(self):
        return self.states["last_node_idx"]

    def get_mask(self):
        return (1 - self.states["action_mask"]).to(torch.bool)
