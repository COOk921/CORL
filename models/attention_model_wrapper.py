import torch
import numpy as np
from .nets.attention_model.attention_model import *
from .nets.attention_model.gcn import GCN
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
    num_edges = num_nodes * 5  # 随机建立num_nodes*5条边
    data_graphs = []
    for b in range(batch):
        batch_node_features = node_features[b]
        # 随机生成源节点和目标节点
        source_nodes = torch.randint(0, num_nodes, (num_edges,))
        target_nodes = torch.randint(0, num_nodes, (num_edges,))
        # 确保源节点和目标节点不相同
        while torch.any(source_nodes == target_nodes):
            diff_mask = source_nodes == target_nodes
            target_nodes[diff_mask] = torch.randint(0, num_nodes, (diff_mask.sum(),))
        edge_index = torch.stack([source_nodes, target_nodes], dim=0).long()

        # 计算边的特征（距离）
        edge_attr = torch.norm(batch_node_features[source_nodes] - batch_node_features[target_nodes], p=2, dim=-1).view(-1, 1)

        data_graph = Data(x=batch_node_features, edge_index=edge_index, edge_attr=edge_attr)
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
        

    def forward(self, obs):
        state = stateWrapper(obs, device=self.device, problem=self.problem.NAME)
        input = state.states["observations"]


        embedding = self.embedding(input)
        encoded_inputs, _ = self.encoder(embedding)

        # decoding
        cached_embeddings = self.decoder._precompute(encoded_inputs)
        logits, glimpse = self.decoder.advance(cached_embeddings, state)

        return logits, glimpse

    def encode(self, obs):
        state = stateWrapper(obs, device=self.device, problem=self.problem.NAME)
        input = state.states["observations"]        # [batch, num_node, dim]
        
        """构建图模型 """
        graph = graph_data(input).to(self.device)
      
        gcn = GCN(graph.x.shape[-1], self.embedding_dim,self.embedding_dim, self.embedding_dim).to(self.device)
        out = gcn(graph)
      
        embedding = out.view(input.shape[0], input.shape[1], -1)
        encoded_inputs =  embedding

        """embedding + MHA """
        # embedding = self.embedding(input)
        # encoded_inputs, _ = self.encoder(embedding)     # [batch,num_node,hidden_dim]
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
        x = self.backbone(x)
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
