import torch

from .nets.attention_model.attention_model import *
from .nets.attention_model.gcn import GCN
from torch_geometric.data import Data, Batch

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
        input = state.states["observations"]

        """构建图模型 """
        # graphs = []
        # for b in range(len(obs["graph_data"])):
        #     graphs.append(obs["graph_data"][b].to(self.device))
        
        # graph = Batch.from_data_list(graphs)
        # num_samples = input.shape[1]

        #embedding = []
        # gcn = GCN(graph.x.shape[1], self.embedding_dim, self.embedding_dim).to(self.device)
        # out = gcn(graph.x, graph.edge_index)
        # start = 0
        # for batch in range(len(obs["graph_data"])):
        #     num_nodes = obs["graph_data"][batch].num_nodes  # 121,125,131...
        #     wm_nodes = out[start : start + num_samples] 
        #     start += num_nodes
        #     embedding.append(wm_nodes)
        # embedding = torch.stack(embedding).to(self.device)

        embedding = self.embedding(input)
        encoded_inputs, _ = self.encoder(embedding)
        cached_embeddings = self.decoder._precompute(encoded_inputs)  # [batch,num_node,hidden_dim]
      
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
