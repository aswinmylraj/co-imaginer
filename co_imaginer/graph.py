# CoDreamer communication graph and message layer, PLAN.md item 7.
#
# CoDreamer (Toledo and Prorok, arXiv:2406.13600) lets robots exchange messages
# with a graph neural network. This file holds the two pieces every CoDreamer
# module uses:
#   AgentGraph  who can talk to whom: robots of the same world within range C.
#   GraphLayer  one GATv2 attention layer that sends messages along those edges
#               and appends the received message to each robot's own features.
# The RSSM (third_party/r2dreamer/rssm.py) and the reward, continue, actor and
# critic heads (networks.py, dreamer.py) call these. With model.comm.enabled
# False, nothing here is used and the model is exactly IDreamer.
#
# To undo: see "How to revert CoDreamer" in third_party/r2dreamer/UPSTREAM.md.
import torch
from torch import nn
from torch_geometric.nn import GATv2Conv


class AgentGraph:
    # Communication graph for a batch whose first dimension holds robots world by
    # world: rows 0..N-1 are world 0, rows N..2N-1 are world 1, and so on. That is
    # the row order VmasAgents produces and the world-aligned replay sampler keeps.
    def __init__(
        self, positions: torch.Tensor, n_agents: int, comm_range: float
    ) -> None:
        # positions: (B, *rest, 2). "rest" is any extra axes such as time.
        self.n_agents = n_agents
        self.comm_range = comm_range
        self.positions = positions
        # Leading shape of every feature tensor this graph will be applied to.
        self.lead = positions.shape[:-1]
        # (G, N, 2): one group per world and per time step, N robots in each.
        grouped = self.group(positions.float())
        distance = torch.cdist(grouped, grouped)
        # A robot never sends to itself here. GATv2Conv adds its own self-loop.
        not_self = ~torch.eye(n_agents, dtype=torch.bool, device=positions.device)
        # Sec. 2.2.1, CoDreamer: an edge exists when the distance is at most C.
        adjacency = (distance <= comm_range) & not_self
        group, receiver, sender = adjacency.nonzero(as_tuple=True)
        # Turn (group, robot) pairs into flat node numbers, so all groups form
        # one big disconnected graph that GATv2Conv processes in a single call.
        offset = group * n_agents
        # PyG convention: row 0 holds senders, row 1 holds receivers.
        self.edge_index = torch.stack([sender + offset, receiver + offset])
        # Sec. 2.2.1, CoDreamer: the edge feature is the distance between robots.
        self.edge_attr = distance[group, receiver, sender].unsqueeze(-1)

    def group(self, x: torch.Tensor) -> torch.Tensor:
        # (B, *rest, F) -> (worlds, N, R, F) -> (worlds, R, N, F) -> (G, N, F),
        # where R is the product of the rest axes and G = worlds * R.
        worlds = x.shape[0] // self.n_agents
        x = x.reshape(worlds, self.n_agents, -1, x.shape[-1]).transpose(1, 2)
        return x.reshape(-1, self.n_agents, x.shape[-1])

    def ungroup(self, x: torch.Tensor) -> torch.Tensor:
        # Exact inverse of group: (G, N, F) or (G * N, F) -> (B, *rest, F).
        worlds = self.lead[0] // self.n_agents
        x = x.reshape(worlds, -1, self.n_agents, x.shape[-1]).transpose(1, 2)
        return x.reshape(*self.lead, x.shape[-1])

    def repeat_steps(self, steps: int) -> "AgentGraph":
        # Same edges copied over a new trailing time axis. Used for imagination,
        # where CoDreamer keeps the starting graph for every imagined step.
        positions = self.positions.unsqueeze(-2)
        positions = positions.expand(*self.lead, steps, positions.shape[-1])
        return AgentGraph(positions, self.n_agents, self.comm_range)


class GraphLayer(nn.Module):
    # One round of GATv2 message passing between robots in the same world.
    def __init__(self, in_dim: int, units: int, heads: int) -> None:
        super().__init__()
        # Brody et al. 2022, GATv2. Each head produces units // heads numbers and
        # the heads are concatenated, so the message is `units` wide. Averaging
        # full-width heads instead made every layer about 5M parameters on the
        # 2,560-wide Dreamer state. edge_dim=1 feeds the distance into the
        # attention score. Self-loops get distance 0.
        if units % heads:
            raise ValueError("units must be divisible by heads")
        self.conv = GATv2Conv(
            in_dim,
            units // heads,
            heads=heads,
            concat=True,
            edge_dim=1,
            add_self_loops=True,
            fill_value=0.0,
        )
        # Sec. 2.2.2, CoDreamer: the layer output is joined to the original node
        # features, so downstream layers see [own features, received message].
        self.out_dim = in_dim + units

    def forward(self, x: torch.Tensor, graph: AgentGraph) -> torch.Tensor:
        # x may arrive flattened, for example (B*T, F) during imagination.
        # Reshape it to the graph's leading shape first, and restore it at the end.
        shape = x.shape
        x = x.reshape(*graph.lead, shape[-1])
        nodes = graph.group(x).reshape(-1, shape[-1])
        messages = self.conv(nodes, graph.edge_index, graph.edge_attr)
        messages = graph.ungroup(messages).to(x.dtype)
        return torch.cat([x, messages], -1).reshape(*shape[:-1], self.out_dim)
