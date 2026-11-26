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
# Speed fix (dense GATv2): each world has only a handful of robots, so the
# layer computes attention on a small (robots x robots) grid per world with an
# in-range mask, instead of PyG's sparse edge lists. The maths and the weights
# are the same as torch_geometric's GATv2Conv, which tests/test_graph.py checks.
# The sparse path is kept and selected with dense=False (model.comm.dense).
#
# To undo: see "How to revert CoDreamer" in third_party/r2dreamer/UPSTREAM.md.
import torch
import torch.nn.functional as F
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
        eye = torch.eye(n_agents, dtype=torch.bool, device=positions.device)
        # (G, N, N): distance[g, i, j] between receiver i and sender j. The
        # diagonal is set to exactly 0, the edge feature GATv2Conv gives its
        # self-loops (fill_value=0.0).
        self.distance = torch.cdist(grouped, grouped).masked_fill(eye, 0.0)
        # Sec. 2.2.1, CoDreamer: an edge exists when the distance is at most C.
        # Robots never send to themselves through an edge.
        self.adjacency = (self.distance <= comm_range) & ~eye
        # Dense attention mask: real edges plus the self-loop GATv2Conv adds.
        self.mask = self.adjacency | eye
        self._edge_index = None
        self._edge_attr = None

    def _edges(self) -> None:
        # Sparse edge list for the PyG path, built only when asked for, because
        # nonzero() makes the GPU wait for the CPU to learn the edge count.
        if self._edge_index is None:
            group, receiver, sender = self.adjacency.nonzero(as_tuple=True)
            # Turn (group, robot) pairs into flat node numbers, so all groups
            # form one big disconnected graph processed in a single call.
            offset = group * self.n_agents
            # PyG convention: row 0 holds senders, row 1 holds receivers.
            self._edge_index = torch.stack([sender + offset, receiver + offset])
            # Sec. 2.2.1, CoDreamer: the edge feature is the distance.
            self._edge_attr = self.distance[group, receiver, sender].unsqueeze(-1)

    @property
    def edge_index(self) -> torch.Tensor:
        self._edges()
        return self._edge_index

    @edge_index.setter
    def edge_index(self, value: torch.Tensor) -> None:
        # Link models (for example keep_edges in co_imaginer/evaluation.py) drop
        # edges by assigning a smaller edge list. The dense mask is rebuilt
        # from it, so the dense layer sees exactly the same dropped links as
        # the sparse one. New tensors are assigned, so a copy.copy of a graph
        # can drop edges without changing the original.
        self._edges()
        self._edge_index = value
        group = value[1] // self.n_agents
        adjacency = torch.zeros_like(self.adjacency)
        adjacency[group, value[1] % self.n_agents, value[0] % self.n_agents] = True
        self.adjacency = adjacency
        eye = torch.eye(self.n_agents, dtype=torch.bool, device=value.device)
        self.mask = adjacency | eye

    @property
    def edge_attr(self) -> torch.Tensor:
        self._edges()
        return self._edge_attr

    @edge_attr.setter
    def edge_attr(self, value: torch.Tensor) -> None:
        # Only the sparse path reads edge_attr. The dense path always uses the
        # distance itself, which is what edge_attr holds for every kept edge.
        self._edges()
        self._edge_attr = value

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
    def __init__(self, in_dim: int, units: int, heads: int, dense: bool = True) -> None:
        super().__init__()
        # Brody et al. 2022, GATv2. Each head produces units // heads numbers and
        # the heads are concatenated, so the message is `units` wide. Averaging
        # full-width heads instead made every layer about 5M parameters on the
        # 2,560-wide Dreamer state. edge_dim=1 feeds the distance into the
        # attention score. Self-loops get distance 0.
        if units % heads:
            raise ValueError("units must be divisible by heads")
        # The PyG module holds the weights for both paths, so parameter names,
        # initialisation and saved checkpoints do not depend on `dense`.
        self.conv = GATv2Conv(
            in_dim,
            units // heads,
            heads=heads,
            concat=True,
            edge_dim=1,
            add_self_loops=True,
            fill_value=0.0,
        )
        self.dense = dense
        # Sec. 2.2.2, CoDreamer: the layer output is joined to the original node
        # features, so downstream layers see [own features, received message].
        self.out_dim = in_dim + units

    def _dense_conv(self, nodes: torch.Tensor, graph: AgentGraph) -> torch.Tensor:
        # nodes (G, N, F) -> messages (G, N, H * C). Same steps as
        # GATv2Conv.forward and GATv2Conv.edge_update, on a full N x N grid.
        conv = self.conv
        heads, channels = conv.heads, conv.out_channels
        groups, n_agents, _ = nodes.shape
        # Brody et al. 2022, Eq. 7: x_l is the sender projection, x_r the
        # receiver projection. (G, N, H, C) each.
        x_l = conv.lin_l(nodes).view(groups, n_agents, heads, channels)
        x_r = conv.lin_r(nodes).view(groups, n_agents, heads, channels)
        # Edge feature projection of the distance, no bias. (G, N, N, H, C).
        distance = graph.distance.to(x_l.dtype).unsqueeze(-1)
        edge = conv.lin_edge(distance)
        edge = edge.view(groups, n_agents, n_agents, heads, channels)
        # score[g, i, j, h] for receiver i and sender j:
        # att . LeakyReLU(x_r[i] + x_l[j] + edge[i, j]).
        hidden = x_r.unsqueeze(2) + x_l.unsqueeze(1) + edge
        hidden = F.leaky_relu(hidden, conv.negative_slope)
        score = (hidden * conv.att.view(1, 1, 1, heads, channels)).sum(-1)
        # Softmax over senders j that are in range of i, plus i itself. The
        # self-loop guarantees every row has at least one allowed sender.
        score = score.masked_fill(~graph.mask.unsqueeze(-1), float("-inf"))
        # The softmax runs in at least float32, also under bf16 autocast.
        precise = torch.promote_types(score.dtype, torch.float32)
        alpha = torch.softmax(score.to(precise), dim=2).to(x_l.dtype)
        alpha = F.dropout(alpha, p=conv.dropout, training=self.training)
        # Weighted sum of sender projections (G, N, H, C), heads concatenated.
        out = torch.einsum("gijh,gjhc->gihc", alpha, x_l)
        out = out.reshape(groups, n_agents, heads * channels)
        if conv.bias is not None:
            out = out + conv.bias
        return out

    def forward(self, x: torch.Tensor, graph: AgentGraph) -> torch.Tensor:
        # x may arrive flattened, for example (B*T, F) during imagination.
        # Reshape it to the graph's leading shape first, and restore it at the end.
        shape = x.shape
        x = x.reshape(*graph.lead, shape[-1])
        nodes = graph.group(x)
        if self.dense:
            messages = self._dense_conv(nodes, graph)
        else:
            # Original sparse path through torch_geometric's message passing.
            messages = self.conv(
                nodes.reshape(-1, shape[-1]), graph.edge_index, graph.edge_attr
            )
        messages = graph.ungroup(messages).to(x.dtype)
        return torch.cat([x, messages], -1).reshape(*shape[:-1], self.out_dim)
