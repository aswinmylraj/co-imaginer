import copy

import pytest
import torch

from co_imaginer.graph import AgentGraph, GraphLayer

# Tests for co_imaginer/graph.py: edges stay inside a world, and messages
# never leak from one VMAS world to another.


# Two worlds of two robots each. Rows 0-1 are world 0, rows 2-3 are world 1.
@pytest.mark.parametrize(
    ("positions", "comm_range", "expected_edges"),
    [
        # Both pairs within range: edges both ways inside each world only.
        (
            [[0.0, 0.0], [0.3, 0.0], [0.0, 0.0], [0.3, 0.0]],
            0.5,
            [(0, 1), (1, 0), (2, 3), (3, 2)],
        ),
        # World 1 pair out of range: only world 0 edges remain.
        (
            [[0.0, 0.0], [0.3, 0.0], [0.0, 0.0], [0.9, 0.0]],
            0.5,
            [(0, 1), (1, 0)],
        ),
        # Robots of different worlds share positions but must never connect.
        (
            [[0.0, 0.0], [1.0, 0.0], [0.0, 0.0], [1.0, 0.0]],
            0.5,
            [],
        ),
    ],
)
def test_agent_graph(positions, comm_range, expected_edges):
    graph = AgentGraph(torch.tensor(positions), 2, comm_range)
    edges = sorted(map(tuple, graph.edge_index.t().tolist()))
    assert edges == expected_edges
    assert graph.edge_attr.shape == (len(expected_edges), 1)


# Lead shapes: one step (B,), a time axis (B, T), and imagination (B, T, H),
# for both the dense path and the original sparse torch_geometric path.
@pytest.mark.parametrize(
    ("lead", "dense"),
    [((8,), True), ((8, 3), True), ((8, 3, 2), True), ((8, 3), False)],
)
def test_graph_layer(lead, dense):
    torch.manual_seed(0)
    graph = AgentGraph(torch.rand(*lead, 2), 4, 0.5)
    layer = GraphLayer(6, 8, 4, dense)
    x = torch.randn(*lead, 6)
    y = layer(x, graph)
    assert y.shape == (*lead, 14)
    # Own features pass through unchanged in front of the message.
    assert torch.equal(y[..., :6], x)
    # Changing world 1 (rows 4-7) must leave world 0 (rows 0-3) untouched.
    changed = x.clone()
    changed[4:] += 10.0
    assert torch.equal(layer(changed, graph)[:4], y[:4])


# The dense GATv2 must match torch_geometric's GATv2Conv on the same weights:
# robots far apart (only self-loops), all in range, and a mix, over several
# lead shapes and robot counts.
@pytest.mark.parametrize(
    ("lead", "n_agents", "comm_range", "spread"),
    [
        ((8,), 4, 0.5, 1.0),
        ((8, 3), 4, 0.5, 1.0),
        ((16, 5), 8, 0.5, 1.0),
        ((8, 3), 4, 0.01, 1.0),
        ((8, 3), 4, 5.0, 1.0),
    ],
)
def test_dense_conv(lead, n_agents, comm_range, spread):
    torch.manual_seed(0)
    graph = AgentGraph(torch.rand(*lead, 2) * spread, n_agents, comm_range)
    layer = GraphLayer(6, 8, 4, dense=True).double()
    nodes = graph.group(torch.randn(*lead, 6, dtype=torch.float64))
    nodes.requires_grad_(True)
    dense = layer._dense_conv(nodes, graph)
    sparse = layer.conv(
        nodes.reshape(-1, 6), graph.edge_index, graph.edge_attr.double()
    ).view_as(dense)
    torch.testing.assert_close(dense, sparse)
    # Gradients reach the inputs and every weight identically on both paths.
    weights = [nodes, *layer.parameters()]
    grad_dense = torch.autograd.grad(dense.square().sum(), weights)
    grad_sparse = torch.autograd.grad(sparse.square().sum(), weights)
    for a, b in zip(grad_dense, grad_sparse):
        torch.testing.assert_close(a, b)


# Dropping edges by assigning a smaller edge list (how link models such as
# keep_edges in co_imaginer/evaluation.py work) must change the dense layer
# exactly as it changes the sparse one, and leave the original graph intact.
@pytest.mark.parametrize("keep_every", [1, 2, 3, 1000])
def test_edge_index(keep_every):
    torch.manual_seed(0)
    graph = AgentGraph(torch.rand(8, 3, 2) * 0.6, 4, 0.5)
    original_mask = graph.mask.clone()
    kept = copy.copy(graph)
    keep = torch.arange(graph.edge_index.shape[1]) % keep_every == 0
    kept.edge_index = graph.edge_index[:, keep]
    kept.edge_attr = graph.edge_attr[keep]
    assert int(kept.adjacency.sum()) == int(keep.sum())
    assert torch.equal(graph.mask, original_mask)
    layer = GraphLayer(6, 8, 4).double()
    nodes = kept.group(torch.randn(8, 3, 6, dtype=torch.float64))
    dense = layer._dense_conv(nodes, kept)
    sparse = layer.conv(
        nodes.reshape(-1, 6), kept.edge_index, kept.edge_attr.double()
    ).view_as(dense)
    torch.testing.assert_close(dense, sparse)
