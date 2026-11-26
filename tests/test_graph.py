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


# Lead shapes: one step (B,), a time axis (B, T), and imagination (B, T, H).
@pytest.mark.parametrize("lead", [(8,), (8, 3), (8, 3, 2)])
def test_graph_layer(lead):
    torch.manual_seed(0)
    graph = AgentGraph(torch.rand(*lead, 2), 4, 0.5)
    layer = GraphLayer(6, 8, 4)
    x = torch.randn(*lead, 6)
    y = layer(x, graph)
    assert y.shape == (*lead, 14)
    # Own features pass through unchanged in front of the message.
    assert torch.equal(y[..., :6], x)
    # Changing world 1 (rows 4-7) must leave world 0 (rows 0-3) untouched.
    changed = x.clone()
    changed[4:] += 10.0
    assert torch.equal(layer(changed, graph)[:4], y[:4])
