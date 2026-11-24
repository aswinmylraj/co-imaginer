import pytest
import torch

from co_imaginer.churn import (
    churn_counts,
    churn_metrics,
    edge_matrix,
    rollout_positions,
)

# Tests for co_imaginer/churn.py. Each test lists its cases in parametrize.


# Three robots on a line: which pairs are within range.
@pytest.mark.parametrize(
    ("positions", "comm_range", "expected_edges"),
    [
        ([[0.0, 0.0], [0.3, 0.0], [1.0, 0.0]], 0.35, [(0, 1)]),
        ([[0.0, 0.0], [0.3, 0.0], [1.0, 0.0]], 0.25, []),
        ([[0.0, 0.0], [0.3, 0.0], [0.6, 0.0]], 0.35, [(0, 1), (1, 2)]),
    ],
)
def test_edge_matrix(positions, comm_range, expected_edges):
    edges = edge_matrix(torch.tensor(positions), comm_range)
    assert sorted(map(tuple, edges.nonzero().tolist())) == expected_edges


# Two robots over time: one boolean per step says whether they are linked.
def _edges(*steps):
    return torch.tensor([[[False, a], [False, False]] for a in steps])


# Cases: link always on, link drops and stays off, link drops and comes back
# (no change end to end but one flip inside), and no link at all.
@pytest.mark.parametrize(
    ("edges", "horizon", "expected"),
    [
        (
            _edges(True, True, True),
            2,
            {"changed": 0, "union": 1, "kept": 1, "flipped": 0},
        ),
        (
            _edges(True, False, False),
            2,
            {"changed": 1, "union": 1, "kept": 0, "flipped": 1},
        ),
        (
            _edges(True, False, True),
            2,
            {"changed": 0, "union": 1, "kept": 1, "flipped": 1},
        ),
        (
            _edges(False, False, False),
            1,
            {"changed": 0, "union": 0, "kept": 0, "flipped": 0},
        ),
    ],
)
def test_churn_counts(edges, horizon, expected):
    counts = churn_counts(edges, horizon)
    assert {key: counts[key] for key in expected} == expected


# Ratios from hand-computed counts, and the empty case without division by zero.
@pytest.mark.parametrize(
    ("counts", "expected"),
    [
        (
            {
                "changed": 1,
                "union": 4,
                "start": 3,
                "end": 2,
                "kept": 2,
                "flipped": 2,
                "ever_present": 4,
            },
            {"churn": 0.25, "frozen_f1": 0.8, "flipped_within": 0.5},
        ),
        (
            {
                "changed": 0,
                "union": 0,
                "start": 0,
                "end": 0,
                "kept": 0,
                "flipped": 0,
                "ever_present": 0,
            },
            {"churn": 0.0, "frozen_f1": 1.0, "flipped_within": 0.0},
        ),
    ],
)
def test_churn_metrics(counts, expected):
    assert churn_metrics(counts) == pytest.approx(expected)


# A short VMAS rollout returns (steps + 1, worlds, robots, xy).
@pytest.mark.parametrize(("n_agents", "policy"), [(4, "random"), (8, "scripted")])
def test_rollout_positions(n_agents, policy):
    positions = rollout_positions(n_agents, policy, 1.0, steps=5, num_envs=2, seed=0)
    assert positions.shape == (6, 2, n_agents, 2)
