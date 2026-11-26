import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from churn_check import comm_graph, window_churn


@pytest.mark.parametrize(
    ("pos", "comm_range", "expected"),
    [
        ([[0.0, 0.0], [0.3, 0.0], [1.0, 0.0]], 0.35, [True, False, False]),
        ([[0.0, 0.0], [0.3, 0.0], [1.0, 0.0]], 0.75, [True, False, True]),
        ([[0.0, 0.0], [0.5, 0.0]], 0.5, [True]),
    ],
)
def test_comm_graph(pos: list, comm_range: float, expected: list) -> None:
    edges = comm_graph(torch.tensor(pos), comm_range)
    assert edges.tolist() == expected


@pytest.mark.parametrize(
    ("edges", "horizon", "churn", "f1"),
    [
        ([[True, False], [True, False]], 1, 0.0, 1.0),
        ([[True, False], [False, True]], 1, 1.0, 0.0),
        ([[True, True], [True, False]], 1, 0.5, 2 / 3),
    ],
)
def test_window_churn(edges: list, horizon: int, churn: float, f1: float) -> None:
    result = window_churn(torch.tensor(edges), horizon)
    assert result["edge_churn"] == pytest.approx(churn)
    assert result["frozen_graph_f1"] == pytest.approx(f1)
