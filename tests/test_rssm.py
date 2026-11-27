import pytest
import torch
from omegaconf import OmegaConf
from rssm import RSSM

from co_imaginer.graph import AgentGraph

# Tests for the CoDreamer switch in third_party/r2dreamer/rssm.py.

RSSM_CONFIG = OmegaConf.create(
    {
        "stoch": 4,
        "deter": 16,
        "hidden": 8,
        "discrete": 4,
        "act": "SiLU",
        "unimix_ratio": 0.01,
        "initial": "learned",
        "device": "cpu",
        "obs_layers": 1,
        "img_layers": 1,
        "dyn_layers": 1,
        "blocks": 2,
    }
)


def _comm(enabled):
    return OmegaConf.create(
        {
            "enabled": enabled,
            "n_agents": 2,
            "comm_range": 0.5,
            "heads": 2,
            "units": 4,
        }
    )


@pytest.mark.parametrize("enabled", [False, True])
def test_graph(enabled):
    torch.manual_seed(0)
    model = RSSM(RSSM_CONFIG, 6, 2, comm=_comm(enabled))
    graph = model.graph(torch.zeros(4, 2))
    if not enabled:
        # Switch off: no graph, and weights identical to the original IDreamer RSSM.
        assert graph is None
        torch.manual_seed(0)
        original = RSSM(RSSM_CONFIG, 6, 2)
        assert model.state_dict().keys() == original.state_dict().keys()
        for key, value in original.state_dict().items():
            assert torch.equal(model.state_dict()[key], value)
    else:
        assert isinstance(graph, AgentGraph)
        # A posterior step with messages runs and keeps the usual output shapes.
        stoch, deter = model.initial(4)
        stoch, deter, _ = model.obs_step(
            stoch,
            deter,
            torch.zeros(4, 2),
            torch.zeros(4, 6),
            torch.zeros(4, dtype=torch.bool),
            graph,
        )
        assert stoch.shape == (4, 4, 4)
        assert deter.shape == (4, 16)


@pytest.mark.parametrize(
    ("positions", "expected_edges"),
    [
        ([[0.0, 0.0], [0.3, 0.0], [0.0, 0.0], [0.3, 0.0]], 4),
        ([[0.0, 0.0], [0.3, 0.0], [0.0, 0.0], [0.9, 0.0]], 2),
        ([[0.0, 0.0], [0.9, 0.0], [0.0, 0.0], [0.9, 0.0]], 0),
    ],
)
def test_img_step_rebuilt(positions, expected_edges):
    torch.manual_seed(0)
    model = RSSM(RSSM_CONFIG, 6, 2, comm=_comm(True))
    stoch, deter = model.initial(4)
    located = []

    def locate(new_deter):
        located.append(new_deter)
        return torch.tensor(positions)

    stoch, deter, graph = model.img_step_rebuilt(
        stoch, deter, torch.ones(4, 2), locate, (4,)
    )
    assert torch.equal(located[0], deter)
    assert graph.edge_index.shape[1] == expected_edges
    assert stoch.shape == (4, 4, 4)
