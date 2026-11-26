import math

import pytest

from baselines.ippo_discovery import TASK_KWARGS, random_policy_return, task_kwargs


@pytest.mark.parametrize("n_agents", [4, 8])
def test_task_kwargs(n_agents: int) -> None:
    kwargs = task_kwargs(n_agents)
    assert kwargs["n_agents"] == n_agents
    assert {k: v for k, v in kwargs.items() if k != "n_agents"} == {
        k: v for k, v in TASK_KWARGS.items() if k != "n_agents"
    }


@pytest.mark.parametrize("n_agents", [4, 8])
def test_random_policy_return(n_agents: int) -> None:
    stats = random_policy_return(
        {**task_kwargs(n_agents), "max_steps": 10}, episodes=4, seed=0, device="cpu"
    )
    assert stats["episodes"] == 4
    assert math.isfinite(stats["mean"])
    assert stats["mean"] >= 0.0
