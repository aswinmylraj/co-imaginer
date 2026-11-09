import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "third_party" / "r2dreamer")
)

from envs.vmas_agents import VmasAgentEnv

KWARGS = {
    "n_agents": 4,
    "n_targets": 7,
    "lidar_range": 0.35,
    "covering_range": 0.25,
    "agents_per_target": 2,
    "targets_respawn": True,
    "shared_reward": True,
}


def make(num_envs: int, n_agents: int, max_steps: int) -> VmasAgentEnv:
    return VmasAgentEnv(
        "discovery", num_envs, max_steps, "cpu", 0, {**KWARGS, "n_agents": n_agents}
    )


@pytest.mark.parametrize("num_envs,n_agents", [(3, 4), (2, 8)])
def test_env_num(num_envs, n_agents):
    assert make(num_envs, n_agents, 10).env_num == num_envs * n_agents


@pytest.mark.parametrize("num_envs,n_agents,max_steps", [(3, 4, 5), (2, 8, 3)])
def test_step(num_envs, n_agents, max_steps):
    env = make(num_envs, n_agents, max_steps)
    n = env.env_num
    done = torch.ones(n, dtype=torch.bool)
    action = torch.zeros(n, 2)
    trans, done = env.step(action, done)
    assert trans["vector"].shape == (n, env.observation_space["vector"].shape[0])
    assert trans["is_first"].all() and not done.any()
    assert (trans["reward"] == 0).all()
    for t in range(max_steps):
        trans, done = env.step(torch.rand(n, 2) * 2 - 1, done)
        assert not trans["is_first"].any()
        assert done.all() == (t == max_steps - 1)
    assert trans["is_last"].all() and not trans["is_terminal"].any()
    trans, done = env.step(action, done)
    assert trans["is_first"].all() and (trans["reward"] == 0).all()


@pytest.mark.parametrize("reset_env", [0, 2])
def test_step_partial_reset(reset_env):
    env = make(3, 4, 50)
    done = torch.ones(env.env_num, dtype=torch.bool)
    trans, done = env.step(torch.zeros(env.env_num, 2), done)
    trans, done = env.step(torch.ones(env.env_num, 2), done)
    before = trans["vector"].view(3, 4, -1).clone()
    done = torch.zeros(3, 4, dtype=torch.bool)
    done[reset_env] = True
    trans, _ = env.step(torch.ones(env.env_num, 2), done.flatten())
    is_first = trans["is_first"].view(3, 4)
    assert is_first[reset_env].all() and is_first.sum() == 4
    assert (trans["reward"].view(3, 4)[reset_env] == 0).all()
    assert not torch.equal(trans["vector"].view(3, 4, -1)[reset_env], before[reset_env])
