import pytest
import torch

from co_imaginer.vmas_agents import VmasAgents

N_WORLDS = 3
N_AGENTS = 4


@pytest.mark.parametrize(
    (
        "max_steps",
        "action_value",
        "reset_worlds",
        "expected_first_worlds",
        "expected_last_worlds",
    ),
    [
        (100, 0.0, [1], [1], []),
        (1, 0.0, [], [], [0, 1, 2]),
        (100, 1.3, [], [], []),
    ],
)
def test_step(
    max_steps, action_value, reset_worlds, expected_first_worlds, expected_last_worlds
):
    env = VmasAgents("discovery", N_WORLDS, N_AGENTS, max_steps, 0, "cpu")
    action = torch.full((env.env_num, 2), action_value)
    env.step(action, torch.ones(env.env_num, dtype=torch.bool))
    done = torch.zeros(N_WORLDS, N_AGENTS, dtype=torch.bool)
    done[reset_worlds] = True

    transition, is_last = env.step(action, done.flatten())

    def rows(worlds):
        expected = torch.zeros(N_WORLDS, N_AGENTS, dtype=torch.bool)
        expected[worlds] = True
        return expected.flatten()

    assert transition["obs"].shape == (env.env_num, 19)
    assert torch.equal(transition["is_first"][:, 0], rows(expected_first_worlds))
    assert torch.equal(is_last, rows(expected_last_worlds))
    assert torch.all(transition["reward"][rows(expected_first_worlds)] == 0)
    assert not transition["is_terminal"].any()
    assert transition["log_collisions"].shape == (env.env_num, 1)
    assert torch.all(transition["log_collisions"][rows(expected_first_worlds)] == 0)
