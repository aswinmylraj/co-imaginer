import math

import pytest
import torch
from omegaconf import OmegaConf
from rssm import RSSM

from co_imaginer.evaluation import (
    AgentPolicy,
    RandomPolicy,
    Trajectory,
    WorldModelPredictor,
    adjacency,
    episode_metrics,
    evaluate_graph_prediction,
    f1_score,
    frozen_positions,
    graph_f1,
    keep_edges,
    perfect_links,
    reward_error,
    rollout,
    true_positions,
    valid_starts,
    with_links,
)
from co_imaginer.graph import AgentGraph
from co_imaginer.vmas_agents import VmasAgents

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


def make_trajectory(positions, is_first=None, reward=None, collisions=None):
    positions = torch.tensor(positions, dtype=torch.float32)
    worlds, steps, n_agents = positions.shape[:3]
    obs = torch.cat([positions, torch.zeros(worlds, steps, n_agents, 3)], -1)
    if is_first is None:
        is_first = torch.zeros(worlds, steps, dtype=torch.bool)
        is_first[:, 0] = True
    is_first = torch.as_tensor(is_first, dtype=torch.bool)
    is_last = torch.zeros_like(is_first)
    is_last[:, :-1] = is_first[:, 1:]
    is_last[:, -1] = True
    shape = (worlds, steps, n_agents)
    return Trajectory(
        obs=obs,
        action=torch.zeros(*shape, 2),
        reward=torch.zeros(shape) if reward is None else torch.tensor(reward),
        collisions=torch.zeros(shape)
        if collisions is None
        else torch.tensor(collisions),
        is_first=is_first,
        is_last=is_last,
    )


def line_positions(steps, speed):
    return [[[0.0, 0.0], [0.3 + speed * t, 0.0]] for t in range(steps)]


@pytest.mark.parametrize(
    ("positions", "comm_range", "expected"),
    [
        ([[0.0, 0.0], [0.3, 0.0]], 0.5, [[False, True], [True, False]]),
        ([[0.0, 0.0], [0.5, 0.0]], 0.5, [[False, True], [True, False]]),
        ([[0.0, 0.0], [0.9, 0.0]], 0.5, [[False, False], [False, False]]),
        (
            [[0.0, 0.0], [0.4, 0.0], [0.8, 0.0]],
            0.5,
            [[False, True, False], [True, False, True], [False, True, False]],
        ),
    ],
)
def test_adjacency(positions, comm_range, expected):
    result = adjacency(torch.tensor(positions), comm_range)
    assert torch.equal(result, torch.tensor(expected))


@pytest.mark.parametrize(
    ("predicted", "true", "expected"),
    [
        ([True, False, True], [True, False, True], 1.0),
        ([False, False], [False, False], 1.0),
        ([True, True], [False, False], 0.0),
        ([True, True, False, False], [True, False, True, False], 0.5),
        ([True, False, False], [True, True, True], 0.5),
    ],
)
def test_f1_score(predicted, true, expected):
    assert f1_score(torch.tensor(predicted), torch.tensor(true)) == pytest.approx(
        expected
    )


@pytest.mark.parametrize(
    ("is_first", "horizon", "stride", "expected"),
    [
        ([[True, False, False, False]], 2, 1, [(0, 0), (0, 1)]),
        ([[True, False, True, False, False]], 2, 1, [(0, 2)]),
        ([[True, False, False, False, False]], 2, 2, [(0, 0), (0, 2)]),
        ([[True, False, False], [True, True, False]], 1, 1, [(0, 0), (0, 1), (1, 1)]),
        ([[True, False]], 2, 1, []),
    ],
)
def test_valid_starts(is_first, horizon, stride, expected):
    worlds, starts = valid_starts(torch.tensor(is_first), horizon, stride)
    assert sorted(zip(worlds.tolist(), starts.tolist())) == expected


@pytest.mark.parametrize(("start", "horizon"), [(0, 2), (1, 3)])
def test_true_positions(start, horizon):
    trajectory = make_trajectory([line_positions(6, 0.1)])
    result = true_positions(
        trajectory, torch.tensor([0]), torch.tensor([start]), horizon
    )
    assert result.shape == (1, horizon + 1, 2, 2)
    expected = trajectory.positions[0, start : start + horizon + 1]
    assert torch.equal(result[0], expected)


@pytest.mark.parametrize(("start", "horizon"), [(0, 2), (2, 3)])
def test_frozen_positions(start, horizon):
    trajectory = make_trajectory([line_positions(6, 0.1)])
    result = frozen_positions(
        trajectory, torch.tensor([0]), torch.tensor([start]), horizon
    )
    assert result.shape == (1, horizon + 1, 2, 2)
    for h in range(horizon + 1):
        assert torch.equal(result[0, h], trajectory.positions[0, start])


@pytest.mark.parametrize(
    ("speed", "horizons", "expected"),
    [
        (0.0, (1, 3), {1: 1.0, 3: 1.0}),
        (0.1, (1, 3), {1: 1.0, 3: 0.0}),
        (0.1, (1, 9), {1: 1.0}),
    ],
)
def test_graph_f1(speed, horizons, expected):
    trajectory = make_trajectory([line_positions(5, speed)])
    worlds, starts = torch.tensor([0]), torch.tensor([0])
    true = true_positions(trajectory, worlds, starts, 4)
    predicted = frozen_positions(trajectory, worlds, starts, 4)
    assert graph_f1(predicted, true, 0.5, horizons) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("keep", "expected"),
    [
        ([True, True, True, True], [(0, 1), (1, 0), (2, 3), (3, 2)]),
        ([False, True, False, True], None),
        ([False, False, False, False], []),
    ],
)
def test_keep_edges(keep, expected):
    positions = torch.tensor([[0.0, 0.0], [0.3, 0.0], [0.0, 0.0], [0.3, 0.0]])
    graph = AgentGraph(positions, 2, 0.5)
    keep = torch.tensor(keep)
    kept = keep_edges(graph, keep)
    if expected is None:
        expected = [tuple(e) for e in graph.edge_index[:, keep].t().tolist()]
    assert sorted(map(tuple, kept.edge_index.t().tolist())) == sorted(expected)
    assert kept.edge_attr.shape == (len(expected), 1)
    assert graph.edge_index.shape[1] == 4


@pytest.mark.parametrize("comm_range", [0.5, 2.0])
def test_perfect_links(comm_range):
    graph = AgentGraph(torch.rand(4, 2), 2, comm_range)
    assert perfect_links(graph) is graph


@pytest.mark.parametrize(("enabled", "expected_edges"), [(True, 0), (False, None)])
def test_with_links(enabled, expected_edges):
    comm = OmegaConf.create(
        {"enabled": enabled, "n_agents": 2, "comm_range": 0.5, "heads": 2, "units": 4}
    )
    model = RSSM(RSSM_CONFIG, 6, 2, comm=comm)

    def drop_all(graph):
        return keep_edges(
            graph, torch.zeros(graph.edge_index.shape[1], dtype=torch.bool)
        )

    with_links(model, drop_all)
    graph = model.graph(torch.zeros(4, 2))
    if expected_edges is None:
        assert graph is None
    else:
        assert graph.edge_index.shape[1] == expected_edges


@pytest.mark.parametrize(("batch_size", "action_size"), [(4, 2), (8, 3)])
def test_random_policy(batch_size, action_size):
    policy = RandomPolicy(action_size)
    assert torch.equal(policy.reset(batch_size), torch.zeros(batch_size, action_size))
    action = policy(torch.zeros(batch_size))
    assert action.shape == (batch_size, action_size)
    assert action.abs().max() <= 1.0


class FakeAgent:
    def get_initial_state(self, batch_size):
        return {"prev_action": torch.ones(batch_size, 2)}

    def act(self, transition, state, eval=False):
        action = state["prev_action"] + 1
        return action, {"prev_action": action}


@pytest.mark.parametrize("batch_size", [2, 4])
def test_agent_policy(batch_size):
    policy = AgentPolicy(FakeAgent())
    assert torch.equal(policy.reset(batch_size), torch.ones(batch_size, 2))
    assert torch.equal(policy(None), torch.full((batch_size, 2), 2.0))
    assert torch.equal(policy(None), torch.full((batch_size, 2), 3.0))


@pytest.mark.parametrize(("worlds", "n_agents", "steps"), [(2, 2, 12), (1, 4, 7)])
def test_rollout(worlds, n_agents, steps):
    envs = VmasAgents("discovery", worlds, n_agents, 5, 0, "cpu")
    policy = RandomPolicy(envs.action_space.shape[0])
    trajectory = rollout(envs, policy, steps, n_agents)
    assert trajectory.obs.shape[:3] == (worlds, steps, n_agents)
    assert trajectory.reward.shape == (worlds, steps, n_agents)
    assert trajectory.is_first.shape == (worlds, steps)
    assert trajectory.is_first[:, 0].all()
    assert torch.equal(trajectory.positions, trajectory.obs[..., :2])
    assert trajectory.is_last.any()


@pytest.mark.parametrize(
    ("is_first", "reward", "expected"),
    [
        (
            [[True, False, True, False]],
            [[[1.0, 1.0], [2.0, 2.0], [0.0, 0.0], [3.0, 5.0]]],
            (2, 3.5, 0.5),
        ),
        ([[True, False, False]], [[[1.0], [1.0], [1.0]]], (1, 3.0, 0.0)),
    ],
)
def test_episode_metrics(is_first, reward, expected):
    n_agents = len(reward[0][0])
    positions = [[[[0.0, 0.0]] * n_agents] * len(is_first[0])]
    trajectory = make_trajectory(positions, is_first=is_first, reward=reward)
    metrics = episode_metrics(trajectory)
    assert metrics.episodes == expected[0]
    assert metrics.mean_return == pytest.approx(expected[1])
    assert metrics.std_return == pytest.approx(expected[2])
    assert metrics.mean_collisions == 0.0


@pytest.mark.parametrize(("offset", "expected"), [(0.0, 0.0), (0.5, 0.5)])
def test_reward_error(offset, expected):
    reward = [[[float(t)] * 2 for t in range(6)]]
    trajectory = make_trajectory([line_positions(6, 0.0)], reward=reward)
    worlds, starts = torch.tensor([0]), torch.tensor([1])
    predicted = trajectory.reward[0, 1:5][None] + offset
    assert reward_error(predicted, trajectory, worlds, starts, (1, 3)) == pytest.approx(
        {1: expected, 3: expected}
    )


def make_predictor(enabled, graph_mode):
    torch.manual_seed(0)
    comm = OmegaConf.create(
        {"enabled": enabled, "n_agents": 2, "comm_range": 0.5, "heads": 2, "units": 4}
    )
    model = RSSM(RSSM_CONFIG, 6, 2, comm=comm)
    encoder = torch.nn.Linear(5, 6)
    position_head = torch.nn.Linear(16, 2)
    reward_head = torch.nn.Linear(32, 1)
    return WorldModelPredictor(
        model,
        lambda data: encoder(data["obs"]),
        lambda stoch, deter: position_head(deter),
        lambda feat, graph: reward_head(feat)[..., 0],
        graph_mode,
    )


@pytest.mark.parametrize(
    ("enabled", "graph_mode"),
    [(False, "frozen"), (True, "frozen"), (True, "rebuilt")],
)
def test_world_model_predictor(enabled, graph_mode):
    predictor = make_predictor(enabled, graph_mode)
    trajectory = make_trajectory([line_positions(8, 0.05), line_positions(8, 0.0)])
    worlds, starts = torch.tensor([0, 1, 1]), torch.tensor([0, 2, 3])
    predicted = predictor(trajectory, worlds, starts, 4)
    assert predicted.shape == (3, 5, 2, 2)
    assert predictor.predicted_reward.shape == (3, 5, 2)
    assert torch.isfinite(predicted).all()
    assert predictor.imagined_steps_per_second > 0
    with pytest.raises(ValueError):
        WorldModelPredictor(None, None, None, None, "dynamic")


@pytest.mark.parametrize(
    ("speed", "expected_late_f1"),
    [(0.0, 1.0), (0.05, 0.0)],
)
def test_evaluate_graph_prediction(speed, expected_late_f1):
    trajectory = make_trajectory([line_positions(8, speed)])
    predictors = {
        "frozen": frozen_positions,
        "world_model": make_predictor(True, "frozen"),
    }
    results = evaluate_graph_prediction(trajectory, predictors, 0.5, (1, 5), stride=1)
    assert results["windows"] == 3
    assert results["frozen"]["graph_f1"][1] == 1.0
    assert results["frozen"]["graph_f1"][5] == expected_late_f1
    world_model = results["world_model"]
    assert set(world_model["graph_f1"]) == {1, 5}
    assert set(world_model["reward_error"]) == {1, 5}
    assert all(math.isfinite(v) for v in world_model["position_error"].values())
