import copy
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

import torch
from tensordict import TensorDict

from co_imaginer.graph import AgentGraph

HORIZONS = (15, 30, 50)


@dataclass
class Trajectory:
    obs: torch.Tensor
    action: torch.Tensor
    reward: torch.Tensor
    collisions: torch.Tensor
    is_first: torch.Tensor
    is_last: torch.Tensor

    @property
    def positions(self) -> torch.Tensor:
        return self.obs[..., :2]

    @property
    def n_agents(self) -> int:
        return self.obs.shape[2]

    def rows(self, key: str) -> torch.Tensor:
        value = getattr(self, key)
        worlds, steps, n_agents = value.shape[:3]
        return value.transpose(1, 2).reshape(worlds * n_agents, steps, *value.shape[3:])


@dataclass
class EpisodeMetrics:
    episodes: int
    mean_return: float
    std_return: float
    mean_collisions: float


class Policy(Protocol):
    def reset(self, batch_size: int) -> torch.Tensor: ...

    def __call__(self, transition: TensorDict) -> torch.Tensor: ...


class LinkModel(Protocol):
    def __call__(self, graph: AgentGraph) -> AgentGraph: ...


class PositionPredictor(Protocol):
    def __call__(
        self,
        trajectory: Trajectory,
        worlds: torch.Tensor,
        starts: torch.Tensor,
        horizon: int,
    ) -> torch.Tensor: ...


def adjacency(positions: torch.Tensor, comm_range: float) -> torch.Tensor:
    positions = positions.float()
    distance = torch.cdist(positions, positions)
    n_agents = positions.shape[-2]
    not_self = ~torch.eye(n_agents, dtype=torch.bool, device=positions.device)
    # Sec. 2.2.1, CoDreamer
    return (distance <= comm_range) & not_self


def f1_score(predicted: torch.Tensor, true: torch.Tensor) -> float:
    true_positive = (predicted & true).sum().item()
    false_positive = (predicted & ~true).sum().item()
    false_negative = (~predicted & true).sum().item()
    denominator = 2 * true_positive + false_positive + false_negative
    if denominator == 0:
        return 1.0
    return 2 * true_positive / denominator


def valid_starts(
    is_first: torch.Tensor, horizon: int, stride: int = 1
) -> tuple[torch.Tensor, torch.Tensor]:
    steps = is_first.shape[1]
    if steps <= horizon:
        empty = torch.zeros(0, dtype=torch.long, device=is_first.device)
        return empty, empty
    resets = is_first.long().cumsum(1)
    starts = torch.arange(0, steps - horizon, stride, device=is_first.device)
    same_episode = resets[:, starts + horizon] == resets[:, starts]
    world_index, start_index = same_episode.nonzero(as_tuple=True)
    return world_index, starts[start_index]


def true_positions(
    trajectory: Trajectory, worlds: torch.Tensor, starts: torch.Tensor, horizon: int
) -> torch.Tensor:
    steps = starts[:, None] + torch.arange(horizon + 1, device=starts.device)
    return trajectory.positions[worlds[:, None], steps]


def frozen_positions(
    trajectory: Trajectory, worlds: torch.Tensor, starts: torch.Tensor, horizon: int
) -> torch.Tensor:
    start = trajectory.positions[worlds, starts]
    return start[:, None].expand(-1, horizon + 1, -1, -1)


def graph_f1(
    predicted: torch.Tensor,
    true: torch.Tensor,
    comm_range: float,
    horizons: Sequence[int] = HORIZONS,
) -> dict[int, float]:
    predicted_edges = adjacency(predicted, comm_range)
    true_edges = adjacency(true, comm_range)
    return {
        h: f1_score(predicted_edges[:, h], true_edges[:, h])
        for h in horizons
        if h < predicted.shape[1]
    }


def keep_edges(graph: AgentGraph, keep: torch.Tensor) -> AgentGraph:
    kept = copy.copy(graph)
    kept.edge_index = graph.edge_index[:, keep]
    kept.edge_attr = graph.edge_attr[keep]
    return kept


def perfect_links(graph: AgentGraph) -> AgentGraph:
    return graph


LINK_MODELS: dict[str, LinkModel] = {"perfect": perfect_links}


def with_links(rssm: torch.nn.Module, links: LinkModel) -> torch.nn.Module:
    build = rssm.graph

    def linked_graph(positions: torch.Tensor | None) -> AgentGraph | None:
        graph = build(positions)
        return None if graph is None else links(graph)

    rssm.graph = linked_graph
    return rssm


class AgentPolicy:
    def __init__(self, agent: torch.nn.Module) -> None:
        self.agent = agent
        self.state = None

    def reset(self, batch_size: int) -> torch.Tensor:
        self.state = self.agent.get_initial_state(batch_size)
        return self.state["prev_action"].clone()

    def __call__(self, transition: TensorDict) -> torch.Tensor:
        action, self.state = self.agent.act(transition, self.state, eval=True)
        return action


class RandomPolicy:
    def __init__(self, action_size: int, seed: int = 0) -> None:
        self.action_size = action_size
        self.generator = torch.Generator().manual_seed(seed)
        self.batch_size = 0

    def reset(self, batch_size: int) -> torch.Tensor:
        self.batch_size = batch_size
        return torch.zeros(batch_size, self.action_size)

    def __call__(self, transition: TensorDict) -> torch.Tensor:
        action = torch.rand(self.batch_size, self.action_size, generator=self.generator)
        return (2 * action - 1).to(transition.device)


@torch.no_grad()
def rollout(envs, policy: Policy, steps: int, n_agents: int) -> Trajectory:
    batch_size = envs.env_num
    action = policy.reset(batch_size)
    done = torch.ones(batch_size, dtype=torch.bool)
    transitions = []
    for _ in range(steps):
        transition, done = envs.step(action.detach(), done)
        transition["action"] = action.to(transition.device)
        transitions.append(transition.clone())
        action = policy(transition)
    stacked = torch.stack(transitions, 1)
    worlds = batch_size // n_agents

    def per_world(key: str) -> torch.Tensor:
        value = stacked[key].float().cpu()
        value = value.reshape(worlds, n_agents, steps, *value.shape[2:])
        return value.transpose(1, 2)

    return Trajectory(
        obs=per_world("obs"),
        action=per_world("action"),
        reward=per_world("reward")[..., 0],
        collisions=per_world("log_collisions")[..., 0],
        is_first=per_world("is_first")[:, :, 0, 0].bool(),
        is_last=per_world("is_last")[:, :, 0, 0].bool(),
    )


def episode_metrics(trajectory: Trajectory) -> EpisodeMetrics:
    returns, collisions = [], []
    worlds, steps = trajectory.is_first.shape
    for world in range(worlds):
        start = None
        for step in range(steps):
            if trajectory.is_first[world, step]:
                start = step
            if trajectory.is_last[world, step] and start is not None:
                window = slice(start, step + 1)
                returns.append(trajectory.reward[world, window].sum(0).mean().item())
                collisions.append(
                    trajectory.collisions[world, window].sum(0).mean().item()
                )
                start = None
    if not returns:
        return EpisodeMetrics(0, float("nan"), float("nan"), float("nan"))
    returns_tensor = torch.tensor(returns)
    return EpisodeMetrics(
        episodes=len(returns),
        mean_return=returns_tensor.mean().item(),
        std_return=returns_tensor.std(correction=0).item(),
        mean_collisions=torch.tensor(collisions).mean().item(),
    )


class WorldModelPredictor:
    def __init__(
        self,
        rssm: torch.nn.Module,
        encoder: Callable[[dict], torch.Tensor],
        decode_positions: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        reward: Callable[[torch.Tensor, AgentGraph | None], torch.Tensor],
        graph_mode: str = "frozen",
        device: str = "cpu",
        locate: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> None:
        if graph_mode not in ("frozen", "rebuilt"):
            raise ValueError("graph_mode must be frozen or rebuilt")
        self.rssm = rssm
        self.encoder = encoder
        self.decode_positions = decode_positions
        self.reward = reward
        self.graph_mode = graph_mode
        self.device = torch.device(device)
        self.locate = locate
        self.predicted_reward = None
        self.imagined_steps_per_second = float("nan")

    @classmethod
    def from_agent(
        cls,
        agent: torch.nn.Module,
        graph_mode: str = "frozen",
        decode_positions: Callable | None = None,
    ) -> "WorldModelPredictor":
        locate = None
        if hasattr(agent, "position"):

            def locate(deter):
                return agent.position(deter).mode()

            def decode_positions(stoch, deter):
                return locate(deter)

        if decode_positions is None:

            def decode_positions(stoch, deter):
                return agent.decoder(stoch, deter)["obs"].mode()[..., :2]

        return cls(
            agent.rssm,
            agent.encoder,
            decode_positions,
            lambda feat, graph: agent.reward(feat, graph).mode()[..., 0],
            graph_mode,
            str(agent.device),
            locate,
        )

    @torch.no_grad()
    def __call__(
        self,
        trajectory: Trajectory,
        worlds: torch.Tensor,
        starts: torch.Tensor,
        horizon: int,
    ) -> torch.Tensor:
        n_agents = trajectory.n_agents
        obs = trajectory.rows("obs").to(self.device)
        action = trajectory.rows("action").to(self.device)
        is_first = trajectory.is_first.repeat_interleave(n_agents, 0).to(self.device)
        embed = self.encoder({"obs": obs})
        initial = self.rssm.initial(obs.shape[0])
        post_stoch, post_deter, _ = self.rssm.observe(
            embed, action, initial, is_first, obs[..., :2]
        )
        rows = (worlds[:, None] * n_agents + torch.arange(n_agents)).reshape(-1)
        rows = rows.to(self.device)
        times = starts.repeat_interleave(n_agents).to(self.device)
        stoch, deter = post_stoch[rows, times], post_deter[rows, times]
        start_graph = self.rssm.graph(obs[rows, times, :2])
        positions = [self.decode_positions(stoch, deter)]
        rewards = [self.reward(self.rssm.get_feat(stoch, deter), start_graph)]
        began = time.perf_counter()
        for h in range(1, horizon + 1):
            step_action = action[rows, times + h]
            if self.graph_mode == "rebuilt" and self.locate is not None:
                deter = self.rssm._deter_net(stoch, deter, step_action)
                graph = self.rssm.graph(self.locate(deter))
                stoch, _ = self.rssm.prior(deter, graph)
            else:
                graph = start_graph
                if self.graph_mode == "rebuilt":
                    graph = self.rssm.graph(positions[-1])
                stoch, deter = self.rssm.img_step(stoch, deter, step_action, graph)
            positions.append(self.decode_positions(stoch, deter))
            rewards.append(self.reward(self.rssm.get_feat(stoch, deter), graph))
        elapsed = max(time.perf_counter() - began, 1e-9)
        self.imagined_steps_per_second = rows.numel() * horizon / elapsed
        samples = worlds.numel()
        self.predicted_reward = (
            torch.stack(rewards, 1)
            .reshape(samples, n_agents, horizon + 1)
            .transpose(1, 2)
        ).cpu()
        predicted = torch.stack(positions, 1).reshape(samples, n_agents, horizon + 1, 2)
        return predicted.transpose(1, 2).float().cpu()


def reward_error(
    predicted: torch.Tensor,
    trajectory: Trajectory,
    worlds: torch.Tensor,
    starts: torch.Tensor,
    horizons: Sequence[int] = HORIZONS,
) -> dict[int, float]:
    steps = starts[:, None] + torch.arange(predicted.shape[1])
    true = trajectory.reward[worlds[:, None], steps]
    error = (predicted - true).abs()
    return {h: error[:, h].mean().item() for h in horizons if h < predicted.shape[1]}


def evaluate_graph_prediction(
    trajectory: Trajectory,
    predictors: dict[str, PositionPredictor],
    comm_range: float,
    horizons: Sequence[int] = HORIZONS,
    stride: int = 1,
) -> dict[str, dict]:
    horizon = max(horizons)
    worlds, starts = valid_starts(trajectory.is_first, horizon, stride)
    true = true_positions(trajectory, worlds, starts, horizon)
    results = {"windows": worlds.numel()}
    for name, predictor in predictors.items():
        predicted = predictor(trajectory, worlds, starts, horizon)
        result = {"graph_f1": graph_f1(predicted, true, comm_range, horizons)}
        if isinstance(predictor, WorldModelPredictor):
            result["reward_error"] = reward_error(
                predictor.predicted_reward, trajectory, worlds, starts, horizons
            )
            result["position_error"] = {
                h: (predicted[:, h] - true[:, h]).norm(dim=-1).mean().item()
                for h in horizons
            }
            result["imagined_steps_per_second"] = predictor.imagined_steps_per_second
        results[name] = result
    return results
