import argparse
import itertools
import json

import torch
import vmas
from vmas.scenarios.discovery import HeuristicPolicy


def comm_graph(pos: torch.Tensor, comm_range: float) -> torch.Tensor:
    n_agents = pos.shape[-2]
    adjacency = torch.cdist(pos, pos) <= comm_range
    upper = torch.triu(torch.ones(n_agents, n_agents, dtype=torch.bool), diagonal=1)
    return adjacency[..., upper]


def window_churn(edges: torch.Tensor, horizon: int) -> dict[str, float]:
    start = edges[:-horizon]
    end = edges[horizon:]
    changed = (start ^ end).sum(-1).float()
    union = (start | end).sum(-1).float()
    common = (start & end).sum(-1).float()
    total = (start.sum(-1) + end.sum(-1)).float()
    has_edges = union > 0
    frozen_f1 = torch.where(total > 0, 2 * common / total.clamp(min=1), 1.0)
    return {
        "edge_churn": (changed[has_edges] / union[has_edges]).mean().item(),
        "pair_flip_rate": (changed / edges.shape[-1]).mean().item(),
        "frozen_graph_f1": frozen_f1.mean().item(),
        "graph_changed": (changed > 0).float().mean().item(),
    }


def rollout_positions(
    n_agents: int,
    policy: str,
    speed: float,
    n_envs: int,
    n_steps: int,
    seed: int,
) -> tuple[torch.Tensor, float]:
    env = vmas.make_env(
        "discovery",
        num_envs=n_envs,
        device="cpu",
        continuous_actions=True,
        seed=seed,
        n_agents=n_agents,
    )
    for agent in env.agents:
        agent.action._u_multiplier = speed
        agent.action._u_multiplier_tensor = None
    heuristic = HeuristicPolicy(continuous_action=True)
    generator = torch.Generator().manual_seed(seed)
    observations = env.reset()
    positions = []
    velocities = []
    for _ in range(n_steps):
        if policy == "random":
            actions = [
                torch.rand(n_envs, 2, generator=generator) * 2 - 1 for _ in env.agents
            ]
        else:
            actions = [heuristic.compute_action(obs, 1.0) for obs in observations]
        observations, _, _, _ = env.step(actions)
        positions.append(torch.stack([a.state.pos for a in env.agents], dim=1))
        velocities.append(torch.stack([a.state.vel for a in env.agents], dim=1))
    speeds = torch.linalg.vector_norm(torch.stack(velocities), dim=-1)
    return torch.stack(positions), speeds.mean().item()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-agents", type=int, nargs="+", default=[4, 8])
    parser.add_argument("--policies", nargs="+", default=["random", "scripted"])
    parser.add_argument("--ranges", type=float, nargs="+", default=[0.35, 0.5, 0.75])
    parser.add_argument("--speeds", type=float, nargs="+", default=[1.0, 2.0])
    parser.add_argument("--horizons", type=int, nargs="+", default=[15, 30])
    parser.add_argument("--n-envs", type=int, default=128)
    parser.add_argument("--n-steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="churn_check.json")
    args = parser.parse_args()
    rows = []
    for n_agents, policy, speed in itertools.product(
        args.n_agents, args.policies, args.speeds
    ):
        positions, mean_speed = rollout_positions(
            n_agents, policy, speed, args.n_envs, args.n_steps, args.seed
        )
        for comm_range in args.ranges:
            edges = comm_graph(positions, comm_range)
            mean_degree = 2 * edges.sum(-1).float().mean().item() / n_agents
            for horizon in args.horizons:
                row = {
                    "n_agents": n_agents,
                    "policy": policy,
                    "speed_multiplier": speed,
                    "mean_speed": round(mean_speed, 4),
                    "comm_range": comm_range,
                    "mean_degree": round(mean_degree, 3),
                    "horizon": horizon,
                    **{k: round(v, 4) for k, v in window_churn(edges, horizon).items()},
                }
                rows.append(row)
                print(json.dumps(row), flush=True)
    with open(args.out, "w") as handle:
        json.dump(rows, handle, indent=1)


if __name__ == "__main__":
    main()
