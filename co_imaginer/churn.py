# Churn check, PLAN.md item 3.
#
# Question it answers: if the communication graph is frozen at the start of a
# 15-step imagination (as CoDreamer does), how wrong is that graph by the end?
# The proposal requires at least 10 percent of edges to change within 15 steps,
# otherwise rebuilding the graph cannot help and the 2x2 ablation is pointless.
#
# What it does: runs VMAS Discovery with random or scripted robots, records every
# robot position at every step, turns positions into edges (robots within range
# C), and measures how many edges change over a horizon.
#
# Result used by the project: C = 0.5 and robot speed 1.0 (configs/run.yaml).
# With 4 robots, random policy, horizon 15: churn 0.28, flipped_within 0.33.
#
# To undo: delete this file and tests/test_churn.py. Nothing else imports it.
import argparse
import csv
import itertools
from pathlib import Path

import torch
import vmas
from vmas.scenarios.discovery import HeuristicPolicy

# The two behaviours tested: uniform random actions, and VMAS's built-in
# heuristic that drives robots towards targets.
POLICIES = ("random", "scripted")


def edge_matrix(positions: torch.Tensor, comm_range: float) -> torch.Tensor:
    # positions (..., n_agents, 2) -> edges (..., n_agents, n_agents) as booleans.
    # An edge i-j exists when the robots are closer than comm_range.
    distances = torch.cdist(positions, positions)
    n_agents = positions.shape[-2]
    # Keep only the upper triangle: links are symmetric here, so each pair is
    # counted once, and a robot is never linked to itself.
    upper = torch.ones(n_agents, n_agents, dtype=torch.bool).triu(diagonal=1)
    return (distances < comm_range) & upper.to(positions.device)


def churn_counts(edges: torch.Tensor, horizon: int) -> dict[str, float]:
    # edges (time, ..., n, n). Every time step t is the start of one window that
    # ends at t + horizon. All counts are summed over windows, worlds and pairs.
    windows = edges.shape[0] - horizon
    start = edges[:windows]  # graph at the start of each window (the frozen graph)
    end = edges[horizon:]  # true graph at the end of each window
    changed = start ^ end  # edge present at one end and not the other
    union = start | end  # edge present at either end
    # flips: edge switched on or off at least once inside the window, even if it
    # returned to its starting state by the end.
    flips = torch.zeros_like(start)
    # ever_present: edge existed at any step of the window.
    ever_present = start.clone()
    for offset in range(1, horizon + 1):
        flips |= (
            edges[offset - 1 : offset - 1 + windows] ^ edges[offset : offset + windows]
        )
        ever_present |= edges[offset : offset + windows]
    return {
        "changed": float(changed.sum()),
        "union": float(union.sum()),
        "start": float(start.sum()),
        "end": float(end.sum()),
        "kept": float((start & end).sum()),
        "flipped": float(flips.sum()),
        "ever_present": float(ever_present.sum()),
    }


def churn_metrics(counts: dict[str, float]) -> dict[str, float]:
    # Turns raw counts into the three numbers reported in runs/churn/*.csv.
    denominator = counts["start"] + counts["end"]
    return {
        # Fraction of edges (present at start or end) that differ between the two.
        # This is the "fraction of edges that change within 15 steps" in AGENTS.md.
        "churn": counts["changed"] / counts["union"] if counts["union"] else 0.0,
        # F1 of the frozen start graph used as a prediction of the end graph.
        # 1.0 means freezing the graph loses nothing.
        "frozen_f1": 2 * counts["kept"] / denominator if denominator else 1.0,
        # Fraction of edges that flipped at least once during the window.
        "flipped_within": counts["flipped"] / counts["ever_present"]
        if counts["ever_present"]
        else 0.0,
    }


def rollout_positions(
    n_agents: int,
    policy: str,
    speed_scale: float,
    steps: int,
    num_envs: int,
    seed: int,
) -> torch.Tensor:
    # Runs num_envs Discovery worlds for `steps` steps on the CPU and returns
    # positions (steps + 1, num_envs, n_agents, 2), including the reset state.
    env = vmas.make_env(
        "discovery",
        num_envs=num_envs,
        device="cpu",
        continuous_actions=True,
        seed=seed,
        n_agents=n_agents,
    )
    # Robot speed: VMAS multiplies each action by _u_multiplier before applying
    # it as a force. Scaling it scales how fast robots can move. Clearing the
    # cached tensor forces VMAS to rebuild it with the new value.
    for agent in env.agents:
        agent.action._u_multiplier = speed_scale
        agent.action._u_multiplier_tensor = None
    heuristic = HeuristicPolicy(continuous_action=True)
    observations = env.reset()
    positions = [torch.stack([agent.state.pos for agent in env.agents], dim=1)]
    for _ in range(steps):
        if policy == "random":
            actions = [env.get_random_action(agent) for agent in env.agents]
        else:
            actions = [
                heuristic.compute_action(obs, agent.u_range)
                for obs, agent in zip(observations, env.agents)
            ]
        observations, _, _, _ = env.step(actions)
        positions.append(torch.stack([agent.state.pos for agent in env.agents], dim=1))
    return torch.stack(positions)


def main() -> None:
    # Example sweep (the one behind runs/churn/sweep.csv):
    #   uv run python -m co_imaginer.churn --ranges 0.25 0.35 0.5 \
    #     --speeds 0.5 1.0 2.0 --output runs/churn/sweep.csv
    parser = argparse.ArgumentParser()
    parser.add_argument("--agents", type=int, nargs="+", default=[4, 8])
    parser.add_argument("--policies", nargs="+", default=list(POLICIES))
    parser.add_argument("--ranges", type=float, nargs="+", default=[0.35])
    parser.add_argument("--speeds", type=float, nargs="+", default=[1.0])
    parser.add_argument("--horizons", type=int, nargs="+", default=[15, 30])
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--output", type=Path, default=Path("runs/churn/churn.csv"))
    args = parser.parse_args()

    rows = []
    # One rollout per (robots, policy, speed, seed). Range and horizon only change
    # how positions are turned into edges, so they reuse the same rollouts.
    for n_agents, policy, speed in itertools.product(
        args.agents, args.policies, args.speeds
    ):
        trajectories = [
            rollout_positions(n_agents, policy, speed, args.steps, args.num_envs, seed)
            for seed in args.seeds
        ]
        for comm_range, horizon in itertools.product(args.ranges, args.horizons):
            per_seed = [
                churn_metrics(churn_counts(edge_matrix(p, comm_range), horizon))
                for p in trajectories
            ]
            row = {
                "n_agents": n_agents,
                "policy": policy,
                "speed_scale": speed,
                "comm_range": comm_range,
                "horizon": horizon,
            }
            # Mean and standard deviation over seeds for each metric.
            for key in per_seed[0]:
                values = torch.tensor([m[key] for m in per_seed])
                row[key] = round(float(values.mean()), 4)
                row[f"{key}_std"] = round(float(values.std(correction=0)), 4)
            rows.append(row)
            print(row)

    # One CSV row per setting. runs/ is gitignored, so results stay local.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
