import argparse
import json
import sys
import time
from pathlib import Path

import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "third_party" / "r2dreamer")]

from co_imaginer.evaluation import (
    HORIZONS,
    LINK_MODELS,
    AgentPolicy,
    WorldModelPredictor,
    episode_metrics,
    evaluate_graph_prediction,
    frozen_positions,
    rollout,
    with_links,
)
from co_imaginer.vmas_agents import VmasAgents

CHECKPOINT_NAMES = ("latest.pt", "checkpoint.pt")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--links", choices=sorted(LINK_MODELS), default="perfect")
    parser.add_argument("--graph-mode", choices=["frozen", "rebuilt"])
    parser.add_argument("--worlds", type=int, default=16)
    parser.add_argument("--steps", type=int, default=505)
    parser.add_argument("--horizons", type=int, nargs="+", default=list(HORIZONS))
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def find_checkpoint(run_dir: Path, checkpoint: Path | None) -> Path:
    if checkpoint is not None:
        return checkpoint
    for name in CHECKPOINT_NAMES:
        if (run_dir / name).exists():
            return run_dir / name
    raise FileNotFoundError(f"no checkpoint in {run_dir}")


def load_agent(run_dir: Path, checkpoint: Path, device: str):
    from dreamer import Dreamer

    config = OmegaConf.load(run_dir / ".hydra" / "config.yaml")
    config.device = device
    config.model.device = device
    config.model.compile = False
    envs = VmasAgents(
        config.env.task.split("_", 1)[1],
        1,
        config.env.n_agents,
        config.env.max_steps,
        0,
        device,
        **config.env.scenario_kwargs,
    )
    agent = Dreamer(config.model, envs.observation_space, envs.action_space).to(device)
    state = torch.load(checkpoint, map_location=device)
    agent.load_state_dict(state["agent_state_dict"])
    agent.eval()
    return agent, config


def main(argv: list[str] | None = None) -> dict:
    args = parse_args(argv)
    checkpoint = find_checkpoint(args.run_dir, args.checkpoint)
    agent, config = load_agent(args.run_dir, checkpoint, args.device)
    comm = config.model.comm
    comm_range = float(comm.comm_range)
    with_links(agent._frozen_rssm, LINK_MODELS[args.links])
    envs = VmasAgents(
        config.env.task.split("_", 1)[1],
        args.worlds,
        config.env.n_agents,
        config.env.max_steps,
        args.seed,
        args.device,
        **config.env.scenario_kwargs,
    )
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    began = time.monotonic()
    trajectory = rollout(envs, AgentPolicy(agent), args.steps, int(config.env.n_agents))
    episodes = episode_metrics(trajectory)
    predictors = {"frozen": frozen_positions}
    if hasattr(agent, "decoder"):
        graph_mode = args.graph_mode or "frozen"
        predictors["world_model"] = WorldModelPredictor.from_agent(agent, graph_mode)
    graphs = evaluate_graph_prediction(
        trajectory, predictors, comm_range, args.horizons, args.stride
    )
    results = {
        "run_dir": str(args.run_dir),
        "checkpoint": str(checkpoint),
        "communication": bool(comm.enabled),
        "links": args.links,
        "comm_range": comm_range,
        "seed": args.seed,
        "worlds": args.worlds,
        "steps": args.steps,
        "episodes": episodes.episodes,
        "return_mean": episodes.mean_return,
        "return_std": episodes.std_return,
        "collisions_mean": episodes.mean_collisions,
        **graphs,
        "wall_clock_seconds": time.monotonic() - began,
        "peak_gpu_memory_gb": (
            torch.cuda.max_memory_allocated() / 1e9
            if torch.cuda.is_available()
            else 0.0
        ),
    }
    output = args.output or args.run_dir / f"evaluation_{args.links}.json"
    output.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    main()
