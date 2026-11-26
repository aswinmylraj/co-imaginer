import argparse
import json
import time
from pathlib import Path

import torch
import vmas
from benchmarl.algorithms import IppoConfig
from benchmarl.environments import VmasTask
from benchmarl.experiment import Experiment, ExperimentConfig
from benchmarl.models.mlp import MlpConfig
from torchrl.envs.utils import ExplorationType, set_exploration_type

TASK_KWARGS = {
    "max_steps": 100,
    "n_agents": 4,
    "n_targets": 7,
    "lidar_range": 0.35,
    "covering_range": 0.25,
    "agents_per_target": 2,
    "targets_respawn": True,
    "shared_reward": True,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-agents", type=int, default=4)
    parser.add_argument("--frames", type=int, default=3_000_000)
    parser.add_argument("--frames-per-batch", type=int, default=6000)
    parser.add_argument("--n-envs", type=int, default=60)
    parser.add_argument("--eval-interval", type=int, default=120_000)
    parser.add_argument("--checkpoint-interval", type=int, default=600_000)
    parser.add_argument("--random-episodes", type=int, default=200)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--out", type=Path, default=Path("logdir/ippo_discovery"))
    return parser.parse_args()


def task_kwargs(n_agents: int) -> dict:
    return {**TASK_KWARGS, "n_agents": n_agents}


def random_policy_return(kwargs: dict, episodes: int, seed: int, device: str) -> dict:
    env = vmas.make_env(
        "discovery",
        num_envs=episodes,
        device=device,
        continuous_actions=True,
        seed=seed,
        **{k: v for k, v in kwargs.items() if k != "max_steps"},
    )
    env.reset()
    returns = torch.zeros(episodes, kwargs["n_agents"], device=device)
    for _ in range(kwargs["max_steps"]):
        actions = [env.get_random_action(agent) for agent in env.agents]
        _, rewards, _, _ = env.step(actions)
        returns += torch.stack(rewards, dim=-1)
    per_episode = returns.mean(-1)
    return {
        "episodes": episodes,
        "mean": per_episode.mean().item(),
        "std": per_episode.std().item(),
    }


def trained_policy_return(experiment: Experiment, episodes: int) -> dict:
    returns = []
    with torch.no_grad(), set_exploration_type(ExplorationType.DETERMINISTIC):
        while sum(r.numel() for r in returns) < episodes:
            rollout = experiment.test_env.rollout(
                max_steps=experiment.max_steps,
                policy=experiment.policy,
                auto_cast_to_device=True,
                break_when_any_done=False,
            )
            reward = rollout.get(("next", "agents", "reward"))
            returns.append(reward.sum(1).mean((-2, -1)).cpu())
    per_episode = torch.cat(returns)[:episodes]
    return {
        "episodes": per_episode.numel(),
        "mean": per_episode.mean().item(),
        "std": per_episode.std().item(),
    }


def build_experiment(args: argparse.Namespace, save_folder: Path) -> Experiment:
    task = VmasTask.DISCOVERY.get_from_yaml()
    task.config.update(task_kwargs(args.n_agents))

    experiment_config = ExperimentConfig.get_from_yaml()
    experiment_config.sampling_device = args.device
    experiment_config.train_device = args.device
    experiment_config.buffer_device = args.device
    experiment_config.max_n_frames = args.frames
    experiment_config.on_policy_collected_frames_per_batch = args.frames_per_batch
    experiment_config.on_policy_n_envs_per_worker = args.n_envs
    experiment_config.evaluation_interval = args.eval_interval
    experiment_config.evaluation_episodes = 32
    experiment_config.render = False
    experiment_config.loggers = ["csv", "tensorboard"]
    experiment_config.save_folder = str(save_folder)
    experiment_config.checkpoint_interval = args.checkpoint_interval
    experiment_config.checkpoint_at_end = True

    return Experiment(
        task=task,
        algorithm_config=IppoConfig.get_from_yaml(),
        model_config=MlpConfig.get_from_yaml(),
        critic_model_config=MlpConfig.get_from_yaml(),
        seed=args.seed,
        config=experiment_config,
    )


def main() -> None:
    args = parse_args()
    save_folder = args.out / f"n{args.n_agents}_s{args.seed}"
    save_folder.mkdir(parents=True, exist_ok=True)

    random_stats = random_policy_return(
        task_kwargs(args.n_agents), args.random_episodes, args.seed, args.device
    )
    print(f"random policy return: {random_stats}", flush=True)

    experiment = build_experiment(args, save_folder)
    start = time.time()
    experiment.run()
    wall_clock = time.time() - start

    trained_stats = trained_policy_return(experiment, args.random_episodes)
    summary = {
        "task": task_kwargs(args.n_agents),
        "seed": args.seed,
        "frames": args.frames,
        "wall_clock_s": wall_clock,
        "peak_gpu_mem_mb": torch.cuda.max_memory_allocated() / 2**20
        if torch.cuda.is_available()
        else 0.0,
        "random_policy": random_stats,
        "trained_policy": trained_stats,
        "beats_random": trained_stats["mean"] > random_stats["mean"],
        "folder": str(experiment.folder_name),
    }
    (save_folder / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
