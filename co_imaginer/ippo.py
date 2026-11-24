# IPPO baseline on VMAS Discovery, PLAN.md item 4.
#
# IPPO is independent PPO: every robot runs PPO on its own observation, with
# shared policy weights and no communication. It is one of the two baselines.
# BenchMARL provides the whole algorithm. This file only configures it and adds
# a random-policy score to compare against.
#
# Result on this machine (4 robots, seed 0, 3M frames): final evaluation return
# 1.97 per robot per episode, against 0.11 for a random policy.
#
# Run: uv run python -m co_imaginer.ippo --agents 4 --seed 0
# Output: runs/ippo/ippo_n<robots>_s<seed>/ with BenchMARL CSV logs, a final
# checkpoint and random_baseline.json.
#
# To undo: delete this file and tests/test_ippo.py. Nothing else imports it.
import argparse
import json
from pathlib import Path

import torch
import vmas
from benchmarl.algorithms import IppoConfig
from benchmarl.environments import VmasTask
from benchmarl.experiment import Experiment, ExperimentConfig
from benchmarl.models.mlp import MlpConfig


def discovery_task_config(n_agents: int) -> dict:
    # BenchMARL's stock Discovery settings (100-step episodes, shared reward),
    # with only the robot count changed. The IDreamer env config copies these.
    config = VmasTask.DISCOVERY.get_from_yaml().config
    config["n_agents"] = n_agents
    return config


def make_experiment(
    n_agents: int,
    seed: int,
    max_frames: int,
    save_folder: Path,
    device: str,
) -> Experiment:
    task = VmasTask.DISCOVERY.get_from_yaml()
    task.config = discovery_task_config(n_agents)
    # Start from BenchMARL's default experiment settings and change only these.
    config = ExperimentConfig.get_from_yaml()
    config.sampling_device = device
    config.train_device = device
    config.buffer_device = device
    # Total environment frames to train for (frames = steps x worlds).
    config.max_n_frames = max_frames
    # 60 parallel VMAS worlds collecting data at once.
    config.on_policy_n_envs_per_worker = 60
    # Evaluate every 120,000 frames on 32 episodes.
    config.evaluation_interval = 120_000
    config.evaluation_episodes = 32
    config.render = False
    # CSV logs only, so no W&B account is needed for this baseline.
    config.loggers = ["csv"]
    config.checkpoint_at_end = True
    config.save_folder = str(save_folder)
    save_folder.mkdir(parents=True, exist_ok=True)
    # BenchMARL default IPPO hyperparameters and MLP sizes for actor and critic.
    return Experiment(
        task=task,
        algorithm_config=IppoConfig.get_from_yaml(),
        model_config=MlpConfig.get_from_yaml(),
        critic_model_config=MlpConfig.get_from_yaml(),
        seed=seed,
        config=config,
    )


def random_episode_return(n_agents: int, episodes: int, seed: int) -> float:
    # Average return per robot per episode when every robot acts uniformly at
    # random. This is the bar "beats a random policy" in PLAN.md refers to.
    task_config = discovery_task_config(n_agents)
    max_steps = task_config.pop("max_steps")
    # One VMAS world per episode, all run in parallel on the CPU.
    env = vmas.make_env(
        "discovery",
        num_envs=episodes,
        device="cpu",
        continuous_actions=True,
        seed=seed,
        **task_config,
    )
    env.reset()
    returns = torch.zeros(episodes, n_agents)
    for _ in range(max_steps):
        _, rewards, _, _ = env.step([env.get_random_action(a) for a in env.agents])
        returns += torch.stack(rewards, dim=1)
    return float(returns.mean())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agents", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--frames", type=int, default=3_000_000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, default=Path("runs/ippo"))
    args = parser.parse_args()

    run_dir = args.output / f"ippo_n{args.agents}_s{args.seed}"
    # Random baseline first (1,000 episodes, a few seconds on the CPU).
    baseline = random_episode_return(args.agents, episodes=1000, seed=args.seed)
    print(f"random policy episode return: {baseline:.3f}")
    experiment = make_experiment(
        args.agents, args.seed, args.frames, run_dir, args.device
    )
    # Trains IPPO. BenchMARL writes its logs and checkpoint under run_dir.
    experiment.run()
    (run_dir / "random_baseline.json").write_text(
        json.dumps({"random_episode_return": baseline})
    )


if __name__ == "__main__":
    main()
