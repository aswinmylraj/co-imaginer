# r2dreamer (vendored)

DreamerV3 base for this project (PLAN item 2). Source: https://github.com/NM512/r2dreamer at commit `becf3387a495e5291d2fc7af94763bebc2f9afef` (17 Oct 2026), MIT license in `LICENSE`. Paper: Morihira et al., R2-Dreamer, ICLR 2026.

We run it with `model.rep_loss=dreamer`, which is the repository's PyTorch DreamerV3 reproduction (decoder, reconstruction loss). The upstream README reports it trains about 5x faster than NM512/dreamerv3-torch, which is archived and predates the DreamerV3 revisions.

Kept: the core files, the DMC environment wrappers and the DMC and model configs. Dropped: Atari, Crafter, Meta-World, Memory Maze, IsaacLab, DMC Subtle, Docker and run scripts.

Upstream code keeps its docstrings and comments, so the AGENTS.md comment rule does not apply to this folder. `ruff.toml` sets line length 120 to match upstream and ignores the lint rules that upstream code breaks, so the pre-commit hook passes without rewriting it.

## Local changes

1. `dreamer.py`, `configs/model/_base_.yaml`: `model.compute_dtype` (`bfloat16`, `float16`, `float32`) replaces the hardcoded bfloat16. `float16` adds dynamic loss scaling with `torch.amp.GradScaler` semantics (start 2^16, halve on a non-finite gradient, double after 2000 clean updates). Turing GPUs such as the RTX 2070 SUPER have no native bfloat16.
2. `envs/dmc.py`, `configs/env/dmc_proprio.yaml`: with `env.size: null` the DMC wrapper does not render an image. Proprio runs no longer render 64x64 frames that the model never reads.

3. `envs/vmas_agents.py`, `envs/__init__.py`, `configs/env/vmas_discovery.yaml`: IDreamer on VMAS (PLAN item 5). `VmasAgentEnv` runs one vectorized VMAS scenario and exposes every (environment, agent) pair as its own environment slot, flattened as `env * n_agents + agent`. One `Dreamer` with one set of weights acts for and trains on all slots, which gives independent per-agent world models and policies with shared weights. Each slot keeps its own RSSM state and replay trajectory. Agents of one VMAS environment reset together. Actions in [-1, 1] are scaled by each agent's `u_range`. VMAS `terminated` maps to `is_terminal`, the `time_limit` truncation only to `is_last`. `log_collisions` counts agent pairs closer than the scenario's collision distance, summed per agent over the episode in eval logs. Task arguments match the IPPO baseline (`baselines/ippo_discovery.py`): 4 agents, 7 targets, 100 steps, shared reward. With shared reward every agent gets the team reward, so `episode/eval_score` is the team return, the same quantity the IPPO script reports.

## Step counting on VMAS

The trainer counts transitions per slot, so `env.steps` and `train_ratio` count agent steps. With 4 agents, 100,000 environment steps is `env.steps=4e5`, the config default. `train_ratio` keeps its DreamerV3 meaning: replayed steps per collected agent step.

## Run

From the repository root on the RTX 2070 SUPER:

```bash
cd third_party/r2dreamer
MUJOCO_GL=disable uv run python train.py env=dmc_proprio env.task=dmc_walker_walk model.rep_loss=dreamer model.compute_dtype=float16 env.steps=1e5 logdir=../../logdir/dmc_walker_walk_dreamer_s0
```

IDreamer on Discovery with 4 robots:

```bash
cd third_party/r2dreamer
uv run python train.py env=vmas_discovery model.rep_loss=dreamer model.compute_dtype=float16 logdir=../../logdir/idreamer_discovery_n4_s0
```

8 robots: add `env.n_agents=8 env.steps=8e5`. VMAS runs on `device`, so no env worker processes start.

`MUJOCO_GL=disable` is safe for proprio runs because nothing renders. Use `model.compute_dtype=float32` if float16 skips many updates (`train/opt/updates_skipped` in TensorBoard). Watch with `uv run tensorboard --logdir logdir`.
