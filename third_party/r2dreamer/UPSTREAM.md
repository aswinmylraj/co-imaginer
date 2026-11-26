# Upstream

- Source: https://github.com/NM512/r2dreamer (MIT, see `LICENSE`)
- Commit: becf3387a495e5291d2fc7af94763bebc2f9afef (2026-10-17)
- Paper: R2-Dreamer, Morihira et al., ICLR 2026, arXiv:2603.18202. We use its DreamerV3 path (`model.rep_loss=dreamer`).

## Files copied
Core training code and the DMC proprio and 12M configs only: `buffer.py`, `distributions.py`, `dreamer.py`, `networks.py`, `rssm.py`, `tools.py`, `train.py`, `trainer.py`, `envs/{__init__,dmc,parallel,wrappers}.py`, `optim/`, `configs/configs.yaml`, `configs/env/dmc_proprio.yaml`, `configs/model/{_base_,size12M}.yaml`.

## Local changes
- `tools.py`: `Logger` writes scalars to the W&B logger passed in, instead of TensorBoard. Images, videos, histograms and `log_hydra_config` are dropped.
- `train.py`: creates the W&B logger (run ID `dreamerv3_<task>_s<seed>`) and a `co_imaginer.run.Checkpointer`.
- `trainer.py`: saves agent and optimizer state through the checkpointer at each metrics log, every `checkpoint_minutes`.
- `envs/dmc.py`, `envs/__init__.py`: `should_render_image` turns off the 64x64 camera image. Upstream renders it every step even in proprio mode, which costs a GPU render context per env worker and 12 KB per buffer slot.
- `configs/env/dmc_proprio.yaml`: `should_render_image: False`.
- `envs/__init__.py`: a `vmas` suite builds `co_imaginer.vmas_agents.VmasAgents`, where each robot in each VMAS world is one batch row. This is IDreamer: one shared DreamerV3, no messages.
- `configs/env/vmas_discovery.yaml` (new): 4 robots, 4 worlds, 100-step episodes, Discovery settings matching BenchMARL (`shared_reward: True`).
- `train.py`: VMAS runs are named `idreamer_<task>_n<robots>_s<seed>`.
- `configs/configs.yaml`: default env `dmc_proprio`, adds `wandb_project` and `checkpoint_minutes`.
- `configs/model/_base_.yaml`: default `rep_loss` is `dreamer`.
- `configs/configs.yaml`: `buffer.max_size` reads `env.buffer_size` and falls back to 5e5. `configs/env/vmas_discovery.yaml` sets 2e5, twice `env.steps`, because the step counter skips reset steps. A Discovery run never fills it, so sampling matches 5e5.
- CoDreamer (Toledo and Prorok, arXiv:2406.13600), off unless `model.comm.enabled=True`:
  - `configs/model/_base_.yaml`: `comm` block with `comm_range` 0.5, 4 heads, 256-wide messages, `n_agents` from the env.
  - `rssm.py`: one `co_imaginer.graph.GraphLayer` before the posterior network and one before the prior network. `RSSM.graph` builds the graph from positions.
  - `networks.py`: `MLPHead.forward` accepts and ignores a graph. `GraphMLPHead` adds one GATv2 layer before the MLP.
  - `dreamer.py`: reward, continue, actor and critic use `GraphMLPHead`. The decoder does not. Imagination keeps the starting graph for every step.
  - `buffer.py`: with `aligned`, replay samples whole worlds over one shared time window.
  - `configs/configs.yaml`: `buffer.aligned` and `buffer.n_agents` follow `model.comm`.
  - `train.py`: runs are named `codreamer_<task>_n<robots>_s<seed>`.

## How to revert CoDreamer
- Turn it off without touching code: `model.comm.enabled=False`, the default. A test (`tests/test_rssm.py`) checks that the model is then identical to IDreamer, weight for weight.
- Remove it from the code: `git revert` the CoDreamer commits, or restore the files listed above from the `idreamer-vmas-fixes` branch with `git checkout idreamer-vmas-fixes -- <file>`. Then delete `co_imaginer/graph.py`, `tests/test_graph.py`, `tests/test_buffer.py` and `tests/test_rssm.py`, and set `pythonpath = ["."]` back in `pyproject.toml`.
- Undo only the dense GATv2 speed fix: run with `model.comm.dense=False` (sparse torch_geometric path, same weights and checkpoints), or `git revert` the dense GATv2 commit.
- Every CoDreamer change in the code carries a comment that names CoDreamer, the graph or the comm switch, so `git grep -n -i "codreamer\|graph\|comm" third_party/r2dreamer` lists them.

## Run
`MUJOCO_GL=egl PYTHONPATH=. uv run python third_party/r2dreamer/train.py logdir=runs/<name> env.steps=1e5 env.eval_episode_num=0`

CoDreamer on Discovery: `PYTHONPATH=. uv run python third_party/r2dreamer/train.py env=vmas_discovery model.comm.enabled=True logdir=runs/<name>`

With rendering off, the default 5e5-transition buffer fits the RTX 4060. A 20K-step run peaked at 6.9 GB of 8 GB, desktop included.
