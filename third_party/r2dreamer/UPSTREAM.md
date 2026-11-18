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
- `configs/configs.yaml`: default env `dmc_proprio`, adds `wandb_project` and `checkpoint_minutes`.
- `configs/model/_base_.yaml`: default `rep_loss` is `dreamer`.

## Run
`MUJOCO_GL=egl PYTHONPATH=. uv run python third_party/r2dreamer/train.py logdir=runs/<name> env.steps=1e5 env.eval_episode_num=0`

With rendering off, the default 5e5-transition buffer fits the RTX 4060. A 20K-step run peaked at 6.9 GB of 8 GB, desktop included.
