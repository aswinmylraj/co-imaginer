# project-co-imaginer

## Setup (Ubuntu)

```bash
cd project-co-imaginer
export PATH="$PWD/bin:$PATH"
setup-repo.sh
setup-bazel.sh
uv run wandb login  # see WANDB_SETUP.md
```

Both are safe to run again. `setup-bazel.sh` installs Bazel 8.8.2, fetches the pinned uv, ruff and gitleaks, and creates `.venv` with Python 3.13.18. `setup-repo.sh`:

- installs curl, direnv, Node.js and the TeX packages with apt (asks for sudo)
- runs `system-info.sh`, which records this machine's hardware in `.agents/SYSTEM_INFORMATION.md` for agents
- writes `.envrc`, which keeps `bin` and `.venv` on your PATH inside the repo
- links `bin/hooks/pre-commit` into `.git/hooks` and turns on commit signing
- downloads the Ponytail rules to `.agents/PONYTAIL_RULES.md`

## Train

Teaches a simulated robot to walk with DreamerV3 (`third_party/r2dreamer`), 30 minutes on an RTX 4060.

```bash
uv sync
MUJOCO_GL=egl PYTHONPATH=. uv run python third_party/r2dreamer/train.py logdir=runs/walker seed=1 env.steps=1e5 env.eval_episode_num=0
```

Use your own seed. RTX 20 series and newer GPUs work with `uv sync` if `nvidia-smi` shows driver 570 or newer.

## Agents

`AGENTS.md` holds the rules for every agent, Claude Code included. Claude Code enables Ponytail from `.claude/settings.json`, which also covers local Ollama models. Other agents read `.agents/PONYTAIL_RULES.md`.

## Documentation

```bash
generate-documentation.sh
```

Writes `documentation/progress-<date>.tex` and builds the PDFs. Set `DOCUMENTATION_LLM=ollama:<model>` to use a local model.
