# Dynamic communication graphs in multi-robot world-model imagination

Team: AM, JZ, TKBH. University of Waterloo, Fall 2026, instructor Prof. Firoozi.

## Project in one paragraph
CoDreamer (Toledo and Prorok, arXiv:2406.13600) extends DreamerV3 to multiple agents with GATv2 message passing over a graph that connects agents within communication range C. It freezes that graph for all imagined steps and assumes perfect links. We rebuild the graph at every imagined step from positions predicted by the world model, and add realistic links: distance-based packet drops from a path-loss model, and a directed k-nearest receive cap. All variants use the same 15-step imagination horizon.

## Experiment design (from the submitted proposal)
- 2x2 ablation: graph frozen or rebuilt, links perfect or realistic. Frozen with perfect links reproduces CoDreamer.
- Baselines: IPPO (BenchMARL) and IDreamer (independent DreamerV3 per agent, shared weights).
- Main task: VMAS Discovery. Flocking is a stretch goal.
- Priority 1: 6 methods x 4 robots x 3 seeds = 18 runs. Priority 2: same with 8 robots. Stretch: 16 robots.
- Evaluation only: all four variants tested under realistic links. Graph F1 at 15, 30 and 50 imagined steps from saved world models.
- Logged per run: return, collisions, graph F1, reward-prediction error, wall-clock, peak GPU memory, imagined steps per second.
- Feasibility check before training: fraction of edges that change within 15 steps. If under 10 percent, tune C or robot speed within physically plausible values, fixed before training any method.
- No public CoDreamer code was found. It must be reimplemented.

## Stack
Python 3.13.18, torch 2.10.0 (CUDA 12.6 build, supports P100, T4 and RTX 4060), torchrl 0.11.1, vmas 1.5.2, torch-geometric 2.8.1 (GATv2Conv), benchmarl 1.5.2, DreamerV3 base NM512/dreamerv3-torch (archived) or r2dreamer.
- Python and packages: `pyproject.toml` and `uv.lock`. Run Python with `uv run`, for example `uv run python train.py`. Change dependencies with `uv add` or `uv lock`.
- Tools: Bazel 8.8.2 (`.bazelversion`) pins uv 0.13.7, ruff 0.17.4 and gitleaks 8.30.1 with checksums in `MODULE.bazel`. `bin/uv` runs the pinned uv. `.bazelignore` keeps `.venv` out of Bazel, so `bazel query //...` and `bazel build //...` stay clean.
- Kaggle: `pip install uv==0.13.7`, then `uv sync --locked`.

## Code rules for this repo
Agents other than Claude Code: also follow `.agents/PONYTAIL_RULES.md`, created by `bin/setup-repo.sh`.
These add to the user's general review style.
- Format and lint with `ruff format` and `ruff check`. PEP 8 names. Type hints on public functions.
- Markdown file names are ALL CAPS with underscores, such as `PLAN.md` and `SYSTEM_INFORMATION.md`.
- Tests use pytest, one `test_{function}` per function, with new cases added to its `@pytest.mark.parametrize` list.
- Comments: the only allowed comment is a short source reference for an implemented equation or algorithm, such as `# Eq. 1, proposal` or `# DreamerV3 Sec. 3`. No docstrings.
- Hooks: `bin/hooks/pre-commit`, linked into `.git/hooks` by `bin/setup-repo.sh`, runs ruff, gitleaks and a large-file check. No hook framework. Never commit checkpoints, logs or Kaggle and W&B tokens.
- Experiment impact, stated for every proposed fix:
  1. Results: does it change results of runs already finished, breaking the 2x2 comparison?
  2. Cost: GPU-hours and peak GPU memory against the 6-hour cap.
  3. Reproducibility: seeds, config and checkpoint compatibility.

## Compute
- This machine: read `.agents/SYSTEM_INFORMATION.md`. `bin/system-info.sh` writes it during setup. Rerun it after hardware changes.
- Kaggle T4 or P100, about 30 GPU-hours per week per account (unofficial), 12-hour sessions. One account per member.
- Reserve: rented RTX 4090, 100 CAD cap.
- Each run must finish in 6 GPU-hours or less. Checkpoint every 30 minutes.

## Dates
Midterm 13 Nov. Experiment freeze 25 Nov. Presentation 27 Nov or week of 4 Dec (15 minutes). Final report 7 Dec (6 pages, IEEE double column).
Grading: technical depth 30, experiments 25, clarity 20, originality 15, relevance 10.

## Files in this folder
- AGENTS.md: this file, the rules for every agent. Claude Code reads it directly, so there is no CLAUDE.md.
- README.md: setup steps, 200 words at most.
- bin/setup-repo.sh: apt packages, .envrc, git hooks and .agents/PONYTAIL_RULES.md.
- bin/setup-bazel.sh: bazelisk, the pinned tools and `uv sync`.
- bin/generate-documentation.sh: switches to main, pulls, and writes `documentation/progress-<date>.tex` (commit hash, changes, benchmarks) from the git diff since the previous progress file. An LLM writes the changes section following this file. `DOCUMENTATION_LLM` picks it: `claude` (default) or `ollama:<model>`. Builds the PDFs of every tex file in `documentation/`. Leaves the new file uncommitted.
- bin/system-info.sh: writes `.agents/SYSTEM_INFORMATION.md` with this machine's OS, CPU, RAM, GPU and free disk.
- documentation/proposal_submitted.tex: proposal as submitted (team version).
- documentation/progress-<date>.tex: progress reports. PDFs next to them are generated and ignored.
- PLAN.md: 16 dated action items with hardware, software and deadline.
- 2301.04104v2.pdf (DreamerV3) and ECE780_Course_Outline.pdf: add these yourself.

## Known follow-ups for the final report
- Add cost metrics and the return against GPU-hours trade-off.
- With 3 seeds, report mean and standard deviation, not IQM with bootstrap intervals.
- State that 12M-parameter models target robots with embedded GPUs, not nano-drones.

## How to work with me
- Be concise. Verify facts against primary sources and cite them.
- Writing: clear, professional English. No semicolons or slashes in prose, no hedging words. Symbols only in equations and references.
- Answer one point at a time when I ask for it. Make minimal changes to documents near deadlines and list exact edits.
