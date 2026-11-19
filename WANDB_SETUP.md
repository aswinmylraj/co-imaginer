# W&B setup

Training runs log their curves to Weights & Biases (W&B), now part of CoreWeave Forge. Do this once per machine.

1. Create a free account at https://wandb.ai. It signs you in through CoreWeave Forge. Apply for the free academic plan with your university email.
2. Copy your API key from https://wandb.ai/authorize.
3. In the repo, run `uv run wandb login` and paste the key. It is saved in `~/.netrc`, so later runs find it on their own.
4. Train as in `README.md`. Runs appear in your W&B project `co-imaginer`, named after the task and seed, for example `dreamerv3_dmc_walker_walk_s1`. Use a new seed for each run, because the run name is also the W&B run ID and reusing it continues the old run.
5. Without internet, set `WANDB_MODE=offline` before training and upload later with `uv run wandb sync <run folder>/wandb/offline-run-*`.

Never commit the API key. Log only scalar metrics, because the free plan stores 5 GB per month.
