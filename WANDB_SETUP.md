# W&B setup

Training runs log their curves to Weights & Biases (W&B), now part of CoreWeave Forge. Do this once per machine.

1. Create a free account at https://wandb.ai. It signs you in through CoreWeave Forge. Apply for the free academic plan with your university email.
2. Copy your API key from https://wandb.ai/authorize.
3. In the repo, run `uv run wandb login` and paste the key. It is saved in `~/.netrc`, so later runs find it on their own.
4. Accept the invite to the team `uowaterloo-wmmal`. Ask AM for one if you have none.
5. Train as in `README.md`. Runs appear in the team project `uowaterloo-wmmal/co-imaginer`, where every team member sees them. They are named after the method, task and seed, for example `codreamer_vmas_discovery_n4_s1`. Use a seed nobody else has run, because the run name is also the W&B run ID and reusing it continues the old run.
6. Without internet, set `WANDB_MODE=offline` before training and upload later with `uv run wandb sync --entity uowaterloo-wmmal --project co-imaginer <run folder>/wandb/offline-run-*`.

Never commit the API key. Log only scalar metrics, because the free plan stores 5 GB per month.
