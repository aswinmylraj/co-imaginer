# IDreamer results (PLAN.md item 6)

One IDreamer run, 4 robots, VMAS Discovery, seed 0, 100,000 environment steps, finished 22 Nov 2026 on an RTX 5060 Ti (16 GB). Code: branch `idreamer-vmas-fixes` (PR #9), `model.compile=False`.

- `idreamer_n4_s0_eval_curve.csv`: evaluation every 10,000 steps (8 episodes): mean return and collisions per robot.
- `idreamer_n4_s0_eval.json`: final test of `latest.pt` on 256 episodes, with a random policy on the same 256 episodes.
- `idreamer_n4_s0_timing.json`: exit code, wall seconds and GPU name.
- `idreamer_n4_s0_score_vs_step.png`: training and evaluation return against steps.

## Result
| Policy | Return per episode | Standard error | Collisions per robot |
|---|---|---|---|
| IDreamer | 2.43 | 0.09 | 2.80 |
| Random | 0.08 | 0.02 | 0.34 |

IDreamer beats random by about 30 times.

## Timing
Wall time 21,759 s for 100,000 steps. Steady speed 6.15 steps per second, so about 4.5 hours per 100,000 steps, inside the 6 GPU-hour cap. Between about 22:10 and 00:15 another program on the same GPU slowed the run to about 1.6 steps per second, so the wall time is longer than 4.5 hours. Training used about 7.4 GB of GPU memory above the idle desktop.

Kaggle T4 or P100 timing is still to be measured with `bin/kaggle-time-idreamer.sh`. No RTX 4060 timing yet.

Checkpoints (120 MB each), raw logs and W&B files are not committed (AGENTS.md).
