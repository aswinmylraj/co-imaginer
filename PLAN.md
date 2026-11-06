# Execution Plan: Dynamic Communication Graphs in Multi-Robot World-Model Imagination

ECE 780 T13 Robot Learning | AM, JZ, TKBH | Plan date: 6 November 2026

**Fixed dates:** midterm 13 Nov, experiment freeze 25 Nov, presentation 27 Nov or week of 4 Dec, report due 7 Dec.

Hardware lists only what training and testing need. Coding work needs no special hardware. Software lists what the item adds to the shared stack: Python 3.11, PyTorch (CUDA build) and GitHub.

## Action items

### Week 1: setup and baselines

| # | Action | Hardware needed | Software needed | Latest acceptable completion |
|---|---|---|---|---|
| 1 | Set up the repository for training and testing: pinned requirements, one shared run configuration, checkpoint saving every 30 minutes, logging. | Nil | GitHub, Python 3.11, PyTorch (CUDA build), TensorBoard or Weights & Biases | 7 Nov |
| 2 | Select the DreamerV3 base and confirm it learns on one proprioceptive DMC task. | RTX 4060 (8 GB) | NM512/dreamerv3-torch or r2dreamer, dm_control | 9 Nov |
| 3 | Churn check: measure the fraction of edges that change over 15 and 30 steps on Discovery with 4 and 8 robots under random and scripted policies. Fix communication range C and robot speed. | CPU only | vmas 1.5.2 | 9 Nov |
| 4 | Configure IPPO on Discovery with 4 robots and confirm it beats a random policy. | RTX 4060 (8 GB) | benchmarl 1.5.2 | 10 Nov |
| 5 | Build IDreamer: wrap VMAS as per-agent environments with shared world-model and policy weights. | Nil | vmas 1.5.2, DreamerV3 base from item 2 | 11 Nov |
| 6 | Train IDreamer with 4 robots until it beats a random policy, and time 100,000 environment steps on the 4060 and on Kaggle to set steps per run. | RTX 4060 (8 GB), Kaggle T4 or P100 | Kaggle account with phone verification | 12 Nov |

### Week 2: method and Priority 1 runs

| # | Action | Hardware needed | Software needed | Latest acceptable completion |
|---|---|---|---|---|
| 7 | Build CoDreamer: GATv2 layers in the RSSM, reward and continue heads, actor and critic. | Nil | torch-geometric 2.8.1 (GATv2Conv) | 16 Nov |
| 8 | Add the position head and rebuild the graph at every imagined step. Log graph F1 for each step. | Nil | torch-geometric 2.8.1 | 17 Nov |
| 9 | Add realistic links: path-loss drop sampling, attention scaling in imagination and the k-nearest receive cap. Unit test: delivery probability is 0.5 at distance C and above 0.95 at C/2. | Nil | pytest | 17 Nov |
| 10 | Write evaluation scripts: realistic-link deployment test and offline graph F1 at 15, 30 and 50 imagined steps. | Nil | Nil | 18 Nov |
| 11 | Train Priority 1: 6 methods (IPPO, IDreamer and the four ablation variants) on Discovery with 4 robots, horizon 15, 3 seeds. 18 runs, 6 GPU-hours or less each. | RTX 4060 (8 GB), 3 Kaggle accounts (about 30 GPU-hours per week each) | Nil | 22 Nov |

### Week 3: Priority 2 runs and analysis

| # | Action | Hardware needed | Software needed | Latest acceptable completion |
|---|---|---|---|---|
| 12 | Train Priority 2: the same 18 runs with 8 robots. | RTX 4060 (8 GB), 3 Kaggle accounts, rented RTX 4090 as reserve (100 CAD cap, about 200 GPU-hours) | Nil | 25 Nov |
| 13 | Evaluate every finished run: realistic-link deployment test and offline graph F1. | RTX 4060 (8 GB) | Nil | 25 Nov |
| 14 | Produce figures: learning curves, 2x2 results table, graph F1 against horizon, cost table (wall-clock, GPU memory, imagined steps per second). | Nil | matplotlib | 25 Nov |

### Wrap-up

| # | Action | Hardware needed | Software needed | Latest acceptable completion |
|---|---|---|---|---|
| 15 | Build the 15-minute talk and rehearse it once in full. | Nil | Nil | 26 Nov |
| 16 | Write the 6-page IEEE report. | Nil | Overleaf, IEEE conference template | 6 Dec |
| S | Stretch, only if item 12 finishes before 23 Nov: 16 robots, then the Flocking task. | Same as item 12 | Nil | 25 Nov |

## Rules

- **Run length:** if item 6 shows a run needs more than 6 GPU-hours, reduce environment steps per run or use the rented RTX 4090 reserve.
- **Churn:** if fewer than 10% of edges change within 15 steps, lower C or raise robot speed within physically plausible values. Fix both before training any method.
- **Learning:** if IDreamer does not beat a random policy by 14 Nov, stop items 7 to 9, debug against the DMC task from item 2, and drop item 12.
- **Freeze:** no training runs start after 25 Nov. Report only finished runs.
