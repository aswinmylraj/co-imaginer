# Execution Plan: Dynamic Communication Graphs in Multi-Robot World-Model Imagination

AM, JZ, TKBH | Plan date: 6 November 2026, revised 25 November 2026

**Fixed dates:** midterm 13 Nov, experiment freeze 1 Dec, figures 2 Dec, presentation 4 Dec, report due 7 Dec.

Hardware lists only what training and testing need. Coding work needs no special hardware. Software lists what the item adds to the shared stack: Python 3.11, PyTorch (CUDA build) and GitHub.

## Action items

| # | Description | Assigned To | Status | Hardware needed | Software needed | Latest acceptable completion |
|---|---|---|---|---|---|---|
| 0 | Team status check: everyone pushes finished results for items 2 to 6, and `idreamer-vmas` is merged into `main`. | AM, JZ, TKBH | Not done | Nil | GitHub | 19 Nov |
| 1 | Set up the repository for training and testing: pinned requirements, one shared run configuration, checkpoint saving every 30 minutes, logging. | AM | Done | Nil | GitHub, Python 3.11, PyTorch (CUDA build), TensorBoard or Weights & Biases | 7 Nov |
| 2 | Select the DreamerV3 base and confirm it learns on one proprioceptive DMC task. | AM | Done | RTX 4060 (8 GB) | NM512/dreamerv3-torch or r2dreamer, dm_control | 20 Nov |
| 3 | Churn check: measure the fraction of edges that change over 15 and 30 steps on Discovery with 4 and 8 robots under random and scripted policies. Fix communication range C and robot speed. | JZ | Done | CPU only | vmas 1.5.2 | 20 Nov |
| 4 | Configure IPPO on Discovery with 4 robots and confirm it beats a random policy. | JZ | Not done | Kaggle T4 or P100 | benchmarl 1.5.2 | 21 Nov |
| 5 | Build IDreamer: wrap VMAS as per-agent environments with shared world-model and policy weights. | AM, JZ | Not done | Nil | vmas 1.5.2, DreamerV3 base from item 2 | 11 Nov |
| 6 | Train IDreamer with 4 robots until it beats a random policy, and time 100,000 environment steps on the 4060 and on Kaggle to set steps per run. | AM | Not done | RTX 4060 (8 GB), Kaggle T4 or P100 | Kaggle account with phone verification | 22 Nov |
| 7 | Build CoDreamer: GATv2 layers in the RSSM, reward and continue heads, actor and critic. | AM | Not done | Nil | torch-geometric 2.8.1 (GATv2Conv) | 23 Nov |
| 8 | Add the position head and rebuild the graph at every imagined step. Log graph F1 for each step. | AM | Not done | Nil | torch-geometric 2.8.1 | 24 Nov |
| 9 | Add realistic links: path-loss drop sampling, attention scaling in imagination and the learned k-neighbour receive cap (scorer and straight-through Gumbel top-k). Unit tests: delivery probability is 0.5 at distance C and above 0.95 at C/2. With k or fewer neighbours in range, all are selected. With more than k, exactly k are selected and gradients reach the scorer. | AM | Not done | Nil | pytest | 25 Nov |
| 10 | Write evaluation scripts: realistic-link deployment test and offline graph F1 at 15, 30 and 50 imagined steps. | JZ | Not done | Nil | Nil | 25 Nov |
| 11 | Train Priority 1: 6 methods (IPPO, IDreamer and the four ablation variants) on Discovery with 4 robots, horizon 15, 3 seeds. 18 runs, 6 GPU-hours or less each. | TKBH | Not done | RTX 5070 Ti (16 GB), 3 Kaggle accounts (about 30 GPU-hours per week each) | Nil | 29 Nov |
| 12 | Train Priority 2: the same 18 runs with 8 robots. | TKBH | Not done | RTX 5070 Ti (16 GB), 3 Kaggle accounts, rented RTX 4090 as reserve (100 CAD cap, about 200 GPU-hours) | Nil | 1 Dec |
| 13 | Evaluate every finished run: realistic-link deployment test and offline graph F1. | JZ | Not done | Kaggle T4 or P100 | Nil | 1 Dec |
| 14 | Produce figures: learning curves, 2x2 results table, graph F1 against horizon, cost table (wall-clock, GPU memory, imagined steps per second). | JZ | Not done | Nil | matplotlib | 2 Dec |
| 15 | Build the 15-minute talk and rehearse it once in full. | AM, JZ, TKBH | Not done | Nil | Nil | 3 Dec |
| 16 | Write the 6-page IEEE report. | AM, JZ, TKBH | Not done | Nil | Overleaf, IEEE conference template | 6 Dec |
| S | Stretch, only if item 12 finishes before 30 Nov: 16 robots, then the Flocking task. | TKBH | Not done | Same as item 12 | Nil | 1 Dec |

## Rules

- **Run length:** if item 6 shows a run needs more than 6 GPU-hours, reduce environment steps per run or use the rented RTX 4090 reserve.
- **Churn:** if fewer than 10% of edges change within 15 steps, lower C or raise robot speed within physically plausible values. Fix both before training any method.
- **Learning:** if IDreamer does not beat a random policy by 22 Nov, stop items 7 to 9, debug against the DMC task from item 2, and drop item 12.
- **Freeze:** no training runs start after 1 Dec, 12:00. Report only finished runs.
