# Churn check (PLAN item 3)

## Decision
- Communication range C = 0.5 m.
- Robot speed: VMAS default actuation (`u_multiplier` 1.0, `u_range` 1.0, drag 0.25, dt 0.1 s). Robots reach about 0.3 m/s at full thrust, in a 2 m by 2 m arena with 0.05 m radius robots.
- Both stay fixed for every method and seed.

Edge churn within 15 steps is 17 to 26 percent at this setting across all four cases (4 and 8 robots, random and scripted), so the 10 percent rule from PLAN.md does not trigger. No change to C or speed is needed beyond this choice.

## Why C = 0.5 m and not the Discovery default of 0.35 m
- Discovery ties its comms range to the lidar range, 0.35 m. At that range a random 4 robot team averages 0.19 neighbours per robot, so the graph is almost always empty and message passing has little to do.
- C = 0.5 m doubles the random policy mean degree (0.19 to 0.40 for 4 robots, 0.46 to 0.94 for 8 robots), raises it for the scripted policy (1.09 to 1.31, 1.90 to 2.46) and still keeps churn well above 10 percent.
- At C = 0.75 m and above, churn for 4 robots under the slower settings falls toward or below 10 percent (8 percent at half speed), which weakens the case for rebuilding the graph.
- 0.3 m/s is close to small research ground robots such as TurtleBot3 Burger (0.22 m/s) and Waffle Pi (0.26 m/s), so no speed change is required.

## Effect on the frozen graph baseline
A graph frozen at step t, as in CoDreamer, scores these graph F1 values against the true graph at C = 0.5 m and default speed:

| Robots | Policy | F1 at 15 steps | F1 at 30 steps |
|---|---|---|---|
| 4 | random | 0.84 | 0.76 |
| 4 | scripted | 0.89 | 0.81 |
| 8 | random | 0.83 | 0.75 |
| 8 | scripted | 0.87 | 0.78 |

A rebuilt graph has room to improve on this by up to 11 to 17 F1 points at the imagination horizon.

## Method
- Script: `scripts/churn_check.py`, raw results in `documentation/churn_check.json`.
- VMAS 1.5.2 Discovery, default scenario arguments, continuous actions, CPU. 256 parallel environments, 200 steps each, seed 0.
- Random policy: uniform actions in [-1, 1] per step. Scripted policy: the `HeuristicPolicy` shipped with the Discovery scenario.
- Speed is set by `u_multiplier` (0.5, 1.0, 2.0). Mean speed is the average robot speed over all steps.
- Edges are undirected pairs with distance at most C. For every window from t to t + k, with k of 15 or 30:
  - edge churn = size of the symmetric difference of the two edge sets divided by the size of their union, averaged over windows with at least one edge.
  - frozen graph F1 = 2 times the shared edges divided by the sum of both edge counts, set to 1 when both graphs are empty.
- Rerun: `uv run python scripts/churn_check.py --ranges 0.25 0.35 0.5 0.75 1.0 --speeds 0.5 1.0 2.0 --n-envs 256 --n-steps 200 --out documentation/churn_check.json`. Takes about 1 minute on CPU.

## Full results
| u_multiplier | Mean speed (m/s) | C (m) | Robots | Policy | Mean degree | Churn 15 (%) | Churn 30 (%) | Frozen F1 15 | Frozen F1 30 |
|---|---|---|---|---|---|---|---|---|---|
| 0.5 | 0.06 | 0.25 | 4 | random | 0.07 | 41 | 55 | 0.93 | 0.90 |
| 0.5 | 0.12 | 0.25 | 4 | scripted | 0.63 | 18 | 30 | 0.89 | 0.81 |
| 0.5 | 0.06 | 0.25 | 8 | random | 0.17 | 38 | 51 | 0.78 | 0.69 |
| 0.5 | 0.16 | 0.25 | 8 | scripted | 1.11 | 20 | 31 | 0.86 | 0.78 |
| 0.5 | 0.06 | 0.35 | 4 | random | 0.19 | 25 | 35 | 0.91 | 0.86 |
| 0.5 | 0.12 | 0.35 | 4 | scripted | 0.78 | 16 | 27 | 0.90 | 0.82 |
| 0.5 | 0.06 | 0.35 | 8 | random | 0.42 | 23 | 33 | 0.84 | 0.76 |
| 0.5 | 0.16 | 0.35 | 8 | scripted | 1.47 | 18 | 28 | 0.88 | 0.81 |
| 0.5 | 0.06 | 0.5 | 4 | random | 0.41 | 15 | 22 | 0.91 | 0.86 |
| 0.5 | 0.12 | 0.5 | 4 | scripted | 0.99 | 14 | 23 | 0.90 | 0.83 |
| 0.5 | 0.06 | 0.5 | 8 | random | 0.93 | 15 | 22 | 0.90 | 0.86 |
| 0.5 | 0.16 | 0.5 | 8 | scripted | 1.93 | 16 | 27 | 0.90 | 0.83 |
| 0.5 | 0.06 | 0.75 | 4 | random | 0.83 | 8 | 12 | 0.94 | 0.91 |
| 0.5 | 0.12 | 0.75 | 4 | scripted | 1.36 | 11 | 19 | 0.93 | 0.87 |
| 0.5 | 0.06 | 0.75 | 8 | random | 2.02 | 9 | 13 | 0.95 | 0.93 |
| 0.5 | 0.16 | 0.75 | 8 | scripted | 2.75 | 13 | 22 | 0.93 | 0.87 |
| 0.5 | 0.06 | 1.0 | 4 | random | 1.39 | 6 | 9 | 0.96 | 0.94 |
| 0.5 | 0.12 | 1.0 | 4 | scripted | 1.80 | 9 | 16 | 0.94 | 0.90 |
| 0.5 | 0.06 | 1.0 | 8 | random | 3.23 | 6 | 9 | 0.97 | 0.95 |
| 0.5 | 0.16 | 1.0 | 8 | scripted | 3.68 | 10 | 17 | 0.94 | 0.90 |
| 1.0 | 0.11 | 0.25 | 4 | random | 0.09 | 56 | 73 | 0.88 | 0.82 |
| 1.0 | 0.24 | 0.25 | 4 | scripted | 0.90 | 24 | 38 | 0.85 | 0.76 |
| 1.0 | 0.11 | 0.25 | 8 | random | 0.21 | 55 | 69 | 0.63 | 0.50 |
| 1.0 | 0.31 | 0.25 | 8 | scripted | 1.44 | 25 | 37 | 0.83 | 0.73 |
| 1.0 | 0.11 | 0.35 | 4 | random | 0.19 | 40 | 55 | 0.84 | 0.77 |
| 1.0 | 0.24 | 0.35 | 4 | scripted | 1.09 | 20 | 33 | 0.87 | 0.79 |
| 1.0 | 0.11 | 0.35 | 8 | random | 0.46 | 38 | 51 | 0.71 | 0.59 |
| 1.0 | 0.31 | 0.35 | 8 | scripted | 1.90 | 22 | 33 | 0.85 | 0.77 |
| 1.0 | 0.11 | 0.5 | 4 | random | 0.40 | 26 | 36 | 0.84 | 0.76 |
| 1.0 | 0.24 | 0.5 | 4 | scripted | 1.31 | 17 | 27 | 0.89 | 0.81 |
| 1.0 | 0.11 | 0.5 | 8 | random | 0.94 | 26 | 36 | 0.83 | 0.75 |
| 1.0 | 0.31 | 0.5 | 8 | scripted | 2.46 | 21 | 32 | 0.87 | 0.78 |
| 1.0 | 0.11 | 0.75 | 4 | random | 0.83 | 16 | 23 | 0.88 | 0.83 |
| 1.0 | 0.24 | 0.75 | 4 | scripted | 1.61 | 13 | 23 | 0.91 | 0.84 |
| 1.0 | 0.11 | 0.75 | 8 | random | 2.02 | 17 | 24 | 0.90 | 0.86 |
| 1.0 | 0.31 | 0.75 | 8 | scripted | 3.26 | 17 | 29 | 0.90 | 0.81 |
| 1.0 | 0.11 | 1.0 | 4 | random | 1.38 | 12 | 17 | 0.92 | 0.89 |
| 1.0 | 0.24 | 1.0 | 4 | scripted | 2.00 | 11 | 18 | 0.93 | 0.87 |
| 1.0 | 0.11 | 1.0 | 8 | random | 3.21 | 11 | 16 | 0.94 | 0.91 |
| 1.0 | 0.31 | 1.0 | 8 | scripted | 4.09 | 14 | 23 | 0.92 | 0.86 |
| 2.0 | 0.22 | 0.25 | 4 | random | 0.09 | 78 | 88 | 0.79 | 0.74 |
| 2.0 | 0.43 | 0.25 | 4 | scripted | 0.75 | 37 | 50 | 0.78 | 0.69 |
| 2.0 | 0.22 | 0.25 | 8 | random | 0.23 | 77 | 86 | 0.41 | 0.31 |
| 2.0 | 0.60 | 0.25 | 8 | scripted | 1.47 | 34 | 47 | 0.75 | 0.65 |
| 2.0 | 0.22 | 0.35 | 4 | random | 0.19 | 61 | 76 | 0.72 | 0.62 |
| 2.0 | 0.43 | 0.35 | 4 | scripted | 0.97 | 29 | 42 | 0.82 | 0.74 |
| 2.0 | 0.22 | 0.35 | 8 | random | 0.47 | 60 | 74 | 0.51 | 0.35 |
| 2.0 | 0.60 | 0.35 | 8 | scripted | 2.02 | 29 | 40 | 0.81 | 0.71 |
| 2.0 | 0.22 | 0.5 | 4 | random | 0.39 | 44 | 58 | 0.71 | 0.59 |
| 2.0 | 0.43 | 0.5 | 4 | scripted | 1.23 | 24 | 35 | 0.84 | 0.76 |
| 2.0 | 0.22 | 0.5 | 8 | random | 0.95 | 44 | 58 | 0.68 | 0.55 |
| 2.0 | 0.60 | 0.5 | 8 | scripted | 2.73 | 29 | 39 | 0.81 | 0.73 |
| 2.0 | 0.22 | 0.75 | 4 | random | 0.82 | 29 | 39 | 0.78 | 0.69 |
| 2.0 | 0.43 | 0.75 | 4 | scripted | 1.60 | 21 | 33 | 0.86 | 0.77 |
| 2.0 | 0.22 | 0.75 | 8 | random | 1.96 | 29 | 40 | 0.82 | 0.73 |
| 2.0 | 0.60 | 0.75 | 8 | scripted | 3.68 | 28 | 41 | 0.83 | 0.72 |
| 2.0 | 0.22 | 1.0 | 4 | random | 1.33 | 20 | 29 | 0.86 | 0.80 |
| 2.0 | 0.43 | 1.0 | 4 | scripted | 1.99 | 16 | 25 | 0.89 | 0.82 |
| 2.0 | 0.22 | 1.0 | 8 | random | 3.12 | 21 | 29 | 0.88 | 0.82 |
| 2.0 | 0.60 | 1.0 | 8 | scripted | 4.53 | 21 | 32 | 0.87 | 0.79 |
