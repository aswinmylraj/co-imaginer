# IPPO results (PLAN.md item 4)

Produced by `co_imaginer/ippo.py --agents 4 --seed 0` on 20 Nov 2026, RTX 5060 Ti, 3,000,000 frames, BenchMARL 1.5.2 default IPPO.

- `ippo_n4_s0_eval_return.csv`: BenchMARL evaluation return (32 episodes) at each logged iteration. One iteration is 6,000 frames.
- `ippo_n4_s0_random_baseline.json`: return of a uniform random policy over 1,000 episodes.
- `ippo_n4_s0_return_vs_frames.png`: the two above as a plot.

Result: final evaluation return 1.97 against 0.11 for random. IPPO beats random.

Checkpoints and raw BenchMARL logs are not committed (AGENTS.md).
