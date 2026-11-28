import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from make_figures import (
    binned_curve,
    ippo_runs,
    main,
    mean_std,
    method_name,
    run_key,
    seeds_label,
)


@pytest.mark.parametrize(
    ("comm", "expected"),
    [
        (None, "idreamer"),
        ({"enabled": False, "graph_mode": "rebuilt"}, "idreamer"),
        ({"enabled": True}, "codreamer"),
        ({"enabled": True, "graph_mode": "rebuilt"}, "codreamer_rebuilt"),
        (
            {"enabled": True, "graph_mode": "rebuilt", "links": "realistic"},
            "codreamer_rebuilt_realistic",
        ),
    ],
)
def test_method_name(comm: dict | None, expected: str) -> None:
    model = {} if comm is None else {"comm": comm}
    assert method_name(OmegaConf.create({"model": model})) == expected


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("codreamer_vmas_discovery_n4_s0", ("codreamer", 4, 0)),
        ("codreamer_rebuilt_vmas_discovery_n8_s12", ("codreamer_rebuilt", 8, 12)),
        ("idreamer_vmas_discovery_n4_s2", ("idreamer", 4, 2)),
        ("walker_s1", None),
    ],
)
def test_run_key(name: str, expected: tuple | None) -> None:
    assert run_key(name) == expected


@pytest.mark.parametrize(
    ("points", "expected"),
    [
        ([(0, 1.0), (1, 3.0), (5, 4.0)], [2.0, 4.0]),
        ([(6, 1.0)], [math.nan, 1.0]),
        ([(0, 1.0), (12, 3.0)], [1.0, 2.0, 3.0]),
    ],
)
def test_binned_curve(points: list, expected: list) -> None:
    edges = np.arange(0.0, 5.0 * len(expected) + 1, 5.0)
    curve = binned_curve(points, edges)
    np.testing.assert_allclose(curve, expected)


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ([1.0, 3.0], (2.0, 1.0, 2)),
        ([2.0, math.nan], (2.0, 0.0, 1)),
    ],
)
def test_mean_std(values: list, expected: tuple) -> None:
    assert mean_std(values) == expected


@pytest.mark.parametrize(
    ("count", "expected"),
    [(1, "IPPO (1 seed)"), (3, "IPPO (3 seeds)")],
)
def test_seeds_label(count: int, expected: str) -> None:
    assert seeds_label("IPPO", count) == expected


def write_run(root: Path, comm: dict, seed: int) -> None:
    run_dir = root / f"run_{comm['graph_mode']}_{seed}"
    (run_dir / ".hydra").mkdir(parents=True)
    config = {"seed": seed, "env": {"n_agents": 4}, "model": {"comm": comm}}
    OmegaConf.save(OmegaConf.create(config), run_dir / ".hydra" / "config.yaml")
    rows = [{"step": s, "episode/score": s / 100} for s in range(0, 1000, 50)]
    (run_dir / "metrics.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    f1 = {"15": 0.9, "30": 0.8, "50": 0.7}
    evaluation = {
        "run_dir": str(run_dir),
        "return_mean": 1.0 + seed,
        "collisions_mean": 2.0,
        "peak_gpu_memory_gb": 0.5,
        "frozen": {"graph_f1": f1},
        "world_model": {"graph_f1": f1, "imagined_steps_per_second": 1e5},
    }
    (run_dir / "evaluation_perfect.json").write_text(json.dumps(evaluation))


def write_ippo(root: Path, seed: int) -> None:
    folder = root / "ippo" / f"n4_s{seed}"
    folder.mkdir(parents=True)
    summary = {
        "task": {"n_agents": 4},
        "seed": seed,
        "wall_clock_s": 7200.0,
        "peak_gpu_mem_mb": 1024.0,
        "trained_policy": {"mean": 2.0},
    }
    (folder / "summary.json").write_text(json.dumps(summary))


@pytest.mark.parametrize("ippo_seeds", [0, 1])
def test_ippo_runs(tmp_path: Path, ippo_seeds: int) -> None:
    for seed in range(ippo_seeds):
        write_ippo(tmp_path, seed)
    evaluations, costs = ippo_runs(tmp_path)
    assert sorted(evaluations) == [("ippo", 4, s) for s in range(ippo_seeds)]
    assert all(c["runtime_hours"] == 2.0 for c in costs.values())


@pytest.mark.parametrize("graph_modes", [("frozen",), ("frozen", "rebuilt")])
def test_main(tmp_path: Path, graph_modes: tuple) -> None:
    write_ippo(tmp_path / "runs", 0)
    for graph_mode in graph_modes:
        for seed in range(2):
            write_run(
                tmp_path / "runs", {"enabled": True, "graph_mode": graph_mode}, seed
            )
    written = main(
        ["--runs", str(tmp_path / "runs"), "--output", str(tmp_path / "out")]
    )
    assert {p.name for p in written} == {
        "learning_curves.pdf",
        "graph_f1_horizon.pdf",
        "results_table.tex",
        "cost_table.tex",
    }
    table = (tmp_path / "out" / "results_table.tex").read_text()
    assert "1.50 $\\pm$ 0.50" in table
    assert table.count("\\\\") == len(graph_modes) + 2
    assert "IPPO & 4 & 1 & 2.00 & --" in table
