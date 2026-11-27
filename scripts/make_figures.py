import argparse
import csv
import json
import re
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf

METHODS = {
    "ippo": "IPPO",
    "idreamer": "IDreamer",
    "codreamer": "Frozen, perfect",
    "codreamer_rebuilt": "Rebuilt, perfect",
    "codreamer_realistic": "Frozen, realistic",
    "codreamer_rebuilt_realistic": "Rebuilt, realistic",
}
COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300")
MARKERS = ("o", "s", "^", "D", "v", "P")
RUN_NAME = re.compile(r"^(?P<method>.+?)_vmas_\w+?_n(?P<agents>\d+)_s(?P<seed>\d+)$")
SCORE_KEY = "episode/score"
GPU_MEMORY_KEY = "system.gpu.0.memoryAllocatedBytes"
HORIZONS = ("15", "30", "50")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, nargs="+", default=[Path("runs")])
    parser.add_argument(
        "--wandb", help="entity/project, e.g. uowaterloo-wmmal/co-imaginer"
    )
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--links", default="perfect")
    parser.add_argument("--bins", type=int, default=40)
    parser.add_argument("--output", type=Path, default=Path("runs") / "figures")
    return parser.parse_args(argv)


def method_name(config) -> str:
    comm = config.model.get("comm", {})
    if not comm.get("enabled", False):
        return "idreamer"
    method = "codreamer"
    if comm.get("graph_mode", "frozen") == "rebuilt":
        method += "_rebuilt"
    if comm.get("links", "perfect") == "realistic":
        method += "_realistic"
    return method


def run_key(name: str) -> tuple[str, int, int] | None:
    match = RUN_NAME.match(name)
    if match is None:
        return None
    return match["method"], int(match["agents"]), int(match["seed"])


def config_key(run_dir: Path) -> tuple[str, int, int]:
    config = OmegaConf.load(run_dir / ".hydra" / "config.yaml")
    return method_name(config), int(config.env.n_agents), int(config.seed)


def ippo_runs(root: Path) -> tuple[dict, dict]:
    evaluations, costs = {}, {}
    for path in sorted(root.rglob("summary.json")):
        summary = json.loads(path.read_text())
        if "trained_policy" not in summary:
            continue
        key = ("ippo", int(summary["task"]["n_agents"]), int(summary["seed"]))
        evaluations[key] = {
            "run_dir": str(path.parent),
            "return_mean": summary["trained_policy"]["mean"],
            "collisions_mean": float("nan"),
            "peak_gpu_memory_gb": float("nan"),
        }
        costs[key] = {
            "runtime_hours": summary["wall_clock_s"] / 3600,
            "peak_gpu_memory_gb": summary["peak_gpu_mem_mb"] * 2**20 / 1e9,
        }
    return evaluations, costs


def local_runs(roots: list[Path], links: str) -> tuple[dict, dict, dict]:
    curves, evaluations, costs = {}, {}, {}
    for root in roots:
        ippo_evaluations, ippo_costs = ippo_runs(root)
        evaluations.update(ippo_evaluations)
        costs.update(ippo_costs)
        for config_path in sorted(root.rglob(".hydra/config.yaml")):
            run_dir = config_path.parent.parent
            key = config_key(run_dir)
            metrics = run_dir / "metrics.jsonl"
            if metrics.exists():
                rows = [json.loads(line) for line in metrics.read_text().splitlines()]
                points = [(r["step"], r[SCORE_KEY]) for r in rows if SCORE_KEY in r]
                if points:
                    curves[key] = {"points": points, "source": str(run_dir)}
            evaluation = run_dir / f"evaluation_{links}.json"
            if evaluation.exists():
                evaluations[key] = json.loads(evaluation.read_text())
    return curves, evaluations, costs


def wandb_runs(path: str, cache: Path, refresh: bool) -> tuple[dict, dict]:
    if cache.exists() and not refresh:
        stored = json.loads(cache.read_text())
    else:
        import wandb

        stored = []
        for run in wandb.Api().runs(path):
            if run_key(run.name) is None:
                continue
            points = [
                (row["_step"], row[SCORE_KEY])
                for row in run.scan_history(keys=["_step", SCORE_KEY])
            ]
            system = run.history(stream="system", pandas=False)
            memory = [row[GPU_MEMORY_KEY] for row in system if GPU_MEMORY_KEY in row]
            stored.append(
                {
                    "name": run.name,
                    "state": run.state,
                    "points": points,
                    "runtime_hours": run.summary.get("_runtime", float("nan")) / 3600,
                    "peak_gpu_memory_gb": max(memory) / 1e9 if memory else float("nan"),
                    "gpu": run.metadata.get("gpu", "") if run.metadata else "",
                }
            )
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(stored))
    curves, costs = {}, {}
    for run in stored:
        key = run_key(run["name"])
        curves[key] = {"points": run["points"], "source": run["name"]}
        costs[key] = run
    return curves, costs


def binned_curve(points: list, edges: np.ndarray) -> np.ndarray:
    steps, scores = np.asarray(points, dtype=float).T
    index = np.digitize(steps, edges) - 1
    curve = np.full(len(edges) - 1, np.nan)
    for b in range(len(curve)):
        if (index == b).any():
            curve[b] = scores[index == b].mean()
    centres = (edges[:-1] + edges[1:]) / 2
    seen = np.isfinite(curve)
    return np.interp(centres, centres[seen], curve[seen], left=np.nan, right=np.nan)


def mean_std(values: list[float]) -> tuple[float, float, int]:
    array = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if array.size == 0:
        return float("nan"), float("nan"), 0
    return float(array.mean()), float(array.std()), int(array.size)


def by_group(keyed: dict) -> dict:
    groups = defaultdict(dict)
    for (method, agents, seed), value in keyed.items():
        groups[(method, agents)][seed] = value
    return groups


def style(method: str) -> dict:
    index = list(METHODS).index(method) if method in METHODS else len(METHODS) - 1
    return {
        "color": COLORS[index],
        "marker": MARKERS[index],
        "label": METHODS.get(method, method),
    }


def seeds_label(label: str, count: int) -> str:
    return f"{label} ({count} seed{'' if count == 1 else 's'})"


def ordered(methods) -> list[str]:
    known = [m for m in METHODS if m in methods]
    return known + sorted(set(methods) - set(METHODS))


def setup_axes(axes) -> None:
    axes.spines[["top", "right"]].set_visible(False)
    axes.grid(axis="y", color="#e4e4e1", linewidth=0.6)
    axes.set_axisbelow(True)


def plot_learning_curves(curves: dict, bins: int, output: Path) -> Path | None:
    groups = by_group(curves)
    agent_counts = sorted({agents for _, agents in groups})
    if not agent_counts:
        return None
    figure, axes_list = plt.subplots(
        1, len(agent_counts), figsize=(3.5 * len(agent_counts), 2.6), squeeze=False
    )
    for axes, agents in zip(axes_list[0], agent_counts):
        methods = ordered({m for m, a in groups if a == agents})
        last_step = max(
            max(p[0] for p in curve["points"])
            for (m, a), seeds in groups.items()
            if a == agents
            for curve in seeds.values()
        )
        edges = np.linspace(0, last_step + 1, bins + 1)
        centres = (edges[:-1] + edges[1:]) / 2
        for method in methods:
            seeds = groups[(method, agents)]
            stacked = np.stack(
                [binned_curve(c["points"], edges) for c in seeds.values()]
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                mean = np.nanmean(stacked, 0)
                std = np.nanstd(stacked, 0)
            look = style(method)
            label = seeds_label(look["label"], len(seeds))
            axes.plot(centres, mean, color=look["color"], linewidth=1.5, label=label)
            axes.fill_between(
                centres, mean - std, mean + std, color=look["color"], alpha=0.15, lw=0
            )
        setup_axes(axes)
        axes.set_title(f"{agents} robots", fontsize=9)
        axes.set_xlim(left=0)
        axes.set_xlabel("Environment steps", fontsize=8)
        axes.set_ylabel("Episode return", fontsize=8)
        axes.tick_params(labelsize=7)
        axes.legend(fontsize=6, frameon=False)
    figure.tight_layout()
    path = output / "learning_curves.pdf"
    figure.savefig(path)
    figure.savefig(path.with_suffix(".png"), dpi=200)
    plt.close(figure)
    return path


def plot_graph_f1(evaluations: dict, output: Path) -> Path | None:
    groups = by_group(evaluations)
    agent_counts = sorted({agents for _, agents in groups})
    if not agent_counts:
        return None
    horizons = [int(h) for h in HORIZONS]
    figure, axes_list = plt.subplots(
        1, len(agent_counts), figsize=(3.5 * len(agent_counts), 2.6), squeeze=False
    )
    for axes, agents in zip(axes_list[0], agent_counts):
        for method in ordered({m for m, a in groups if a == agents}):
            runs = groups[(method, agents)].values()
            runs = [r for r in runs if "world_model" in r]
            if not runs:
                continue
            look = style(method)
            kept = [[r["frozen"]["graph_f1"][h] for h in HORIZONS] for r in runs]
            axes.plot(
                horizons,
                np.mean(kept, 0),
                color=look["color"],
                linestyle=":",
                linewidth=1.0,
            )
            stats = [
                mean_std([r["world_model"]["graph_f1"][h] for r in runs])
                for h in HORIZONS
            ]
            mean = np.array([s[0] for s in stats])
            std = np.array([s[1] for s in stats])
            axes.errorbar(
                horizons,
                mean,
                yerr=std,
                color=look["color"],
                marker=look["marker"],
                markersize=5,
                linewidth=1.5,
                capsize=2,
                label=seeds_label(look["label"], len(runs)),
            )
        axes.plot([], [], color="#8a8984", linestyle=":", label="Start graph kept")
        setup_axes(axes)
        axes.set_title(f"{agents} robots", fontsize=9)
        axes.set_xticks(horizons)
        axes.set_ylim(0, 1)
        axes.set_xlabel("Imagined steps", fontsize=8)
        axes.set_ylabel("Graph F1", fontsize=8)
        axes.tick_params(labelsize=7)
        axes.legend(fontsize=6, frameon=False)
    figure.tight_layout()
    path = output / "graph_f1_horizon.pdf"
    figure.savefig(path)
    figure.savefig(path.with_suffix(".png"), dpi=200)
    plt.close(figure)
    return path


def cell(values: list[float], digits: int = 2) -> str:
    mean, std, count = mean_std(values)
    if count == 0:
        return "--"
    if count == 1:
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} $\\pm$ {std:.{digits}f}"


def write_table(rows: list[list[str]], header: list[str], path: Path) -> Path:
    with path.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows([[c.replace(" $\\pm$ ", " +- ") for c in r] for r in rows])
    lines = [
        "\\begin{tabular}{l" + "c" * (len(header) - 1) + "}",
        "\\hline",
        " & ".join(header) + " \\\\",
        "\\hline",
        *[" & ".join(r) + " \\\\" for r in rows],
        "\\hline",
        "\\end{tabular}",
    ]
    path.write_text("\n".join(lines) + "\n")
    return path


def results_table(evaluations: dict, output: Path) -> Path | None:
    groups = by_group(evaluations)
    if not groups:
        return None
    header = ["Method", "$n$", "Seeds", "Return", "Collisions"]
    header += [f"F1@{h}" for h in HORIZONS]
    rows = []
    for agents in sorted({a for _, a in groups}):
        for method in ordered({m for m, a in groups if a == agents}):
            runs = list(groups[(method, agents)].values())
            model = [r for r in runs if "world_model" in r]
            rows.append(
                [
                    METHODS.get(method, method),
                    str(agents),
                    str(len(runs)),
                    cell([r["return_mean"] for r in runs]),
                    cell([r["collisions_mean"] for r in runs]),
                    *[
                        cell([r["world_model"]["graph_f1"][h] for r in model])
                        for h in HORIZONS
                    ],
                ]
            )
    return write_table(rows, header, output / "results_table.tex")


def cost_table(costs: dict, evaluations: dict, output: Path) -> Path | None:
    keys = set(costs) | set(evaluations)
    if not keys:
        return None
    header = [
        "Method",
        "$n$",
        "Train hours",
        "Train GPU GB",
        "Imagined steps/s",
        "Eval GPU GB",
    ]
    rows = []
    for agents in sorted({k[1] for k in keys}):
        for method in ordered({k[0] for k in keys if k[1] == agents}):
            train = [v for k, v in costs.items() if k[:2] == (method, agents)]
            evals = [v for k, v in evaluations.items() if k[:2] == (method, agents)]
            rate = [
                r["world_model"]["imagined_steps_per_second"]
                for r in evals
                if "world_model" in r
            ]
            rows.append(
                [
                    METHODS.get(method, method),
                    str(agents),
                    cell([t["runtime_hours"] for t in train]),
                    cell([t["peak_gpu_memory_gb"] for t in train]),
                    cell([r / 1e3 for r in rate], 1) + ("k" if rate else ""),
                    cell([r["peak_gpu_memory_gb"] for r in evals]),
                ]
            )
    return write_table(rows, header, output / "cost_table.tex")


def main(argv: list[str] | None = None) -> list[Path]:
    args = parse_args(argv)
    plt.switch_backend("Agg")
    args.output.mkdir(parents=True, exist_ok=True)
    curves, evaluations, costs = local_runs(args.runs, args.links)
    if args.wandb:
        remote_curves, remote_costs = wandb_runs(
            args.wandb, args.output / "wandb_cache.json", args.refresh
        )
        curves.update(remote_curves)
        costs.update(remote_costs)
    written = [
        plot_learning_curves(curves, args.bins, args.output),
        plot_graph_f1(evaluations, args.output),
        results_table(evaluations, args.output),
        cost_table(costs, evaluations, args.output),
    ]
    written = [p for p in written if p is not None]
    for key in sorted(curves):
        print("curve", key, curves[key]["source"], file=sys.stderr)
    for key in sorted(evaluations):
        print("evaluation", key, evaluations[key]["run_dir"], file=sys.stderr)
    for path in written:
        print(path)
    return written


if __name__ == "__main__":
    main()
