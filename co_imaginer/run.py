import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torchrl.record.loggers.wandb import WandbLogger

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "run.yaml"
CHECKPOINT_NAME = "checkpoint.pt"


def load_run_config(
    overrides: list[str], path: Path = DEFAULT_CONFIG_PATH
) -> DictConfig:
    config = OmegaConf.load(path)
    OmegaConf.set_struct(config, True)
    return OmegaConf.merge(config, OmegaConf.from_dotlist(overrides))


def run_name(config: DictConfig) -> str:
    return (
        f"{config.method}_{config.graph_mode}_{config.link_model}"
        f"_n{config.n_agents}_s{config.seed}"
    )


def make_run_dir(config: DictConfig) -> Path:
    run_dir = Path(config.output_root) / run_name(config)
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def make_logger(config: DictConfig, run_dir: Path) -> WandbLogger:
    name = run_name(config)
    return WandbLogger(
        exp_name=name,
        save_dir=str(run_dir),
        id=name,
        project=config.wandb_project,
        config=OmegaConf.to_container(config),
    )


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def cost_metrics(start_time: float) -> dict[str, float]:
    peak_bytes = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
    return {
        "wall_clock_hours": (time.monotonic() - start_time) / 3600,
        "peak_gpu_memory_gb": peak_bytes / 1e9,
    }


class Checkpointer:
    def __init__(self, run_dir: Path, interval_minutes: float) -> None:
        self.path = run_dir / CHECKPOINT_NAME
        self.interval_seconds = interval_minutes * 60
        self.last_save_time = time.monotonic()

    def maybe_save(self, state: dict) -> bool:
        if time.monotonic() - self.last_save_time < self.interval_seconds:
            return False
        self.save(state)
        return True

    def save(self, state: dict) -> None:
        temporary_path = self.path.with_suffix(".tmp")
        torch.save(state, temporary_path)
        os.replace(temporary_path, self.path)
        self.last_save_time = time.monotonic()

    def load(self) -> dict | None:
        if not self.path.exists():
            return None
        return torch.load(self.path)
