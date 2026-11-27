import atexit
import pathlib
import sys
import warnings

import hydra
import torch
from omegaconf import OmegaConf
from torchrl.record.loggers.wandb import WandbLogger

import tools
from buffer import Buffer
from co_imaginer.run import Checkpointer
from dreamer import Dreamer
from envs import make_envs
from trainer import OnlineTrainer

warnings.filterwarnings("ignore")
sys.path.append(str(pathlib.Path(__file__).parent))
# torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")


@hydra.main(version_base=None, config_path="configs", config_name="configs")
def main(config):
    tools.set_seed_everywhere(config.seed)
    if config.deterministic_run:
        tools.enable_deterministic_run()
    logdir = pathlib.Path(config.logdir).expanduser()
    logdir.mkdir(parents=True, exist_ok=True)

    # Mirror stdout/stderr to a file under logdir while keeping console output.
    console_f = tools.setup_console_log(logdir, filename="console.log")
    atexit.register(lambda: console_f.close())

    print("Logdir", logdir)

    method = "idreamer" if "n_agents" in config.env else "dreamerv3"
    # Name CoDreamer runs separately so their logs and W&B IDs never mix with IDreamer.
    if config.model.comm.enabled:
        method = "codreamer"
    if config.model.comm.enabled and config.model.comm.graph_mode == "rebuilt":
        method = "codreamer_rebuilt"
    agents = f"_n{config.env.n_agents}" if "n_agents" in config.env else ""
    run_name = f"{method}_{config.env.task}{agents}_s{config.seed}"
    writer = WandbLogger(
        exp_name=run_name,
        save_dir=str(logdir),
        id=run_name,
        project=config.wandb_project,
        config=OmegaConf.to_container(config, resolve=True),
    )
    logger = tools.Logger(logdir, writer)
    checkpointer = Checkpointer(logdir, config.checkpoint_minutes)

    replay_buffer = Buffer(config.buffer)

    print("Create envs.")
    train_envs, eval_envs, obs_space, act_space = make_envs(config.env)

    print("Simulate agent.")
    agent = Dreamer(
        config.model,
        obs_space,
        act_space,
    ).to(config.device)

    policy_trainer = OnlineTrainer(config.trainer, replay_buffer, logger, logdir, train_envs, eval_envs, checkpointer)
    policy_trainer.begin(agent)

    items_to_save = {
        "agent_state_dict": agent.state_dict(),
        "optims_state_dict": tools.recursively_collect_optim_state_dict(agent),
    }
    torch.save(items_to_save, logdir / "latest.pt")
    writer.experiment.finish()


if __name__ == "__main__":
    main()
