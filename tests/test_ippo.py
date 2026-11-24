import math

import pytest

from co_imaginer.ippo import (
    discovery_task_config,
    make_experiment,
    random_episode_return,
)

# Tests for co_imaginer/ippo.py. They run on the CPU and train nothing.


# Only the robot count changes from BenchMARL's Discovery defaults.
@pytest.mark.parametrize("n_agents", [4, 8])
def test_discovery_task_config(n_agents):
    config = discovery_task_config(n_agents)
    assert config["n_agents"] == n_agents
    assert config["max_steps"] == 100


# The experiment builds and has one policy entry per robot.
@pytest.mark.parametrize("n_agents", [4])
def test_make_experiment(tmp_path, n_agents):
    experiment = make_experiment(n_agents, 0, 6000, tmp_path / "run", "cpu")
    assert len(experiment.group_map["agents"]) == n_agents


# Random return is a finite, non-negative number (Discovery rewards are >= 0).
@pytest.mark.parametrize("n_agents", [4])
def test_random_episode_return(n_agents):
    value = random_episode_return(n_agents, episodes=4, seed=0)
    assert math.isfinite(value)
    assert value >= 0
