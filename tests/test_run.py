from contextlib import nullcontext

import pytest
import torch
from co_imaginer.run import Checkpointer, load_run_config
from omegaconf.errors import ConfigKeyError


@pytest.mark.parametrize(
    ("overrides", "expected_n_agents", "expectation"),
    [
        ([], 4, nullcontext()),
        (["n_agents=8"], 8, nullcontext()),
        (["n_robots=8"], None, pytest.raises(ConfigKeyError)),
    ],
)
def test_load_run_config(overrides, expected_n_agents, expectation):
    with expectation:
        assert load_run_config(overrides).n_agents == expected_n_agents


@pytest.mark.parametrize(
    ("interval_minutes", "should_save"),
    [(30, False), (0, True)],
)
def test_maybe_save(tmp_path, interval_minutes, should_save):
    checkpointer = Checkpointer(tmp_path, interval_minutes)
    assert checkpointer.maybe_save({"step": torch.tensor(1)}) == should_save
    assert checkpointer.path.exists() == should_save


@pytest.mark.parametrize(
    ("saved_state", "expected_state"),
    [(None, None), ({"step": torch.tensor(7)}, {"step": torch.tensor(7)})],
)
def test_load(tmp_path, saved_state, expected_state):
    checkpointer = Checkpointer(tmp_path, 30)
    if saved_state is not None:
        checkpointer.save(saved_state)
    assert Checkpointer(tmp_path, 30).load() == expected_state
