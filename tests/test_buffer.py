import pytest
import torch
from buffer import Buffer
from omegaconf import OmegaConf
from tensordict import TensorDict

# Tests for the world-aligned replay sampler added to
# third_party/r2dreamer/buffer.py for CoDreamer.

N_ROWS = 8
N_AGENTS = 4


def _filled_buffer(steps, max_size, online=False):
    config = OmegaConf.create(
        {
            "device": "cpu",
            "storage_device": "cpu",
            "batch_size": 4,
            "batch_length": 3,
            "max_size": max_size,
            "online": online,
            "update_latents": False,
            "aligned": True,
            "n_agents": N_AGENTS,
        }
    )
    buffer = Buffer(config)
    for step in range(steps):
        # Each field records where it came from: its time step and robot row.
        buffer.add_transition(
            TensorDict(
                {
                    "time": torch.full((N_ROWS, 1), step),
                    "row": torch.arange(N_ROWS)[:, None],
                    "action": torch.zeros(N_ROWS, 2),
                    "stoch": torch.zeros(N_ROWS, 2, 2),
                    "deter": torch.zeros(N_ROWS, 3),
                    "episode": torch.arange(N_ROWS),
                },
                batch_size=(N_ROWS,),
            )
        )
    return buffer


# Not yet full, exactly full, and wrapped around the circle several times.
@pytest.mark.parametrize(("steps", "max_size"), [(10, 400), (25, 200), (90, 200)])
def test_sample_worlds(steps, max_size):
    buffer = _filled_buffer(steps, max_size)
    for _ in range(20):
        data, index, _ = buffer._sample_worlds()
        time = data["time"][..., 0].view(-1, N_AGENTS, 3)
        row = data["row"][..., 0].view(-1, N_AGENTS, 3)
        # All robots of a sampled world share the same time steps.
        assert (time == time[:, :1]).all()
        # Time runs forward one step at a time, never across the write cursor.
        assert (time[..., 1:] - time[..., :-1] == 1).all()
        # The robots are one complete world, in order.
        assert (row[:, :, 0] % N_AGENTS == torch.arange(N_AGENTS)).all()
        assert (row[:, :, :1] == row).all()
        assert index[0].shape == (4, 3)


@pytest.mark.parametrize(
    ("envs", "expected"),
    [([0, 1, 2, 3], True), ([4, 5, 6, 7, 0, 1, 2, 3], True), ([1, 2, 3, 4], False)],
)
def test_worlds_intact(envs, expected):
    buffer = _filled_buffer(1, 400)
    assert buffer._worlds_intact(torch.tensor(envs)) == expected
