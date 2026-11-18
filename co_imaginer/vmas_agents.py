import gymnasium as gym
import numpy as np
import torch
import vmas
from tensordict import TensorDict


class VmasAgents:
    def __init__(
        self,
        scenario: str,
        num_worlds: int,
        n_agents: int,
        max_steps: int,
        seed: int,
        device: str,
        **scenario_kwargs,
    ) -> None:
        self._env = vmas.make_env(
            scenario,
            num_envs=num_worlds,
            device=device,
            continuous_actions=True,
            seed=seed,
            max_steps=max_steps,
            n_agents=n_agents,
            **scenario_kwargs,
        )
        self._num_worlds = num_worlds
        self._n_agents = n_agents
        self._device = torch.device(device)
        observation_size = self._env.observation_space[0].shape[0]
        action_size = self._env.action_space[0].shape[0]
        self.observation_space = gym.spaces.Dict(
            {"obs": gym.spaces.Box(-np.inf, np.inf, (observation_size,), np.float32)}
        )
        self.action_space = gym.spaces.Box(-1.0, 1.0, (action_size,), np.float32)

    @property
    def env_num(self) -> int:
        return self._num_worlds * self._n_agents

    def step(
        self, action: torch.Tensor, done: torch.Tensor
    ) -> tuple[TensorDict, torch.Tensor]:
        per_world_action = (
            action.to(self._device)
            .clamp(-1.0, 1.0)
            .view(self._num_worlds, self._n_agents, -1)
        )
        should_reset_world = done.to(self._device).view(
            self._num_worlds, self._n_agents
        )[:, 0]
        observations, rewards, is_world_done, _ = self._env.step(
            list(per_world_action.unbind(1))
        )
        for world in should_reset_world.nonzero().flatten().tolist():
            observations = self._env.reset_at(world)
        reward = torch.stack(rewards, 1).masked_fill(should_reset_world[:, None], 0.0)
        is_last = (is_world_done & ~should_reset_world).repeat_interleave(
            self._n_agents
        )
        transition = TensorDict(
            {
                "obs": torch.stack(observations, 1).reshape(self.env_num, -1),
                "reward": reward.reshape(self.env_num, 1),
                "is_first": should_reset_world.repeat_interleave(self._n_agents)[
                    :, None
                ],
                "is_last": is_last[:, None],
                "is_terminal": torch.zeros(
                    self.env_num, 1, dtype=torch.bool, device=self._device
                ),
            },
            batch_size=(self.env_num,),
            device=self._device,
        )
        return transition, is_last
