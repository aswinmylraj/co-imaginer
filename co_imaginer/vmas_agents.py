import itertools

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
            terminated_truncated=True,
            **scenario_kwargs,
        )
        self._num_worlds = num_worlds
        self._n_agents = n_agents
        self._device = torch.device(device)
        self._u_range = torch.stack(
            [agent.action.u_range_tensor for agent in self._env.agents]
        ).to(self._device)
        self._pairs = list(itertools.combinations(range(n_agents), 2))
        self._min_collision_distance = getattr(
            self._env.scenario, "min_collision_distance", 0.005
        )
        observation_size = self._env.observation_space[0].shape[0]
        action_size = self._env.action_space[0].shape[0]
        self.observation_space = gym.spaces.Dict(
            {"obs": gym.spaces.Box(-np.inf, np.inf, (observation_size,), np.float32)}
        )
        self.action_space = gym.spaces.Box(-1.0, 1.0, (action_size,), np.float32)

    @property
    def env_num(self) -> int:
        return self._num_worlds * self._n_agents

    def _collisions(self) -> torch.Tensor:
        world = self._env.world
        counts = torch.zeros(self._num_worlds, self._n_agents, device=self._device)
        for i, j in self._pairs:
            distance = world.get_distance(world.agents[i], world.agents[j])
            hit = (distance < self._min_collision_distance).float()
            counts[:, i] += hit
            counts[:, j] += hit
        return counts

    def step(
        self, action: torch.Tensor, done: torch.Tensor
    ) -> tuple[TensorDict, torch.Tensor]:
        per_world_action = (
            action.to(self._device)
            .clamp(-1.0, 1.0)
            .view(self._num_worlds, self._n_agents, -1)
            * self._u_range
        )
        should_reset_world = done.to(self._device).view(
            self._num_worlds, self._n_agents
        )[:, 0]
        observations, rewards, is_terminated, is_truncated, _ = self._env.step(
            list(per_world_action.unbind(1))
        )
        collisions = self._collisions().masked_fill(should_reset_world[:, None], 0.0)
        for world in should_reset_world.nonzero().flatten().tolist():
            observations = self._env.reset_at(world)
        reward = torch.stack(rewards, 1).masked_fill(should_reset_world[:, None], 0.0)
        is_terminal = (is_terminated & ~should_reset_world).repeat_interleave(
            self._n_agents
        )
        is_last = (
            (is_terminated | is_truncated) & ~should_reset_world
        ).repeat_interleave(self._n_agents)
        transition = TensorDict(
            {
                "obs": torch.stack(observations, 1).reshape(self.env_num, -1),
                "reward": reward.reshape(self.env_num, 1),
                "is_first": should_reset_world.repeat_interleave(self._n_agents)[
                    :, None
                ],
                "is_last": is_last[:, None],
                "is_terminal": is_terminal[:, None],
                "log_collisions": collisions.reshape(self.env_num, 1),
            },
            batch_size=(self.env_num,),
            device=self._device,
        )
        return transition, is_last
