import itertools

import gymnasium as gym
import numpy as np
import torch
import vmas
from tensordict import TensorDict


class VmasAgentEnv:
    def __init__(
        self,
        scenario: str,
        num_envs: int,
        max_steps: int,
        device: str,
        seed: int,
        scenario_kwargs: dict,
    ) -> None:
        self.device = torch.device(device)
        self._env = vmas.make_env(
            scenario,
            num_envs=num_envs,
            device=self.device,
            continuous_actions=True,
            max_steps=max_steps,
            seed=seed,
            clamp_actions=True,
            terminated_truncated=True,
            **scenario_kwargs,
        )
        self.num_envs = num_envs
        self.n_agents = self._env.n_agents
        self._u_range = torch.stack([agent.action.u_range_tensor for agent in self._env.agents]).to(self.device)
        self._pairs = list(itertools.combinations(range(self.n_agents), 2))
        self._min_collision_distance = getattr(self._env.scenario, "min_collision_distance", 0.005)
        obs_dim = self._env.observation_space[0].shape[0]
        act_dim = self._u_range.shape[-1]
        self.observation_space = gym.spaces.Dict(
            {"vector": gym.spaces.Box(-np.inf, np.inf, (obs_dim,), dtype=np.float32)}
        )
        self.action_space = gym.spaces.Box(-1.0, 1.0, (act_dim,), dtype=np.float32)

    @property
    def env_num(self) -> int:
        return self.num_envs * self.n_agents

    def _per_env(self, done: torch.Tensor) -> torch.Tensor:
        return done.to(self.device).view(self.num_envs, self.n_agents).any(-1)

    def _flatten(self, per_agent: list[torch.Tensor]) -> torch.Tensor:
        return torch.stack(per_agent, dim=1).flatten(0, 1)

    def _expand(self, flag: torch.Tensor) -> torch.Tensor:
        return flag.repeat_interleave(self.n_agents).unsqueeze(-1)

    def _collisions(self) -> torch.Tensor:
        world = self._env.world
        counts = torch.zeros(self.num_envs, self.n_agents, device=self.device)
        for i, j in self._pairs:
            hit = (world.get_distance(world.agents[i], world.agents[j]) < self._min_collision_distance).float()
            counts[:, i] += hit
            counts[:, j] += hit
        return counts.flatten()

    def step(self, action: torch.Tensor, done: torch.Tensor) -> tuple[TensorDict, torch.Tensor]:
        reset = self._per_env(done)
        zeros = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        if reset.all():
            obs = self._env.reset()
            reward = torch.zeros(self.env_num, device=self.device)
            terminated, truncated = zeros, zeros
        else:
            action = action.to(self.device).view(self.num_envs, self.n_agents, -1)
            action = action.clamp(-1.0, 1.0) * self._u_range
            obs, rewards, terminated, truncated, _ = self._env.step(list(action.unbind(1)))
            reward = self._flatten(rewards)
            if reset.any():
                for index in reset.nonzero().flatten().tolist():
                    self._env.reset_at(index, return_observations=False)
                obs = self._env.get_from_scenario(
                    get_observations=True, get_rewards=False, get_infos=False, get_dones=False
                )[0]
                reward = reward * ~reset.repeat_interleave(self.n_agents)
                terminated = terminated & ~reset
                truncated = truncated & ~reset
        is_last = terminated | truncated
        trans = TensorDict(
            {
                "vector": self._flatten(obs).float(),
                "reward": reward.float().unsqueeze(-1),
                "is_first": self._expand(reset),
                "is_last": self._expand(is_last),
                "is_terminal": self._expand(terminated),
                "log_collisions": self._collisions().unsqueeze(-1),
            },
            batch_size=(self.env_num,),
            device=self.device,
        )
        return trans, is_last.repeat_interleave(self.n_agents)
