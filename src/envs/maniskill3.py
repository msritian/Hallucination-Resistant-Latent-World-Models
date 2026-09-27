"""ManiSkill3 tasks with the interface TD-MPC2's trainer expects (execution_final.md §4.2).

State observations, end-effector delta-pose control at the native 20 Hz (no action repeat), normalized dense reward,
non-episodic: every episode runs to the time limit and ``terminated`` is always 0. ``success`` is success at any step.
"""
from collections import defaultdict

import gymnasium as gym
import numpy as np
import torch

from src.envs.lavapipe import register_lavapipe

TASKS = ["PushCube-v1", "PickSingleYCB-v1", "PegInsertionSide-v1", "StackCube-v1", "PickCube-v1"]
# Tasks whose single-env default rebuilds the scene at every reset (new object / new peg shape).
REBUILDING_TASKS = {"PickSingleYCB-v1", "PegInsertionSide-v1"}


class ManiSkill3Env:
	def __init__(self, task: str, seed: int, render: bool = False, reconfiguration_freq=None):
		"""``reconfiguration_freq``: rebuild the scene (new YCB object / new peg shape) every N resets.
		None keeps ManiSkill's default (every reset for single-env PickSingleYCB and PegInsertionSide), which is
		used for evaluation; training uses a larger N because each rebuild costs ~0.5 s on the CPU renderer.
		Only applied to REBUILDING_TASKS; other tasks keep ManiSkill's default (no rebuilds)."""
		import mani_skill.envs  # noqa: F401  (registers tasks)
		from mani_skill.utils.registration import REGISTERED_ENVS

		assert task in TASKS, f"Unknown task {task}; expected one of {TASKS}"
		self.task = task
		self.env = gym.make(
			task,
			num_envs=1,
			obs_mode="state",
			control_mode="pd_ee_delta_pose",
			reward_mode="normalized_dense",
			sim_backend="cpu",
			render_backend=register_lavapipe(),
			render_mode="rgb_array" if render else None,
			**({} if reconfiguration_freq is None or task not in REBUILDING_TASKS
			   else {"reconfiguration_freq": reconfiguration_freq}),
		)
		self.max_episode_steps = REGISTERED_ENVS[task].max_episode_steps
		obs_dim = self.env.observation_space.shape[-1]
		self.observation_space = gym.spaces.Box(-np.inf, np.inf, (obs_dim,), np.float32)
		low, high = self.env.action_space.low, self.env.action_space.high
		self.action_space = gym.spaces.Box(
			np.full(low.shape[-1:], low.min(), np.float32), np.full(high.shape[-1:], high.max(), np.float32), dtype=np.float32
		)
		self.action_space.seed(seed)
		self._seed = seed
		self._reset_count = 0
		self._t = 0
		self._success = False

	@staticmethod
	def _obs(obs) -> torch.Tensor:
		return torch.as_tensor(obs).reshape(-1).float().cpu()

	def rand_act(self) -> torch.Tensor:
		return torch.from_numpy(self.action_space.sample().astype(np.float32))

	def reset(self, seed=None) -> torch.Tensor:
		if seed is None and self._reset_count == 0:
			seed = self._seed
		obs, _ = self.env.reset(seed=seed)
		self._reset_count += 1
		self._t = 0
		self._success = False
		return self._obs(obs)

	def step(self, action: torch.Tensor):
		obs, reward, _, _, info = self.env.step(action.detach().cpu().numpy().reshape(1, -1))
		self._t += 1
		self._success = self._success or bool(torch.as_tensor(info["success"]).any())
		out = defaultdict(float)
		out["success"] = float(self._success)
		out["terminated"] = torch.tensor(0.0)
		done = self._t >= self.max_episode_steps
		return self._obs(obs), torch.tensor(float(torch.as_tensor(reward).reshape(-1)[0]), dtype=torch.float32), done, out

	def render(self) -> np.ndarray:
		frame = torch.as_tensor(self.env.render())
		return frame.reshape(frame.shape[-3:]).cpu().numpy().astype(np.uint8)

	def close(self):
		self.env.close()
