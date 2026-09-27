"""Small CPU stand-ins for TD-MPC2's WorldModel with the same method signatures."""
from types import SimpleNamespace

import torch
import torch.nn as nn
from torch.func import functional_call

from src.auditor.injection import simnorm


def make_cfg(num_bins: int = 101, vmin: float = -10.0, vmax: float = 10.0):
	return SimpleNamespace(num_bins=num_bins, vmin=vmin, vmax=vmax, simnorm_dim=8)


class MockWorldModel(nn.Module):
	"""Linear networks; K critic heads; SimNorm latents. Outputs per-head logits like TD-MPC2."""

	def __init__(self, obs_dim=6, latent_dim=16, action_dim=3, num_q=5, num_bins=101, seed=0):
		super().__init__()
		torch.manual_seed(seed)
		out = max(num_bins, 1)
		self.latent_dim = latent_dim
		self._encoder = nn.Linear(obs_dim, latent_dim)
		self._dynamics = nn.Linear(latent_dim + action_dim, latent_dim)
		self._reward = nn.Linear(latent_dim + action_dim, out)
		self._pi = nn.Linear(latent_dim, 2 * action_dim)
		self._Qs = nn.ModuleList([nn.Linear(latent_dim + action_dim, out) for _ in range(num_q)])
		self.eval()

	def encode(self, obs, task):
		return simnorm(self._encoder(obs))

	def next(self, z, a, task):
		return simnorm(self._dynamics(torch.cat([z, a], -1)))

	def reward(self, z, a, task):
		return self._reward(torch.cat([z, a], -1))

	def pi(self, z, task):
		mean, log_std = self._pi(z).chunk(2, dim=-1)
		action = torch.tanh(mean + torch.randn_like(mean) * log_std.exp())
		return action, {"mean": torch.tanh(mean)}

	def Q(self, z, a, task, return_type="min", target=False, detach=False):
		assert return_type == "all"
		x = torch.cat([z, a], -1)
		outs = []
		for q in self._Qs:
			if detach:
				outs.append(functional_call(q, {k: v.detach() for k, v in q.named_parameters()}, (x,)))
			else:
				outs.append(q(x))
		return torch.stack(outs)


class ConsistentWorldModel(nn.Module):
	"""Raw scalars (num_bins=0) with Q(z, a) = v.z and r(z, a) = v.z - gamma * v.next(z, a): delta == 0 exactly."""

	def __init__(self, latent_dim=16, action_dim=3, num_q=5, gamma=0.95, seed=0):
		super().__init__()
		torch.manual_seed(seed)
		self.gamma, self.num_q = gamma, num_q
		self._dynamics = nn.Linear(latent_dim + action_dim, latent_dim)
		self._pi = nn.Linear(latent_dim, action_dim)
		self.v = nn.Parameter(torch.randn(latent_dim))

	def next(self, z, a, task):
		return simnorm(self._dynamics(torch.cat([z, a], -1)))

	def value(self, z):
		return (z * self.v).sum(-1, keepdim=True)

	def reward(self, z, a, task):
		return self.value(z) - self.gamma * self.value(self.next(z, a, task))

	def pi(self, z, task):
		mean = torch.tanh(self._pi(z))
		return mean, {"mean": mean}

	def Q(self, z, a, task, return_type="all", target=False, detach=False):
		return self.value(z).unsqueeze(0).expand(self.num_q, *z.shape[:-1], 1)
