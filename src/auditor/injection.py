"""Controlled hallucination injection for the Phase 2 preflight (execution_final.md §6, Phase 2).

All injected latents stay on TD-MPC2's SimNorm manifold (per-group simplices of size simnorm_dim).
"""
from typing import Optional

import torch


def simnorm(x: torch.Tensor, simnorm_dim: int = 8) -> torch.Tensor:
	shp = x.shape
	return torch.softmax(x.view(*shp[:-1], -1, simnorm_dim), dim=-1).view(shp)


def structured_noise(
	z: torch.Tensor, sigma: float, simnorm_dim: int = 8, generator: Optional[torch.Generator] = None
) -> torch.Tensor:
	"""Type 1: add Gaussian noise in logit space, then re-apply SimNorm."""
	eps = torch.randn(z.shape, generator=generator, device=z.device, dtype=z.dtype) * sigma
	return simnorm(torch.log(z + 1e-8) + eps, simnorm_dim)


def goal_interpolation(z: torch.Tensor, z_goal: torch.Tensor, alpha: float) -> torch.Tensor:
	"""Type 2: (1 - alpha) z + alpha z_goal (convex combinations stay on the simplex)."""
	return (1 - alpha) * z + alpha * z_goal


def value_neutral_match(
	q_target: torch.Tensor,
	z_target: torch.Tensor,
	q_pool: torch.Tensor,
	z_pool: torch.Tensor,
	q_tol: float,
	min_dist: float,
) -> torch.Tensor:
	"""Type 3: for each target, index of a pool latent with |Q diff| < q_tol and distance > min_dist (-1 if none).

	q_target [M], z_target [M, d], q_pool [P], z_pool [P, d]. Among valid matches, the closest in Q is chosen.
	"""
	q_diff = (q_target[:, None] - q_pool[None, :]).abs()                     # [M, P]
	dist = torch.cdist(z_target, z_pool)                                     # [M, P]
	valid = (q_diff < q_tol) & (dist > min_dist)
	q_diff = q_diff.masked_fill(~valid, float("inf"))
	idx = q_diff.argmin(dim=1)
	return torch.where(valid.any(dim=1), idx, torch.full_like(idx, -1))


def injector_at(step: int, fn):
	"""Build an ``inject(t, z)`` hook for audit_rollout that applies ``fn(z)`` only at imagined step ``step``."""
	def inject(t: int, z: torch.Tensor) -> torch.Tensor:
		return fn(z) if t == step else z
	return inject
