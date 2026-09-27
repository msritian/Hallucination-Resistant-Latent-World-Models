"""Bellman-consistency signals on imagined latent rollouts (execution_final.md §3.3).

Works with any model exposing TD-MPC2's WorldModel interface:
	next(z, a, task), reward(z, a, task), pi(z, task) -> (action, info["mean"]),
	Q(z, a, task, return_type='all', detach=...) -> per-head outputs [K, ..., num_bins or 1].
All per-step tensors are shaped [H, N, 1] (steps, candidates, 1).
"""
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import torch

import src.tdmpc2_path  # noqa: F401  (puts TD-MPC2 on sys.path)
from common import math

Injector = Callable[[int, torch.Tensor], torch.Tensor]


@dataclass
class Rollout:
	z: torch.Tensor          # [H+1, N, d]  imagined latents; z[0] is the start latent
	r_hat: torch.Tensor      # [H, N, 1]    decoded predicted rewards
	q_sa: torch.Tensor       # [H, N, 1]    mean-head Q(z_t, a_t)
	v_heads: torch.Tensor    # [K, H+1, N, 1] per-head Q(z_t, mu_pi(z_t)) for t = 0..H
	delta: torch.Tensor      # [H, N, 1]    Signal A: Bellman residual
	discount: float

	@property
	def v_mean(self) -> torch.Tensor:  # [H+1, N, 1]
		return self.v_heads.mean(0)

	@property
	def v_std(self) -> torch.Tensor:  # [H+1, N, 1]
		return self.v_heads.std(0)

	@property
	def q_terminal(self) -> torch.Tensor:  # [N, 1]
		return self.v_mean[-1]

	def critic_spread(self) -> torch.Tensor:
		"""Signal B: std over critic heads of Q(z_{t+1}, mu_pi(z_{t+1})), aligned with delta_t."""
		return self.v_std[1:]

	def ucb(self, beta: float) -> torch.Tensor:
		"""Signal E (ELVIS-style) raw UCB for every state t = 0..H: [H+1, N, 1]."""
		return self.v_mean + beta * self.v_std


def decode(logits: torch.Tensor, cfg) -> torch.Tensor:
	"""TD-MPC2 two-hot logits -> scalars (keeps a trailing dim of 1)."""
	return math.two_hot_inv(logits, cfg)


def q_heads(model, z, a, cfg, task=None, detach=False) -> torch.Tensor:
	"""Decoded per-head Q-values: [K, ..., 1]."""
	return decode(model.Q(z, a, task, return_type="all", detach=detach), cfg)


def pi_mean(model, z, task=None) -> torch.Tensor:
	_, info = model.pi(z, task)
	return info["mean"]


def audit_rollout(
	model,
	cfg,
	z0: torch.Tensor,
	actions: torch.Tensor,
	discount: float,
	task=None,
	differentiable: bool = False,
	inject: Optional[Injector] = None,
) -> Rollout:
	"""Roll the dynamics out along ``actions`` [H, N, A] from ``z0`` [N, d] and compute the residual.

	``differentiable=True`` (source fix) keeps gradients through the latents; Q is evaluated with
	``detach=True`` so its parameters get no gradient, the predicted reward is detached so the reward
	head cannot absorb the residual, and the caller must freeze the policy.
	``inject(t, z)`` may replace the imagined latent z_t (t >= 1) before the rollout continues.
	"""
	ctx = nullcontext() if differentiable else torch.no_grad()
	with ctx:
		H = actions.shape[0]
		zs = [z0]
		for t in range(H):
			z_next = model.next(zs[-1], actions[t], task)
			if inject is not None:
				z_next = inject(t + 1, z_next)
			zs.append(z_next)
		z = torch.stack(zs)                                   # [H+1, N, d]
		r_hat = decode(model.reward(z[:-1], actions, task), cfg)  # [H, N, 1]
		if differentiable:
			r_hat = r_hat.detach()
		q_sa = q_heads(model, z[:-1], actions, cfg, task, detach=differentiable).mean(0)
		v_heads = q_heads(model, z, pi_mean(model, z, task), cfg, task, detach=differentiable)
		delta = (q_sa - (r_hat + discount * v_heads[:, 1:].mean(0))).abs()
	return Rollout(z=z, r_hat=r_hat, q_sa=q_sa, v_heads=v_heads, delta=delta, discount=discount)


def dynamics_disagreement(heads: Sequence[Callable], z: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
	"""Signal D (PETS-style): mean over latent dims of the std across dynamics heads.

	``heads`` are callables (z, a) -> z'; ``z`` is the head-0 rollout [H+1, N, d]. Returns [H, N, 1].
	"""
	with torch.no_grad():
		preds = torch.stack([h(z[:-1], actions) for h in heads])  # [M, H, N, d]
		return preds.std(0).mean(-1, keepdim=True)


def bellman_target_spread(
	heads: Sequence[Callable], model, cfg, rollout: Rollout, actions: torch.Tensor, task=None
) -> torch.Tensor:
	"""Signal M (MOBILE-style): std across dynamics heads of r_hat + gamma * mean-head Q(z'_k, mu_pi(z'_k))."""
	with torch.no_grad():
		targets = []
		for h in heads:
			z_k = h(rollout.z[:-1], actions)
			v_k = q_heads(model, z_k, pi_mean(model, z_k, task), cfg, task).mean(0)
			targets.append(rollout.r_hat + rollout.discount * v_k)
		return torch.stack(targets).std(0)
