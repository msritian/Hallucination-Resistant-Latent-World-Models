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
	q_sa_heads: Optional[torch.Tensor] = None  # [K, H, N, 1] per-head Q(z_t, a_t)

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

	def signed_residual(self) -> torch.Tensor:
		"""r_t + gamma V(z_{t+1}) - Q(z_t, a_t): positive when the imagined step promises more than the critic expected."""
		return self.r_hat + self.discount * self.v_mean[1:] - self.q_sa

	def optimism(self) -> torch.Tensor:
		"""Signal Ao: one-sided residual. Only optimistic errors can win the planner's argmax."""
		return self.signed_residual().clamp(min=0)

	def anchored_residual(self) -> torch.Tensor:
		"""Signal Aa: |Q(z_0, a_0) - (sum_{k<=t} gamma^k r_k + gamma^{t+1} V(z_{t+1}))| for t = 0..H-1: [H, N, 1].

		z_0 is the real (encoded) start, where a grounded critic is reliable. The (t+1)-step imagined return must agree
		with it, so small per-step inconsistencies accumulate against a fixed reference instead of being judged one
		step at a time. At t = 0 this equals Signal A.
		"""
		H = self.r_hat.shape[0]
		disc = self.discount ** torch.arange(H + 1, device=self.r_hat.device, dtype=self.r_hat.dtype).view(-1, 1, 1)
		partial = torch.cumsum(disc[:H] * self.r_hat, 0)                     # sum_{k<=t} gamma^k r_k
		imagined = partial + disc[1:] * self.v_mean[1:]                       # + gamma^{t+1} V(z_{t+1})
		return (self.q_sa[:1] - imagined).abs()

	def k_step_residual(self, k: int) -> torch.Tensor:
		"""Signal A{k}: k-step Bellman residual ending at each step t: [H, N, 1].

		|Q(z_s, a_s) - (sum_{j=s}^{t} gamma^{j-s} r_j + gamma^{t-s+1} V(z_{t+1}))| with s = max(0, t - k + 1).
		k = 1 is Signal A; k >= H is the anchored residual (s = 0, the real start).
		"""
		H = self.r_hat.shape[0]
		g = self.discount
		out = []
		for t in range(H):
			s = max(0, t - k + 1)
			disc = g ** torch.arange(t - s + 1, device=self.r_hat.device, dtype=self.r_hat.dtype).view(-1, 1, 1)
			ret = (disc * self.r_hat[s:t + 1]).sum(0) + g ** (t - s + 1) * self.v_mean[t + 1]
			out.append((self.q_sa[s] - ret).abs())
		return torch.stack(out)

	def cumulative_signed_residual(self) -> torch.Tensor:
		"""Signal Ac: |sum_{k<=t} gamma^k (r_k + gamma V(z_{k+1}) - Q(z_k, a_k))|: [H, N, 1].

		Critic noise has a random sign per step and cancels; systematic drift has a consistent sign and accumulates
		(a CUSUM-style drift statistic on the Bellman residual).
		"""
		H = self.r_hat.shape[0]
		disc = self.discount ** torch.arange(H, device=self.r_hat.device, dtype=self.r_hat.dtype).view(-1, 1, 1)
		return torch.cumsum(disc * self.signed_residual(), 0).abs()

	def head_residual_spread(self) -> torch.Tensor:
		"""Signal P: std over critic heads k of r_t + gamma V_k(z_{t+1}) - Q_k(z_t, a_t) (same head on both sides)."""
		per_head = self.r_hat + self.discount * self.v_heads[:, 1:] - self.q_sa_heads
		return per_head.std(0)


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
		q_sa_heads = q_heads(model, z[:-1], actions, cfg, task, detach=differentiable)
		q_sa = q_sa_heads.mean(0)
		v_heads = q_heads(model, z, pi_mean(model, z, task), cfg, task, detach=differentiable)
		delta = (q_sa - (r_hat + discount * v_heads[:, 1:].mean(0))).abs()
	return Rollout(z=z, r_hat=r_hat, q_sa=q_sa, v_heads=v_heads, delta=delta, discount=discount, q_sa_heads=q_sa_heads)


def training_target_residual(model, cfg, roll: Rollout, task=None) -> torch.Tensor:
	"""Signal At: residual against the target the critic was trained toward.

	TD-MPC2 regresses Q(z, a) onto r + gamma * min of two random target heads at (z', pi(z')), so mean-head Q sits
	below r + gamma * mean-head V by a systematic margin. Using the expected pairwise minimum of the target heads
	removes that offset from the residual. Returns [H, N, 1].
	"""
	with torch.no_grad():
		z1 = roll.z[1:]
		v = decode(model.Q(z1, pi_mean(model, z1, task), task, return_type="all", target=True), cfg)  # [K, H, N, 1]
		K = v.shape[0]
		i, j = torch.triu_indices(K, K, offset=1)
		pair_min = torch.minimum(v[i], v[j]).mean(0)
		return (roll.q_sa - (roll.r_hat + roll.discount * pair_min)).abs()


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
