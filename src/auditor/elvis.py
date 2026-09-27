"""ELVIS-style UCB λ-return baseline (arXiv 2605.04709, Eqs. 14-17; execution_final.md §3.7)."""
import torch


class RunningNorm:
	"""EMA mean/variance of UCB over planning calls, mapped to [0, 1] via a clipped z-score."""

	def __init__(self, decay: float = 0.99, clip: float = 3.0, eps: float = 1e-6):
		self.decay, self.clip, self.eps = decay, clip, eps
		self.mean = None
		self.var = None

	def update(self, x: torch.Tensor) -> None:
		m, v = x.mean().detach(), x.var(unbiased=False).detach()
		if self.mean is None:
			self.mean, self.var = m, v
		else:
			self.mean = self.decay * self.mean + (1 - self.decay) * m
			self.var = self.decay * self.var + (1 - self.decay) * v

	def __call__(self, x: torch.Tensor) -> torch.Tensor:
		if self.mean is None:
			self.update(x)
		u = ((x - self.mean) / (self.var.sqrt() + self.eps)).clamp(-self.clip, self.clip)
		return (u + self.clip) / (2 * self.clip)


def elvis_lambdas(ucb: torch.Tensor, norm: RunningNorm, lambda_min: float, lambda_max: float) -> torch.Tensor:
	"""lambda_t = lambda_max - (lambda_max - lambda_min) * norm(UCB_t)."""
	return lambda_max - (lambda_max - lambda_min) * norm(ucb)


def elvis_return(r_hat: torch.Tensor, v_mean: torch.Tensor, lam: torch.Tensor, discount: float) -> torch.Tensor:
	"""G_H = mu_H;  G_t = r_t + gamma ((1 - lambda_t) mu_{t+1} + lambda_t G_{t+1});  score = G_0.

	r_hat: [H, N, 1]; v_mean: [H+1, N, 1] (mu_0..mu_H); lam: [H+1, N, 1] or [H, N, 1] (lambda_t for t < H).
	"""
	G = v_mean[-1]
	for t in reversed(range(r_hat.shape[0])):
		G = r_hat[t] + discount * ((1 - lam[t]) * v_mean[t + 1] + lam[t] * G)
	return G
