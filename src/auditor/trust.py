"""Trust weights and the audited trust score (execution_final.md §3.5)."""
import math

import torch


def trust_weights(s: torch.Tensor, kappa: float = 1.0) -> torch.Tensor:
	"""w_t = exp(-kappa * max(0, s_t - 1)); kappa = inf gives hard truncation w_t = 1[s_t <= 1]."""
	s = s.nan_to_num(nan=1e6, posinf=1e6, neginf=0.0)
	if math.isinf(kappa):
		return (s <= 1).to(s.dtype)
	return torch.exp(-kappa * (s - 1).clamp(min=0))


def trust_weighted_return(
	r_hat: torch.Tensor, q_sa: torch.Tensor, q_terminal: torch.Tensor, w: torch.Tensor, discount: float
) -> tuple[torch.Tensor, torch.Tensor]:
	"""Forward form of the audited trust score.

	S = sum_t gamma^t [Omega_t r_t + (Omega_{t-1} - Omega_t) Q(z_t, a_t)] + gamma^H Omega_{H-1} Q_terminal

	r_hat, q_sa, w: [H, N, 1]; q_terminal: [N, 1]. Returns (scores [N, 1], Omega [H, N, 1]).
	"""
	H = r_hat.shape[0]
	omega = torch.cumprod(w, dim=0)
	omega_prev = torch.cat([torch.ones_like(omega[:1]), omega[:-1]], dim=0)
	disc = discount ** torch.arange(H, device=r_hat.device, dtype=r_hat.dtype).view(H, 1, 1)
	score = (disc * (omega * r_hat + (omega_prev - omega) * q_sa)).sum(0)
	score = score + discount ** H * omega[-1] * q_terminal
	return score, omega


def trust_weighted_return_backward(
	r_hat: torch.Tensor, q_sa: torch.Tensor, q_terminal: torch.Tensor, w: torch.Tensor, discount: float
) -> torch.Tensor:
	"""Same score as a λ-return with λ_t = w_t: G_t = (1 - w_t) Q(z_t, a_t) + w_t (r_t + gamma G_{t+1})."""
	G = q_terminal
	for t in reversed(range(r_hat.shape[0])):
		G = (1 - w[t]) * q_sa[t] + w[t] * (r_hat[t] + discount * G)
	return G


def effective_horizon(omega: torch.Tensor) -> torch.Tensor:
	"""H_eff = sum_t Omega_t, per candidate: [N, 1]."""
	return omega.sum(0)
