"""Per-step threshold calibration tau(t) on ground-truth-filtered replays (execution_final.md §3.4)."""
from dataclasses import dataclass

import torch


@dataclass
class ErrorThresholds:
	eps_clean: float
	eps_hall: float


@dataclass
class Calibration:
	tau: torch.Tensor           # [H]
	clean_counts: torch.Tensor  # [H] number of clean replays used at each step
	carried: torch.Tensor       # [H] bool: threshold carried forward from the previous step


def latent_errors(z_hat: torch.Tensor, z_real: torch.Tensor) -> torch.Tensor:
	"""e_{t+1} = ||z_hat_{t+1} - z_real_{t+1}||_2 for t = 0..H-1. Inputs [H+1, M, d] -> [H, M]."""
	return (z_hat[1:] - z_real[1:]).norm(dim=-1)


def error_thresholds(err: torch.Tensor, clean_pct: float = 90, hall_pct: float = 99) -> ErrorThresholds:
	"""Thresholds from one-step errors e_1 = err[0]."""
	e1 = err[0].flatten().float()
	return ErrorThresholds(
		eps_clean=torch.quantile(e1, clean_pct / 100).item(),
		eps_hall=torch.quantile(e1, hall_pct / 100).item(),
	)


def clean_prefix_mask(err: torch.Tensor, eps_clean: float) -> torch.Tensor:
	"""[H, M] bool: replay is clean at step t iff e_{j+1} <= eps_clean for all j <= t."""
	return torch.cumprod((err <= eps_clean).int(), dim=0).bool()


def calibrate(
	signal: torch.Tensor,
	err: torch.Tensor,
	eps_clean: float,
	tau_pct: float = 95,
	min_clean_samples: int = 200,
	floor: float = 1e-6,
) -> Calibration:
	"""tau(t) = P_{tau_pct}(signal_t | clean at t), carried forward when fewer than min_clean_samples.

	signal, err: [H, M]. At t = 0 with too few clean samples, the unfiltered percentile is used.
	"""
	H = signal.shape[0]
	mask = clean_prefix_mask(err, eps_clean)
	tau = torch.empty(H)
	counts = mask.sum(1)
	carried = torch.zeros(H, dtype=torch.bool)
	q = tau_pct / 100
	for t in range(H):
		if counts[t] >= min_clean_samples:
			tau[t] = torch.quantile(signal[t][mask[t]].float(), q)
		elif t > 0:
			tau[t] = tau[t - 1]
			carried[t] = True
		else:
			tau[t] = torch.quantile(signal[t].flatten().float(), q)
			carried[t] = True
	return Calibration(tau=tau.clamp(min=floor), clean_counts=counts, carried=carried)
