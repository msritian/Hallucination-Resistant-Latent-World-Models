"""Model-independent preflight analysis: calibration, normalization, detection scores (execution_final.md §3.4, §6).

Inputs are per-step signal tensors [H, M] (H steps, M replay windows) and true latent errors e_{t+1} [H, M].
"""
import torch

from src.auditor.calibrator import calibrate, error_thresholds


def calibrate_all(signals: dict, err: torch.Tensor, cfg: dict) -> tuple[dict, dict]:
	"""Error thresholds from one-step errors, then tau(t) per signal on ground-truth-clean replays."""
	th = error_thresholds(err, cfg["eps_clean_pct"], cfg["eps_hall_pct"])
	taus = {
		name: calibrate(x, err, th.eps_clean, cfg["tau_pct"], cfg["min_clean_samples"])
		for name, x in signals.items()
	}
	return taus, dict(eps_clean=th.eps_clean, eps_hall=th.eps_hall)


def normalize(signals: dict, taus: dict) -> dict:
	"""s_t = x_t / tau_x(t); Signal C = max(s_A, s_B)."""
	s = {name: x / taus[name].tau.to(x.device).view(-1, 1) for name, x in signals.items()}
	if "A" in s and "B" in s:
		s["C"] = torch.maximum(s["A"], s["B"])
	return s


def replay_scores(s: dict, first_step: int = 2) -> dict:
	"""Types 1-3 detection score per replay: max over steps t >= first_step of s_t. Returns {signal: [M]}."""
	return {name: x[first_step:].max(dim=0).values for name, x in s.items()}


def natural_labels(err: torch.Tensor, eps_clean: float, eps_hall: float) -> tuple[torch.Tensor, torch.Tensor]:
	"""Type 4 per-step labels from e_{t+1}: positive if > eps_hall, negative if <= eps_clean, else excluded.

	Returns (labels [H, M] bool, keep [H, M] bool).
	"""
	pos = err > eps_hall
	neg = err <= eps_clean
	return pos, pos | neg
