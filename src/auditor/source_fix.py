"""Auditor loss for the source fix (execution_final.md §3.6). Only the dynamics network receives gradient."""
import torch

from src.auditor.signals import audit_rollout
from src.utils.frozen import frozen


def auditor_loss(model, cfg, obs0, actions, discount: float, q_scale, task=None) -> torch.Tensor:
	"""Discounted mean Bellman residual over an open-loop rollout along logged ``actions`` [H, B, A].

	The start latent is encoded without gradient (no encoder gradient), Q is evaluated with detached
	parameters, the policy is frozen, and the predicted reward is detached.
	"""
	with torch.no_grad():
		z0 = model.encode(obs0, task)
	with frozen(model._pi):
		roll = audit_rollout(model, cfg, z0, actions, discount, task, differentiable=True)
	H = actions.shape[0]
	g = discount ** torch.arange(H, device=roll.delta.device, dtype=roll.delta.dtype).view(H, 1, 1)
	return (g * roll.delta).sum(0).mean() / g.sum() / q_scale


def rescale_to_cap(aux: torch.Tensor, cap: torch.Tensor) -> torch.Tensor:
	"""Scale ``aux`` so its value never exceeds ``cap`` while keeping its gradient direction."""
	return aux * torch.clamp(cap.detach() / (aux.detach() + 1e-8), max=1.0)


def ramp(step: int, total_steps: int, warmup_frac: float = 0.25, ramp_frac: float = 0.10) -> float:
	"""0 during warm-up, then linear to 1 over ``ramp_frac`` of training."""
	start, length = warmup_frac * total_steps, ramp_frac * total_steps
	if step < start:
		return 0.0
	return min(1.0, (step - start) / max(length, 1.0))
