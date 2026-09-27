import torch

from src.auditor.source_fix import auditor_loss, ramp, rescale_to_cap
from tests.mocks import MockWorldModel, make_cfg

B, OBS, A = 8, 6, 3


def _loss(model, H):
	g = torch.Generator().manual_seed(0)
	obs0 = torch.randn(B, OBS, generator=g)
	actions = torch.rand(H, B, A, generator=g) * 2 - 1
	return auditor_loss(model, make_cfg(), obs0, actions, discount=0.95, q_scale=1.0)


def test_gradient_reaches_only_dynamics_through_next_latent():
	# H = 1: Q(z_0, a_0) does not depend on the dynamics, so any dynamics gradient must flow through z_1.
	model = MockWorldModel()
	model.zero_grad(set_to_none=True)
	_loss(model, H=1).backward()
	assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model._dynamics.parameters())
	for name in ["_encoder", "_reward", "_pi", "_Qs"]:
		for p in getattr(model, name).parameters():
			assert p.grad is None, f"{name} received gradient from the auditor loss"


def test_policy_requires_grad_restored():
	model = MockWorldModel()
	_loss(model, H=12).backward()
	assert all(p.requires_grad for p in model._pi.parameters())


def test_rescale_caps_value_and_keeps_direction():
	x = torch.tensor(5.0, requires_grad=True)
	out = rescale_to_cap(x * 2, torch.tensor(1.0))
	assert abs(out.item() - 1.0) < 1e-6
	out.backward()
	assert x.grad.item() > 0
	y = torch.tensor(0.1, requires_grad=True)
	assert abs(rescale_to_cap(y, torch.tensor(1.0)).item() - 0.1) < 1e-6


def test_ramp_schedule():
	assert ramp(0, 1000) == 0.0
	assert ramp(249, 1000) == 0.0
	assert abs(ramp(300, 1000) - 0.5) < 1e-9
	assert ramp(400, 1000) == 1.0
