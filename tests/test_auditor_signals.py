import torch

from src.auditor.injection import injector_at
from src.auditor.signals import audit_rollout, bellman_target_spread, dynamics_disagreement, q_heads
from tests.mocks import ConsistentWorldModel, MockWorldModel, make_cfg

H, N, D, A = 5, 7, 16, 3


def _inputs(seed=0):
	g = torch.Generator().manual_seed(seed)
	z0 = torch.softmax(torch.randn(N, D, generator=g).view(N, -1, 8), -1).view(N, D)
	actions = torch.rand(H, N, A, generator=g) * 2 - 1
	return z0, actions


def test_q_heads_matches_manual_two_hot_decoding():
	cfg = make_cfg()
	model = MockWorldModel(num_bins=cfg.num_bins)
	z0, actions = _inputs()
	logits = model.Q(z0, actions[0], None, return_type="all")
	bins = torch.linspace(cfg.vmin, cfg.vmax, cfg.num_bins)
	x = (torch.softmax(logits, -1) * bins).sum(-1, keepdim=True)
	manual = torch.sign(x) * (torch.exp(x.abs()) - 1)
	assert torch.allclose(q_heads(model, z0, actions[0], cfg), manual, atol=1e-6)


def test_residual_is_zero_for_consistent_model():
	model = ConsistentWorldModel(latent_dim=D, action_dim=A)
	z0, actions = _inputs()
	roll = audit_rollout(model, make_cfg(num_bins=0), z0, actions, discount=model.gamma)
	assert roll.delta.shape == (H, N, 1)
	assert roll.delta.abs().max() < 1e-5


def test_injected_teleport_flags_only_the_jump():
	model = ConsistentWorldModel(latent_dim=D, action_dim=A)
	z0, actions = _inputs()
	other = torch.softmax(torch.randn(N, D).view(N, -1, 8) * 5, -1).view(N, D)
	roll = audit_rollout(
		model, make_cfg(num_bins=0), z0, actions, discount=model.gamma, inject=injector_at(3, lambda z: other)
	)
	# delta_2 judges the transition into the injected z_3; all other steps stay consistent
	assert roll.delta[2].abs().min() > 1e-4
	mask = torch.ones(H, dtype=torch.bool)
	mask[2] = False
	assert roll.delta[mask].abs().max() < 1e-5


def test_signals_are_deterministic_and_shaped():
	cfg = make_cfg()
	model = MockWorldModel(latent_dim=D, action_dim=A)
	z0, actions = _inputs()
	r1 = audit_rollout(model, cfg, z0, actions, discount=0.95)
	r2 = audit_rollout(model, cfg, z0, actions, discount=0.95)
	assert torch.equal(r1.delta, r2.delta)
	assert r1.v_heads.shape == (5, H + 1, N, 1)
	assert r1.critic_spread().shape == (H, N, 1)
	assert r1.ucb(beta=1.0).shape == (H + 1, N, 1)
	assert r1.q_terminal.shape == (N, 1)


def test_ensemble_signals_shapes_and_zero_when_heads_agree():
	cfg = make_cfg()
	model = MockWorldModel(latent_dim=D, action_dim=A)
	z0, actions = _inputs()
	roll = audit_rollout(model, cfg, z0, actions, discount=0.95)
	same = [lambda z, a: model.next(z, a, None)] * 5
	assert dynamics_disagreement(same, roll.z, actions).abs().max() < 1e-6
	assert bellman_target_spread(same, model, cfg, roll, actions).abs().max() < 1e-5
	extra = [torch.nn.Linear(D + A, D) for _ in range(4)]
	heads = [lambda z, a: model.next(z, a, None)] + [
		(lambda h: (lambda z, a: torch.softmax(h(torch.cat([z, a], -1)), -1)))(h) for h in extra
	]
	assert dynamics_disagreement(heads, roll.z, actions).shape == (H, N, 1)
	assert bellman_target_spread(heads, model, cfg, roll, actions).shape == (H, N, 1)
