from types import SimpleNamespace

import pytest
import torch

from src.planning.audited_planner import AuditedPlanner, PlannerConfig
from tests.mocks import MockWorldModel, make_cfg

A = 3


def _agent(aux=0):
	cfg = make_cfg()
	cfg.__dict__.update(action_dim=A, num_samples=64, num_pi_trajs=8, iterations=3, num_elites=8,
	                    temperature=0.5, min_std=0.05, max_std=2.0)
	model = MockWorldModel(obs_dim=6, latent_dim=16, action_dim=A)
	extra = [torch.nn.Linear(16 + A, 16) for _ in range(aux)]
	heads = lambda: [lambda z, a: model.next(z, a, None)] + [
		(lambda h: (lambda z, a: torch.softmax(h(torch.cat([z, a], -1)).view(*z.shape[:-1], -1, 8), -1).view(z.shape)))(h)
		for h in extra]
	return SimpleNamespace(model=model, cfg=cfg, discount=0.95, num_aux_dynamics=aux, aux_dynamics_heads=heads)


def _taus(value, H=24):
	return {k: torch.full((H,), float(value)) for k in "ABDEM"}


def _inputs(H, N=16, seed=0):
	g = torch.Generator().manual_seed(seed)
	z = torch.softmax(torch.randn(N, 16, generator=g).view(N, -1, 8), -1).view(N, 16)
	return z, torch.rand(H, N, A, generator=g) * 2 - 1


def test_trust_with_huge_tau_equals_standard():
	agent = _agent()
	z, act = _inputs(12)
	std, _ = AuditedPlanner(agent, PlannerConfig(horizon=12, score="standard")).estimate_value(z, act)
	tr, omega = AuditedPlanner(agent, PlannerConfig(horizon=12, score="trust", signal="A"), _taus(1e9)).estimate_value(z, act)
	assert torch.allclose(std, tr, atol=1e-4) and torch.all(omega == 1)


def test_trust_with_tiny_tau_gives_critic_of_first_action():
	agent = _agent()
	z, act = _inputs(12)
	p = AuditedPlanner(agent, PlannerConfig(horizon=12, score="trust", signal="A", kappa=float("inf")), _taus(1e-12))
	value, _ = p.estimate_value(z, act)
	from src.auditor.signals import q_heads
	assert torch.allclose(value, q_heads(agent.model, z, act[0], agent.cfg).mean(0), atol=1e-5)


def test_elvis_lambda_one_equals_standard():
	agent = _agent()
	z, act = _inputs(6)
	std, _ = AuditedPlanner(agent, PlannerConfig(horizon=6, score="standard")).estimate_value(z, act)
	el, _ = AuditedPlanner(agent, PlannerConfig(horizon=6, score="elvis", lambda_min=1.0, lambda_max=1.0)).estimate_value(z, act)
	assert torch.allclose(std, el, atol=1e-4)


@pytest.mark.parametrize("H", [3, 12, 24])
@pytest.mark.parametrize("score,signal,aux", [("standard", "A", 0), ("trust", "A", 0), ("trust", "C", 0),
                                              ("trust", "E", 0), ("trust", "D", 4), ("trust", "M", 4), ("elvis", "A", 0)])
def test_plan_runs_for_all_modes_and_horizons(H, score, signal, aux):
	agent = _agent(aux)
	p = AuditedPlanner(agent, PlannerConfig(horizon=H, score=score, signal=signal), _taus(1.0))
	obs = torch.randn(6)
	for t in range(3):
		a = p.act(obs, t0=t == 0)
		assert a.shape == (A,) and torch.isfinite(a).all() and a.abs().max() <= 1
	assert p.prev_mean.shape == (H, A)
	assert len(p.stats["plan_ms"]) == 3
	if score == "trust":
		assert all(0 <= h <= H + 1e-4 for h in p.stats["h_eff"])


def test_missing_tau_or_heads_is_rejected():
	with pytest.raises(AssertionError):
		AuditedPlanner(_agent(), PlannerConfig(horizon=24, score="trust", signal="A"), _taus(1.0, H=12))
	with pytest.raises(AssertionError):
		AuditedPlanner(_agent(0), PlannerConfig(horizon=12, score="trust", signal="D"), _taus(1.0))
