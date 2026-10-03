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
	return {k: torch.full((H,), float(value)) for k in ["A", "B", "D", "E", "M", "Ao", "P", "At"]}


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


@pytest.mark.parametrize("fallback", ["qmin", "v"])
def test_fallbacks_with_tiny_tau(fallback):
	from src.auditor.signals import pi_mean, q_heads
	agent = _agent()
	z, act = _inputs(12)
	p = AuditedPlanner(agent, PlannerConfig(horizon=12, score="trust", signal="A", kappa=float("inf"), fallback=fallback),
	                   _taus(1e-12))
	value, _ = p.estimate_value(z, act)
	a0 = act[0] if fallback == "qmin" else pi_mean(agent.model, z)
	heads = q_heads(agent.model, z, a0, agent.cfg)
	expected = heads.min(0).values if fallback == "qmin" else heads.mean(0)
	assert torch.allclose(value, expected, atol=1e-5)


@pytest.mark.parametrize("signal", ["Ao", "P", "At"])
def test_new_signals_plan_and_huge_tau_matches_standard(signal):
	agent = _agent()
	z, act = _inputs(6)
	std, _ = AuditedPlanner(agent, PlannerConfig(horizon=6, score="standard")).estimate_value(z, act)
	p = AuditedPlanner(agent, PlannerConfig(horizon=6, score="trust", signal=signal), _taus(1e9))
	tr, _ = p.estimate_value(z, act)
	assert torch.allclose(std, tr, atol=1e-4)
	assert p.act(torch.randn(6), t0=True).shape == (A,)


def test_shuffled_trust_permutes_weights_across_candidates():
	agent = _agent()
	z, act = _inputs(6, N=64)
	taus = {k: torch.full((24,), 0.02) for k in ["A"]}
	_, om = AuditedPlanner(agent, PlannerConfig(horizon=6, signal="A"), taus).estimate_value(z, act)
	_, om_s = AuditedPlanner(agent, PlannerConfig(horizon=6, signal="A", shuffle_trust=True), taus).estimate_value(z, act)
	h, h_s = om.sum(0).flatten().sort().values, om_s.sum(0).flatten().sort().values
	assert torch.allclose(h, h_s, atol=1e-5)
	assert "shuffled" in PlannerConfig(shuffle_trust=True).name()


def test_elvis_lambda_signals():
	from src.auditor.elvis import elvis_return
	from src.auditor.signals import audit_rollout
	agent = _agent()
	z, act = _inputs(6)
	p = AuditedPlanner(agent, PlannerConfig(horizon=6, score="elvis", lambda_signal="const", lambda_max=0.5))
	v, _ = p.estimate_value(z, act)
	roll = audit_rollout(agent.model, agent.cfg, z, act, 0.95)
	assert torch.allclose(v, elvis_return(roll.r_hat, roll.v_mean, torch.full_like(roll.v_mean, 0.5), 0.95), atol=1e-5)
	assert p.p.name() == "elvis_H6_const0.5"
	pa = AuditedPlanner(agent, PlannerConfig(horizon=6, score="elvis", lambda_signal="A"))
	assert pa.estimate_value(z, act)[0].shape == (16, 1) and pa.act(torch.randn(6), t0=True).shape == (A,)
	assert pa.p.name().endswith("_sigA")


def test_optimism_penalty():
	from src.auditor.signals import audit_rollout
	from src.planning.audited_planner import optimism_penalty
	agent = _agent()
	z, act = _inputs(6)
	base, _ = AuditedPlanner(agent, PlannerConfig(horizon=6, score="elvis", lambda_signal="const", lambda_max=0.8)).estimate_value(z, act)
	p = AuditedPlanner(agent, PlannerConfig(horizon=6, score="elvis", lambda_signal="const", lambda_max=0.8, penalty=1.0))
	pen, _ = p.estimate_value(z, act)
	roll = audit_rollout(agent.model, agent.cfg, z, act, 0.95)
	lam = torch.full_like(roll.v_mean, 0.8)
	manual = sum(0.95 ** t * 0.8 ** t * roll.optimism()[t] for t in range(6))
	assert torch.allclose(optimism_penalty(roll, lam, 0.95), manual, atol=1e-5)
	assert torch.allclose(pen, base - manual, atol=1e-4) and torch.all(pen <= base + 1e-6)
	assert p.p.name() == "elvis_H6_const0.8_pen1.0"


def test_min_value_aggregation():
	from src.auditor.signals import pi_mean, q_heads
	agent = _agent()
	z, act = _inputs(4)
	std, _ = AuditedPlanner(agent, PlannerConfig(horizon=4, score="standard")).estimate_value(z, act)
	pm = AuditedPlanner(agent, PlannerConfig(horizon=4, score="standard", value_agg="min"))
	vmin, _ = pm.estimate_value(z, act)
	assert torch.all(vmin <= std + 1e-6) and pm.p.name() == "standard_H4_vmin"
	c, _ = AuditedPlanner(agent, PlannerConfig(horizon=4, score="elvis", lambda_signal="const", lambda_max=1.0, value_agg="min")).estimate_value(z, act)
	assert torch.allclose(c, vmin, atol=1e-4)  # lambda = 1 reduces to the standard score with the same value
