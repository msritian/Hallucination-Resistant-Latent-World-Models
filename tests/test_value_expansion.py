"""CPU tests for hallucination-aware value expansion: the mixing rule and the targets of every mode."""
from types import SimpleNamespace

import torch

from src.auditor.signals import decode
from src.training import value_expansion as ve
from tests.test_audit_replay import _episodes
from tests.test_monitor import _agent


def test_mix_limits_and_weights():
	G = torch.arange(6.0).view(6, 1, 1).expand(6, 4, 1)
	assert torch.allclose(ve.mix(G, torch.zeros(5, 4, 1)), G[0])            # no trust -> standard target
	assert torch.allclose(ve.mix(G, torch.ones(5, 4, 1)), G[5])             # full trust -> fixed h-step expansion
	t = torch.rand(5, 4, 1)
	T = torch.cat([torch.ones(1, 4, 1), torch.cumprod(t, 0)])
	w = torch.cat([T[:5] * (1 - t), T[5:]])
	assert torch.allclose(w.sum(0), torch.ones(4, 1))


class QMin(torch.nn.Module):
	"""Adds TD-MPC2's Q(return_type='min', target=True) to the mock model (min of two heads, decoded)."""
	def __init__(self, m, cfg):
		super().__init__(); self.m, self.cfg = m, cfg
	def __getattr__(self, n):
		try:
			return super().__getattr__(n)
		except AttributeError:
			return getattr(self.m, n)
	def Q(self, z, a, task, return_type="min", target=False, detach=False):
		if return_type == "all":
			return self.m.Q(z, a, task, return_type="all", detach=detach)
		return decode(self.m.Q(z, a, task, return_type="all"), self.cfg)[:2].min(0).values


def test_targets_every_mode():
	torch.manual_seed(0)
	base = _agent()
	agent = SimpleNamespace(**vars(base))
	agent.model = QMin(base.model, base.cfg)
	nz = agent.model.encode(torch.randn(3, 8, 6), None)
	rew, term = torch.rand(3, 8, 1), torch.zeros(3, 8, 1)
	for mode in ("fixed", "const", "D", "lba"):
		v = ve.ValueExpansion(mode, h=5, seed=0)
		v.refresh(agent, _episodes(E=60))
		out = v.target(agent, nz, rew, term)
		assert out.shape == (3, 8, 1) and torch.isfinite(out).all()
		if mode in ("D", "lba"):
			assert 0 <= v.stats["mean_trust"] <= 1
