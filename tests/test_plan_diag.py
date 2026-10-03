"""CPU tests for the ground-truth planning diagnostics (mock model, fake simulator)."""
import json
from types import SimpleNamespace

import torch

from src.preflight import run_preflight as rp
from src.tools import plan_diag as pd
from tests.test_preflight_pipeline import _mock_agent


class FakeUnwrapped:
	"""Deterministic 'simulator': state x, step x <- x + a, obs = x, reward = -|x|."""

	def __init__(self):
		self.x = torch.zeros(3)
		self._elapsed_steps = torch.tensor([0])

	def get_state_dict(self):
		return {"x": self.x.clone()}

	def set_state_dict(self, s):
		self.x = s["x"].clone()

	def step(self, a):
		self.x = self.x + torch.as_tensor(a).reshape(-1)
		self._elapsed_steps = self._elapsed_steps + 1
		return self.x.clone().view(1, -1), -self.x.abs().sum(), False, False, {}


def test_snapshot_rollout_restores_state():
	u = FakeUnwrapped()
	sim = pd.SimSnapshot(SimpleNamespace(env=SimpleNamespace(unwrapped=u)))
	u.x = torch.tensor([1.0, 0.0, 0.0])
	snap = sim.save()
	acts = torch.ones(4, 2, 3) * torch.tensor([1.0, -1.0]).view(1, 2, 1)
	rew, obs = sim.rollout(snap, acts)
	assert rew.shape == (4, 2) and obs.shape == (4, 2, 3)
	assert torch.allclose(obs[-1, 0], torch.tensor([5.0, 4.0, 4.0])) and torch.allclose(obs[-1, 1], torch.tensor([-3.0, -4.0, -4.0]))
	assert torch.equal(u.x, torch.tensor([1.0, 0.0, 0.0])) and int(u._elapsed_steps) == 0


def test_pick_candidates_sources():
	pool = dict(value=torch.arange(20.0).view(-1, 1), actions=torch.zeros(3, 20, 2))
	idx, src = pd.pick_candidates(pool, num_pi=4, n_top=5, n_rand=5, gen=torch.Generator().manual_seed(0))
	assert idx[:4].tolist() == [0, 1, 2, 3] and idx[4:9].tolist() == [19, 18, 17, 16, 15]
	assert len(set(idx.tolist())) == 14 and src.tolist() == [0] * 4 + [1] * 5 + [2] * 5


def test_analyze_and_report(tmp_path, monkeypatch):
	monkeypatch.setitem(rp.CAL, "min_clean_samples", 5)
	agent = _mock_agent(aux=4)
	g = torch.Generator().manual_seed(0)
	H, K = 6, 12
	records = [dict(obs0=torch.randn(6, generator=g), actions=torch.rand(H, K, 3, generator=g) * 2 - 1,
	                r_true=torch.rand(H, K, generator=g), obs_true=torch.randn(H, K, 6, generator=g),
	                source=torch.tensor([0] * 4 + [1] * 4 + [2] * 4), owner=torch.zeros(K)) for _ in range(8)]
	taus = {95: {k: [1.0] * 24 for k in ["A", "B", "E", "D", "M", "Ao", "P", "At"]}}
	d = pd.analyze(agent, records, taus)
	assert d["num_candidates"] == 8 * K and set(d["per_step"]) == {"pi", "top", "rand"}
	assert len(d["per_step"]["rand"]["value_err"]) == H
	assert {"standard_H", "standard_H3", "oracle_trust_fbq", "trust_A_candidates_fbv", "trust_A_onpolicy_fbq_shuffled", "elvis_ucb", "elvis_A", "elvis_const0.5"} <= set(d["scoring_rules"])
	assert d["scoring_rules"]["standard_H"]["regret_vs_standard"][0] == 0
	for v in d["scoring_rules"].values():
		assert v["regret"] >= 0
	assert set(d["taus_candidates"]) == {90, 95, 99} and len(d["taus_candidates"][95]["A"]) == H
	d["determinism_max_err"] = 0.0
	json.dumps(d)
	pd.write_report(tmp_path, d)
	assert "Scoring rules" in (tmp_path / "report.md").read_text()


def test_spearman():
	x = torch.arange(10.0)
	assert abs(pd.spearman(x, x) - 1) < 1e-6 and abs(pd.spearman(x, -x) + 1) < 1e-6
