import json
from collections import defaultdict

import torch

from src.evaluate import calibrate_taus, evaluate_arms, preset
from src.planning.audited_planner import PlannerConfig
from tests.test_audited_planner import _agent


class MockEnv:
	def __init__(self, T=10):
		self.T = T

	def reset(self, seed=None):
		self.g = torch.Generator().manual_seed(seed or 0)
		self.t = 0
		return torch.randn(6, generator=self.g)

	def step(self, action):
		self.t += 1
		info = defaultdict(float, success=float(self.t > 5 and float(action.sum()) > 0))
		return torch.randn(6, generator=self.g), torch.tensor(float(action.sum())), self.t >= self.T, info


def test_presets_are_well_formed():
	pilot = preset("pilot", grounded=True)
	assert pilot[0] == "stock"
	names = [a.name() for a in pilot[1:]]
	assert len(names) == len(set(names))
	assert any("trust_D" in n for n in names) and not any("trust_D" in n for n in [a.name() for a in preset("pilot", False)[1:]])
	assert len(preset("tuning", grounded=False)) == 24


def test_calibrate_and_evaluate_end_to_end(tmp_path):
	agent = _agent(aux=4)
	agent.act = lambda obs, t0=False, eval_mode=True: torch.zeros(3)
	E, T = 6, 30
	g = torch.Generator().manual_seed(0)
	data = dict(obs=torch.randn(E, T + 1, 6, generator=g), action=torch.rand(E, T, 3, generator=g) * 2 - 1,
	            success=torch.zeros(E, T))
	taus = calibrate_taus(agent, data, max_h=24, beta=1.0)
	assert set(taus[95]) == {"A", "B", "E", "D", "M"} and len(taus[95]["A"]) == 24
	arms = ["stock", PlannerConfig(horizon=3, score="standard"), PlannerConfig(horizon=12, score="trust", signal="C"),
	        PlannerConfig(horizon=24, score="trust", signal="M", tau_pct=99), PlannerConfig(horizon=6, score="elvis")]
	summary = evaluate_arms(agent, MockEnv(), arms, taus, episodes=2, seed_start=1000, out=tmp_path)
	assert set(summary) == {"stock_tdmpc2_H3", "standard_H3", "trust_C_H12_k1.0_p95", "trust_M_H24_k1.0_p99",
	                        "elvis_H6_lmin0.0_lmax1.0_b1.0"}
	assert "mean_h_eff" in summary["trust_C_H12_k1.0_p95"] and "mean_plan_ms" in summary["standard_H3"]
	rows = (tmp_path / "episodes.csv").read_text().strip().splitlines()
	assert len(rows) == 1 + 5 * 2
	json.loads((tmp_path / "summary.json").read_text())
