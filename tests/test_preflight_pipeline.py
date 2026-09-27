"""End-to-end preflight evaluation on CPU with a mock world model (catches shape/indexing bugs before GPU runs)."""
import json
import math
from types import SimpleNamespace

import torch

from src.preflight import run_preflight as rp
from tests.mocks import MockWorldModel, make_cfg


def _mock_agent(aux: int):
	model = MockWorldModel(obs_dim=6, latent_dim=16, action_dim=3)
	extra = [torch.nn.Linear(16 + 3, 16) for _ in range(aux)]

	def heads():
		hs = [lambda z, a: model.next(z, a, None)]
		return hs + [(lambda h: (lambda z, a: torch.softmax(h(torch.cat([z, a], -1)).view(*z.shape[:-1], -1, 8), -1).view(z.shape)))(h) for h in extra]

	return SimpleNamespace(model=model, cfg=make_cfg(), discount=0.95, num_aux_dynamics=aux, aux_dynamics_heads=heads)


def _data(E=8, T=20, seed=0):
	g = torch.Generator().manual_seed(seed)
	success = torch.zeros(E, T)
	success[1, 12:] = 1.0
	return dict(obs=torch.randn(E, T + 1, 6, generator=g), action=torch.rand(E, T, 3, generator=g) * 2 - 1, success=success)


def test_evaluate_model_runs_and_reports_all_tests(tmp_path, monkeypatch):
	monkeypatch.setitem(rp.CAL, "min_clean_samples", 5)
	report = {}
	for name, aux in [("grounded", 4), ("stock", 0)]:
		report[name] = rp.evaluate_model(_mock_agent(aux), _data(), H=5, beta=1.0)
	g = report["grounded"]
	assert set(g["results"]) == {"A", "B", "E", "D", "M", "C"}
	assert set(report["stock"]["results"]) == {"A", "B", "E", "C"}
	for test in ["type4", "type2", "type2_alpha0.5", "type1_sigma1.0"]:
		for sig, res in g["results"].items():
			v = res[test]["auroc"]
			assert math.isnan(v) or 0.0 <= v <= 1.0, (sig, test, v)
	assert g["type2_goal"].startswith("first success state (episode 1")
	assert len(g["calibration"]["tau"]["A"]) == 5
	assert "type4_value" in g["results"]["A"]
	assert 0 <= g["value_diagnostics"]["frac_latent_errors_that_change_value"] <= 1
	assert g["value_diagnostics"]["critic_noise_median"] >= 0
	report["decision"] = rp.decisions(g["summary"])
	assert set(report["decision"]) == {"primary_value", "original_state"}
	json.dumps(report)
	rp.write_report(tmp_path, report)
	text = (tmp_path / "report.md").read_text()
	assert "Primary gate" in text and "Original gate" in text and "| A |" in text and "Value diagnostics" in text
