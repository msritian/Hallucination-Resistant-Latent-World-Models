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
	return dict(obs=torch.randn(E, T + 1, 6, generator=g), action=torch.rand(E, T, 3, generator=g) * 2 - 1, success=success,
	            reward=torch.rand(E, T, generator=g))


def test_evaluate_model_runs_and_reports_all_tests(tmp_path, monkeypatch):
	monkeypatch.setitem(rp.CAL, "min_clean_samples", 5)
	report = {}
	for name, aux in [("grounded", 4), ("stock", 0)]:
		report[name] = rp.evaluate_model(_mock_agent(aux), _data(), H=5, beta=1.0, dump_path=tmp_path / f"features_{name}.pt")
	g = report["grounded"]
	assert set(g["results"]) - {"LBA", "LBA_lin", "LENS", "LALL"} == {"A", "B", "E", "D", "M", "C", "Ao", "P", "At", "Aa", "Ac", "A+Aa", "A+Ac", "A2", "A3", "A5", "A8", "A+A2", "A+A3", "A+A5", "A+A8", "Ab", "Acb", "Ab+Acb", "A+Aa+P", "A+A5+P"}
	assert set(report["stock"]["results"]) - {"LBA", "LBA_lin"} == {"A", "B", "E", "C", "Ao", "P", "At", "Aa", "Ac", "A+Aa", "A+Ac", "A2", "A3", "A5", "A8", "A+A2", "A+A3", "A+A5", "A+A8", "Ab", "Acb", "Ab+Acb", "A+Aa+P", "A+A5+P"}
	assert 0 <= g["type2_effective_fraction"] <= 1 and "type4_value_cum" in g["results"]["A"]
	assert {"gradual", "gradual_cum", "sudden_natural", "increment"} <= set(g["results"]["A3"])
	assert len(g["critic_bias_by_step"]) == 5
	assert {"real_all", "real_gradual", "real_sudden", "real_all_cum"} <= set(g["results"]["A+Aa+P"])
	assert g["real_label_diagnostics"]["n_neg"] > 0
	dump = torch.load(tmp_path / "features_grounded.pt", weights_only=False)
	assert dump["ev"]["E"].shape[0] == 4 and "Aa" in dump["cal"]["signals"] and len(dump["ev"]["episode"]) == dump["ev"]["E"].shape[1]
	assert {"simrel_all", "simP95_sudden", "simP999_gradual"} <= set(g["results"]["A+Aa+P"])
	assert set(g["real_label_diagnostics"]["answer_key_checks"]) == {"simP95", "simP999", "simrel"}
	learned = {"LBA", "LBA_lin", "LENS", "LALL"} & set(g["results"])
	assert learned in (set(), {"LBA", "LBA_lin", "LENS", "LALL"})  # empty only if too few positives in the mock
	assert "LENS" not in report["stock"]["results"]
	gd = g["gradual_diagnostics"]
	assert gd["n_sudden"] + gd["n_gradual"] >= 0 and 0 <= gd["sign_consistency_clean"] <= 1
	for test in ["type4", "type2", "type2_alpha0.5", "type1_sigma1.0"]:
		for sig, res in g["results"].items():
			if sig in {"LBA", "LBA_lin", "LENS", "LALL"}:  # learned audits are scored on simulator labels only
				continue
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
