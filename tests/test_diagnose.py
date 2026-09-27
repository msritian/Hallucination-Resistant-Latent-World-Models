import json

import torch

from src.preflight import diagnose as dg
from tests.test_preflight_pipeline import _data, _mock_agent


def test_spearman_basic():
	x = torch.arange(10.0)
	assert abs(dg.spearman(x, x ** 3) - 1.0) < 1e-9
	assert abs(dg.spearman(x, -x) + 1.0) < 1e-9


def test_diagnose_runs_on_mock(tmp_path, monkeypatch):
	monkeypatch.setitem(dg.CAL, "min_clean_samples", 5)
	report = {name: dg.diagnose_model(_mock_agent(aux), _data(), H=5) for name, aux in [("grounded", 4), ("stock", 0)]}
	g = report["grounded"]
	assert set(g["per_signal"]) == {"A", "B", "E", "D", "M", "C"}
	for p in g["per_signal"].values():
		for k in ("type4_latent", "type4_value"):
			v = p[k]["auroc"]
			assert v != v or 0 <= v <= 1
	assert 0 <= g["overlap"]["frac_latent_pos_that_change_value"] <= 1
	assert g["critic_noise"]["median_delta_with_real_next_state"] >= 0
	json.dumps(report)
	dg.write_report(tmp_path, report)
	assert "value-error labels" in (tmp_path / "diagnostic.md").read_text()
