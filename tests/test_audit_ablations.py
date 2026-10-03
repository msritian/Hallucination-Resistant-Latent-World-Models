"""End-to-end check of the offline audit analyses on mock dumps (catches shape/indexing bugs)."""
import torch

from src.preflight import run_preflight as rp
from src.tools import audit_ablations as aa
from tests.test_preflight_pipeline import _mock_agent


def _data(E=12, T=24, seed=0):
	g = torch.Generator().manual_seed(seed)
	success = torch.zeros(E, T)
	success[1, 12:] = 1.0
	success[5, 6:] = 1.0
	return dict(obs=torch.randn(E, T + 1, 6, generator=g), action=torch.rand(E, T, 3, generator=g) * 2 - 1,
	            success=success, reward=torch.rand(E, T, generator=g))


def test_ablation_report_runs(tmp_path, monkeypatch):
	monkeypatch.setitem(rp.CAL, "min_clean_samples", 5)
	for name, aux, seed in [("grounded", 4, 0), ("stock", 0, 1)]:
		rp.evaluate_model(_mock_agent(aux), _data(seed=seed), H=5, beta=1.0, dump_path=tmp_path / f"features_{name}.pt")
	dumps = aa.load_dumps([str(tmp_path)])
	assert len(dumps) == 2
	text = aa.run(dumps)
	for heading in ["Which signals matter", "How many episodes", "Transfer", "MOBILE-style", "already solved"]:
		assert heading in text
