import torch

from src.tools import rollout_vs_sim as rv
from tests.test_preflight_pipeline import _mock_agent


def test_probe_recovers_linear_map():
	g = torch.Generator().manual_seed(0)
	z = torch.randn(500, 8, generator=g)
	x = z @ torch.randn(8, 3, generator=g) + 0.5
	pr = rv.fit_probe(z, x)
	assert pr["r2"].min() > 0.99
	assert torch.allclose(rv.probe(pr["W"].double(), z.double()).float(), x, atol=1e-2)


def test_nn_distance():
	bank = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
	q = torch.tensor([[[0.0, 0.1]], [[1.0, 1.0]]])
	assert torch.allclose(rv.nn_distance(q, bank), torch.tensor([[0.1], [0.0]]), atol=1e-6)


def test_analysis_pipeline(tmp_path):
	agent = _mock_agent(0)
	g = torch.Generator().manual_seed(0)
	H, D, A, K = 6, 6, 3, 4
	starts = [dict(obs0=torch.randn(D, generator=g), cand_actions=torch.rand(H, 2 * K, A, generator=g) * 2 - 1,
	               cand_r=torch.rand(H, 2 * K, generator=g), cand_obs=torch.randn(H, 2 * K, D, generator=g),
	               cand_kind=torch.cat([torch.zeros(K), torch.ones(K)]).long(),
	               pol_actions=torch.rand(H, 1, A, generator=g) * 2 - 1, pol_r=torch.rand(H, 1, generator=g),
	               pol_obs=torch.randn(H, 1, D, generator=g)) for _ in range(5)]
	obs = torch.randn(200, D, generator=g)
	z = agent.model.encode(obs, None).detach()
	pr = rv.fit_probe(z, obs)
	res = rv.analyze(agent, starts, pr["W"], z[20:], z[:20])
	names = [f"obs[{i}]" for i in range(D)]
	S = rv.summarize(res, names, pr["r2"], [0.0])
	assert len(S["policy/all"]) == H and set(S) >= {"candidates/elite", "candidates/random"}
	ex = rv.examples(res, names)
	assert len(ex) == 3 and len(ex[0]["steps"]) == H
	rv.write_report(tmp_path, S, ex, type("A", (), dict(task="T", run="grounded", horizon=H))())
	assert "Concrete examples" in (tmp_path / "report.md").read_text()
