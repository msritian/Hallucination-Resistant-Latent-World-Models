"""CPU tests for the Idea 1 pilot (mock model, fake simulator): candidates, features, true scores, every rule, resume."""
import json

import pytest
import torch

from src import rerank as rr
from src.preflight import run_preflight as rp
from src.tools.plan_diag import SimSnapshot
from tests.test_monitor import A, OBS, FakeEnv, _agent, _episodes


def test_rules_offline_and_true_scores():
	torch.manual_seed(0)
	agent, env = _agent(), FakeEnv(T=6)
	sim = SimSnapshot(env)
	from src.planning.audited_planner import AuditedPlanner, PlannerConfig
	pf = lambda: AuditedPlanner(agent, PlannerConfig(horizon=12, score="standard"))
	z = agent.model.encode(torch.randn(13, 4, OBS), None)
	bias = rp.calibration_bias(agent, z, torch.rand(12, 4, A) * 2 - 1, 1.0, 12)
	recs = rr.collect_calibration(agent, env, sim, pf, bias, 8, 2, 0)
	assert len(recs) == 12 and recs[0]["X"].shape[0] == 8 and set(recs[0]["unc"]) == {"D", "M"}
	# true score of a candidate equals executing it by hand
	obs = env.reset(seed=3)
	p = pf(); p.act(obs, t0=True)
	acts, _ = rr.candidates(p, 4)
	snap = sim.save()
	true = rr.true_scores(agent, sim, snap, acts)
	x = env.env.unwrapped.x.clone()
	rew = []
	for t in range(12):
		x[:A] += acts[t, 1].cpu() / 10
		rew.append(float(torch.exp(-x.abs().sum())))
	v = rp.value_fn(agent, agent.model.encode(x.view(1, -1), None))
	expect = sum(0.95 ** t * r for t, r in enumerate(rew)) + 0.95 ** 12 * float(v)
	assert abs(float(true[1]) - expect) < 1e-4
	model, lam = rr.fit_correction(recs), {r: rr.tune_lambda(recs, r[-1]) for r in ("penD", "penM")}
	q = rr.offline_quality(recs, model, lam)
	assert set(q) == {"imagined", "corrected", "penD", "penM", "random"} and all(v >= 0 for v in q.values())
	for rule in rr.RULES:
		rows = rr.run_rule(agent, env, sim, pf, rule, bias, 8, model, lam, 1, 0)
		assert rows[0]["arm"] == rule and rows[0]["seed"] == 0


def test_main_resume(tmp_path, monkeypatch):
	agent = _agent()
	torch.save(_episodes(T=30), tmp_path / "ep.pt")
	monkeypatch.setattr(rp, "load_agent", lambda *a, **k: agent)
	import src.envs.maniskill3 as ms
	monkeypatch.setattr(ms, "ManiSkill3Env", lambda task, seed: FakeEnv(T=5))
	argv = ["--model", "x", "--task", "PushCube-v1", "--episodes_file", str(tmp_path / "ep.pt"), "--out", str(tmp_path / "o"),
	        "--cal_episodes", "4", "--episodes", "2", "--K", "8"]
	with pytest.raises(SystemExit):
		rr.main(argv + ["--max_hours", "0"])
	rr.main(argv)
	assert set(json.loads((tmp_path / "o" / "summary.json").read_text())) == set(rr.RULES)
	assert "offline_regret_heldout" in json.loads((tmp_path / "o" / "offline.json").read_text())
