"""CPU tests for the detector-in-the-planner experiment: exact GPU-tree re-implementation, identical features,
reduction to the existing constant-lambda planner, dropping, shuffling, and the job runner with resume."""
import json

import numpy as np
import pytest
import torch

from src import detector_planning as dp
from src.planning.audited_planner import AuditedPlanner, PlannerConfig
from src.planning.detector_planner import DetectorPlanner, DetectorRule
from src.planning.gpu_audit import GPUTrees, candidate_audit, gpu_features
from src.preflight import run_preflight as rp
from tests.test_monitor import A, OBS, FakeEnv, _agent, _episodes


@pytest.fixture(scope="module")
def setup():
	torch.manual_seed(0)
	agent = _agent()
	lba = rp.fit_lba(agent, _episodes(), 12, beta=1.0)
	return agent, lba


def test_gpu_trees_match_sklearn_exactly():
	g = np.random.default_rng(0)
	X = g.normal(size=(3000, 7))
	y = (X[:, 0] + X[:, 1] * X[:, 2] - 0.5 * X[:, 3] ** 2 > 0.2)
	clf = rp.make_gbt().fit(X, y)
	trees = GPUTrees(clf, "cpu")
	Xt = torch.as_tensor(g.normal(size=(5000, 7)))
	assert trees.check(clf, Xt, tol=1e-6) < 1e-6


def test_gpu_features_equal_detection_features(setup):
	agent, lba = setup
	z = agent.model.encode(torch.randn(6, OBS), None)
	acts = torch.rand(12, 6, A) * 2 - 1
	sig, roll = rp.plan_signals(agent, z, acts, 1.0, bias=lba["bias"])
	ref = rp._features({k: v.cpu() for k, v in sig.items()}, roll, rp.CRITIC_FEATURES, 11, True)
	assert torch.equal(gpu_features(sig, roll, 11).reshape(-1, ref.shape[1]), ref)
	# the planner's audit gives the same probabilities as sklearn on those features
	trees = GPUTrees(lba["clf"], "cpu")
	p, _ = candidate_audit(agent, trees, lba["bias"], z, acts)
	ref_p = torch.as_tensor(lba["clf"].predict_proba(ref.numpy())[:, 1], dtype=torch.float32).view(11, 6)
	assert torch.allclose(p, ref_p, atol=1e-6)
	trees.check(lba["clf"], lba["X_sample"])


def _inputs(N=32, seed=0):
	g = torch.Generator().manual_seed(seed)
	z = torch.softmax(torch.randn(N, 16, generator=g).view(N, -1, 8), -1).view(N, 16)
	return z, torch.rand(12, N, A, generator=g) * 2 - 1


def test_constant_rule_equals_existing_constant_lambda_planner(setup):
	agent, _ = setup
	z, acts = _inputs()
	ours = DetectorPlanner(agent, DetectorRule("c", 12)).estimate_value(z, acts)[0]
	ref = AuditedPlanner(agent, PlannerConfig(horizon=12, score="elvis", lambda_signal="const", lambda_max=0.8)).estimate_value(z, acts)[0]
	assert torch.allclose(ours, ref)


def test_lba_lambda_drop_and_shuffle(setup):
	agent, lba = setup
	audit = dict(trees=GPUTrees(lba["clf"], "cpu"), bias=lba["bias"])
	z, acts = _inputs()
	p, roll = candidate_audit(agent, audit["trees"], audit["bias"], z, acts)
	# lambda_t = 0.8 (1 - p_t): the score is the lambda-return with exactly these weights
	from src.auditor.elvis import elvis_return
	lam = (0.8 * (1 - torch.cat([p, p[-1:]], 0))).unsqueeze(-1)
	s = DetectorPlanner(agent, DetectorRule("l", 12, lam="lba"), audit).estimate_value(z, acts)[0]
	assert torch.allclose(s, elvis_return(roll.r_hat, roll.v_mean, lam, 0.95))
	# dropping removes exactly the 25% with the highest max_t p_t
	d = DetectorPlanner(agent, DetectorRule("d", 12, drop=0.25), audit).estimate_value(z, acts)[0].squeeze(-1)
	worst = set(p.max(0).values.topk(8).indices.tolist())
	assert set(torch.nonzero(torch.isinf(d)).flatten().tolist()) == worst
	dr = DetectorPlanner(agent, DetectorRule("r", 12, drop=0.25, drop_random=True)).estimate_value(z, acts)[0]
	assert int(torch.isinf(dr).sum()) == 8
	# shuffling keeps the set of lambda profiles but reassigns them across candidates
	sh = DetectorPlanner(agent, DetectorRule("s", 12, lam="lba_shuffled"), audit)
	assert sh.estimate_value(z, acts)[0].shape == s.shape
	with pytest.raises(AssertionError):
		DetectorPlanner(agent, DetectorRule("x", 24, lam="lba"), audit)   # audit horizon must match


def test_full_mppi_plan_with_every_arm(setup):
	agent, lba = setup
	audits = {12: dict(trees=GPUTrees(lba["clf"], "cpu"), bias=lba["bias"])}
	lba24 = rp.fit_lba(agent, _episodes(T=60), 24, beta=1.0)
	audits[24] = dict(trees=GPUTrees(lba24["clf"], "cpu"), bias=lba24["bias"])
	obs = torch.randn(OBS)
	for arm in dp.ARMS:
		planner = DetectorPlanner(agent, arm, audits.get(arm.horizon))
		a = planner.act(obs, t0=True)
		a2 = planner.act(obs, t0=False)
		assert a.shape == (A,) and torch.isfinite(a).all() and torch.isfinite(a2).all()


def test_runner_resumes_and_writes_outputs(setup, tmp_path, monkeypatch):
	agent, _ = setup
	torch.save(_episodes(T=60), tmp_path / "ep.pt")
	monkeypatch.setattr(rp, "load_agent", lambda *a, **k: agent)
	import src.envs.maniskill3 as ms
	monkeypatch.setattr(ms, "ManiSkill3Env", lambda task, seed: FakeEnv(T=6))
	out = tmp_path / "plan"
	argv = ["--model", "x", "--run", "grounded", "--task", "PushCube-v1", "--episodes_file", str(tmp_path / "ep.pt"),
	        "--out", str(out), "--episodes", "2"]
	with pytest.raises(SystemExit) as e:
		dp.main(argv + ["--max_hours", "0"])
	assert e.value.code == 85 and len(list((out / "arms").glob("*.pt"))) == 1
	dp.main(argv)
	summary = json.loads((out / "summary.json").read_text())
	assert set(summary) == {a.name for a in dp.ARMS}
	assert len((out / "episodes.csv").read_text().strip().splitlines()) == 1 + 2 * len(dp.ARMS)
	# the report pools robots and states every pre-registered claim
	import shutil
	from src.tools import detplan_report as rep
	shutil.copytree(out, tmp_path / "plan_robot2")
	text = rep.run(rep.load([str(out), str(tmp_path / "plan_robot2")]))
	for key in ("P1", "P2", "P3", "P4", "LBAlam_H24 − L08_H24", "2 robots"):
		assert key in text
