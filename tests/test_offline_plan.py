"""CPU test for pessimistic offline planning (mock robot, fake env): every arm runs; no penalty = standard planner."""
import json

import torch

from src import offline_plan as op
from src.planning.audited_planner import AuditedPlanner, PlannerConfig
from src.preflight import run_preflight as rp
from tests.test_detector_planning import _inputs
from tests.test_monitor import FakeEnv, _agent, _episodes


def test_no_penalty_equals_standard():
	agent = _agent()
	z, acts = _inputs()
	acts = acts[:3]
	ref = AuditedPlanner(agent, PlannerConfig(horizon=3, score="standard")).estimate_value(z, acts)[0]
	ours = op.PessimisticPlanner(agent, None, 0.0).estimate_value(z, acts)[0]
	assert torch.allclose(ours, ref, atol=1e-5)


def test_offline_plan_end_to_end(tmp_path, monkeypatch):
	agent = _agent()
	torch.save(_episodes(E=20, T=30), tmp_path / "ds.pt")
	monkeypatch.setattr(rp, "load_agent", lambda *a, **k: agent)
	import src.envs.maniskill3 as ms
	monkeypatch.setattr(ms, "ManiSkill3Env", lambda task, seed: FakeEnv(T=5))
	op.main(["--model", "x", "--task", "PushCube-v1", "--dataset", str(tmp_path / "ds.pt"), "--fit_episodes", "20",
	         "--episodes", "2", "--out", str(tmp_path / "o")])
	s = json.loads((tmp_path / "o" / "summary.json").read_text())
	assert "none" in s and "LBA_l1.0" in s and "M_l2.0" in s and "B_l0.5" in s
