"""CPU tests for the run-time warning-light experiment (mock world model, fake simulator): every arm type, the
threshold calibration, resume after a checkpoint exit, and the output files."""
import json
from types import SimpleNamespace

import pytest
import torch

from src import monitor as mon
from src.preflight import run_preflight as rp
from src.tools.plan_diag import SimSnapshot
from tests.mocks import MockWorldModel, make_cfg

OBS, A = 6, 3


def _agent(aux=2):
	cfg = make_cfg()
	cfg.__dict__.update(action_dim=A, num_samples=32, num_pi_trajs=4, iterations=2, num_elites=4,
	                    temperature=0.5, min_std=0.05, max_std=2.0)
	model = MockWorldModel(obs_dim=OBS, latent_dim=16, action_dim=A)
	extra = [torch.nn.Linear(16 + A, 16) for _ in range(aux)]
	heads = lambda: [lambda z, a: model.next(z, a, None)] + [
		(lambda h: (lambda z, a: torch.softmax(h(torch.cat([z, a], -1)).view(*z.shape[:-1], -1, 8), -1).view(z.shape)))(h)
		for h in extra]
	return SimpleNamespace(model=model, cfg=cfg, discount=0.95, num_aux_dynamics=aux, aux_dynamics_heads=heads)


class FakeSim:
	"""Deterministic simulator: state x [6], step moves x[:3] by a / 10; reward = exp(-|x|)."""

	def __init__(self):
		self.x = torch.zeros(OBS)
		self._elapsed_steps = torch.tensor([0])

	def get_state_dict(self):
		return {"x": self.x.clone()}

	def set_state_dict(self, s):
		self.x = s["x"].clone()

	def step(self, a):
		self.x[:A] = self.x[:A] + torch.as_tensor(a, dtype=torch.float32).reshape(-1) / 10
		self._elapsed_steps = self._elapsed_steps + 1
		return self.x.clone().view(1, -1), torch.exp(-self.x.abs().sum()).view(1), False, False, {}


class FakeEnv:
	def __init__(self, T=14):
		self.env = SimpleNamespace(unwrapped=FakeSim())
		self.T = T

	def reset(self, seed=None):
		self.env.unwrapped.x = torch.randn(OBS, generator=torch.Generator().manual_seed(seed or 0))
		self.t, self.success = 0, False
		self.last = self.env.unwrapped.x.clone()
		return self.env.unwrapped.x.clone()

	def step(self, action):
		# the real episode must continue from where it was: catches a snapshot restore that rewinds the simulator
		assert torch.equal(self.env.unwrapped.x, self.last), "simulator state changed between real steps"
		o, r, *_ = self.env.unwrapped.step(action.numpy().reshape(1, -1))
		self.last = self.env.unwrapped.x.clone()
		self.t += 1
		self.success = self.success or float(r) > 0.3
		return o.reshape(-1), torch.tensor(float(r)), self.t >= self.T, {"success": float(self.success)}

	def close(self):
		pass


def _episodes(E=12, T=30, seed=0):
	g = torch.Generator().manual_seed(seed)
	return dict(obs=torch.randn(E, T + 1, OBS, generator=g), action=torch.rand(E, T, A, generator=g) * 2 - 1,
	            success=torch.zeros(E, T), reward=torch.rand(E, T, generator=g))


@pytest.fixture(scope="module")
def setup():
	torch.manual_seed(0)
	agent = _agent()
	lba = rp.fit_lba(agent, _episodes(), mon.H_LONG, beta=1.0)
	return agent, lba


def test_fit_lba_matches_detection_recipe(setup):
	agent, lba = setup
	assert len(lba["bias"]) == mon.H_LONG and lba["L"] == mon.H_LONG - 1 and lba["n_pos"] > 0
	assert lba["eps_hall"] >= lba["eps_clean"]


def test_score_plan_gives_every_monitor(setup):
	agent, lba = setup
	steps, roll = mon.score_plan(agent, torch.randn(OBS), torch.rand(mon.H_LONG, A) * 2 - 1, lba)
	assert set(steps) == {"LBA", "A", "B", "E", "D", "M"}
	assert all(v.shape == (lba["L"],) for v in steps.values())
	assert ((steps["LBA"] >= 0) & (steps["LBA"] <= 1)).all()
	with pytest.raises(AssertionError):
		mon.score_plan(agent, torch.randn(OBS), torch.rand(5, A), lba)


def test_plan_signals_match_rollout_signals(setup):
	agent, _ = setup
	z = agent.model.encode(torch.randn(mon.H_LONG + 1, 4, OBS), None)
	acts = torch.rand(mon.H_LONG, 4, A) * 2 - 1
	s1, _, _ = rp.rollout_signals(agent, z, acts, 1.0)
	s2, _ = rp.plan_signals(agent, z[0], acts, 1.0)
	assert set(s1) == set(s2) and all(torch.equal(s1[k], s2[k]) for k in s1)


def test_calibration_and_every_arm(setup):
	agent, lba = setup
	env = FakeEnv()
	sim = SimSnapshot(env)
	_, cal = mon.run_arm(agent, env, sim, mon.Arm("calibration", "H12", replay=True), lba, {}, 3, 100)
	assert cal["E"].shape == (3 * env.T, lba["L"]) and not torch.isnan(cal["E"]).any()
	assert cal["det_err"].max() < 1e-5   # the snapshot reproduces the real step
	th = mon.calibrate_thresholds(cal)
	assert set(th) == {"LBA", "A", "B", "E", "D", "M", "oracle"} and set(th["LBA"]) == {str(r) for r in mon.RATES}
	arms = mon.arms_for(has_ensemble=True)
	names = [a.name for a in arms]
	assert len(names) == len(set(names)) and "H12+D@0.25" in names and "H12+oracle@0.25" in names
	assert "H12+D@0.25" not in [a.name for a in mon.arms_for(has_ensemble=False)]
	for arm in arms:
		rows, rec = mon.run_arm(agent, env, sim, arm, lba, th, 2, 0)
		assert len(rows) == 2 and all(r["steps"] == env.T for r in rows)
		if arm.base == "H12":
			assert len(rec["alarm"]) == 2 * env.T and rec["steps"]["LBA"].shape == (2 * env.T, lba["L"])
			if arm.monitor is None:
				assert not rec["alarm"].any()
			if arm.replay:
				assert not torch.isnan(rec["E"]).any()
		else:
			assert len(rec["alarm"]) == 0
		assert all(0 <= r["warning_rate"] <= 1 for r in rows)
	# a monitor with a threshold below every score warns at every step; above every score never
	low = {k: {r: dict(thr=-1e9, tie=0.0) for r in v} for k, v in th.items()}
	high = {k: {r: dict(thr=1e9, tie=1.0) for r in v} for k, v in th.items()}
	assert mon.run_arm(agent, env, sim, mon.Arm("x", "H12", "LBA", 0.25), lba, low, 1, 0)[1]["alarm"].all()
	assert not mon.run_arm(agent, env, sim, mon.Arm("x", "H12", "LBA", 0.25), lba, high, 1, 0)[1]["alarm"].any()


def test_main_resumes_and_writes_outputs(setup, tmp_path, monkeypatch):
	agent, _ = setup
	torch.save(_episodes(), tmp_path / "episodes_grounded.pt")
	monkeypatch.setattr(rp, "load_agent", lambda *a, **k: agent)
	import src.envs.maniskill3 as ms
	monkeypatch.setattr(ms, "ManiSkill3Env", lambda task, seed: FakeEnv(T=8))
	out = tmp_path / "mon"
	argv = ["--model", "x", "--run", "grounded", "--task", "PushCube-v1", "--episodes_file",
	        str(tmp_path / "episodes_grounded.pt"), "--out", str(out), "--episodes", "2", "--cal_episodes", "2"]
	with pytest.raises(SystemExit) as e:          # time limit hit after the first arm -> checkpoint exit
		mon.main(argv + ["--max_hours", "0"])
	assert e.value.code == 85 and len(list((out / "arms").glob("*.pt"))) == 1
	mon.main(argv)                                # restart: reuses the LBA, thresholds and finished arm
	summary = json.loads((out / "summary.json").read_text())
	assert set(summary) == {a.name for a in mon.arms_for(True)}
	lines = (out / "episodes.csv").read_text().strip().splitlines()
	assert len(lines) == 1 + 2 * len(summary) and lines[0].startswith("arm,episode,seed,success")
	# the report pools robots (directories or tarballs) and states every pre-registered claim
	import shutil
	import tarfile
	from src.tools import monitor_report as mr
	shutil.copytree(out, tmp_path / "mon_robot2")
	with tarfile.open(tmp_path / "mon_robot3.tar.gz", "w:gz") as tf:
		tf.add(out, arcname="mon")
	text = mr.run(mr.load([str(out), str(tmp_path / "mon_robot2"), str(tmp_path / "mon_robot3.tar.gz")]))
	for key in ("W1", "W2", "W3", "W4", "Success rate per arm", "oracle"):
		assert key in text
	assert "3 robots" in text


def test_thresholds_hit_the_rate_even_with_tied_scores():
	g = torch.Generator().manual_seed(0)
	# 60% of decisions score exactly 1.0 (a saturated probability), the rest are spread out
	s = torch.cat([torch.ones(600), torch.rand(400, generator=g) * 0.9])[torch.randperm(1000, generator=g)]
	rec = dict(steps={"LBA": s.view(-1, 1)}, E=torch.rand(1000, 1, generator=g))
	th = mon.calibrate_thresholds(rec)
	for r in mon.RATES:
		t = th["LBA"][str(r)]
		rate = sum(mon.warns(float(x), t, g) for x in s) / len(s)
		assert abs(rate - r) < 0.05, (r, rate, t)
		rate_e = sum(mon.warns(float(x), th["oracle"][str(r)], g) for x in rec["E"][:, 0]) / 1000
		assert abs(rate_e - r) < 0.03
