"""CPU test of Phase A with a mock robot and fake environment (all shift kinds except mass, which needs ManiSkill)."""
import json

import torch

from src import shift as sh
from src.preflight import run_preflight as rp
from tests.test_monitor import A, OBS, FakeEnv, _agent, _episodes


def test_phase_a_runs(tmp_path, monkeypatch):
	agent = _agent()
	agent.act = lambda obs, t0=False, eval_mode=True: torch.tanh(obs[:A])
	torch.save(_episodes(T=30), tmp_path / "ep.pt")
	monkeypatch.setattr(rp, "load_agent", lambda *a, **k: agent)
	import src.envs.maniskill3 as ms
	monkeypatch.setattr(ms, "ManiSkill3Env", lambda task, seed: FakeEnv(T=20))
	sh.main(["--model", "x", "--task", "PushCube-v1", "--episodes_file", str(tmp_path / "ep.pt"), "--out", str(tmp_path / "o"),
	         "--conditions", "none,gain:0.5,bias:0.3,obs:0.5", "--episodes", "4"])
	s = json.loads((tmp_path / "o" / "summary.json").read_text())
	assert set(s) == {"none", "gain:0.5", "bias:0.3", "obs:0.5"} and "obs_dims" in s["obs:0.5"]["info"]
	for c in ("gain:0.5", "bias:0.3", "obs:0.5"):
		d = s[c]["detection"]
		assert set(d) >= {"LBA", "D", "M", "B", "PE1", "PE"} and all(0 <= v["auroc"] <= 1 for v in d.values())
