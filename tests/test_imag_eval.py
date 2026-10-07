"""CPU test for Track 1 (mock robot, fake simulator): twin real/imagined episodes, scores, and the report."""
import json

import torch

from src import imag_eval as ie
from src.preflight import run_preflight as rp
from src.tools import imag_eval_report as rep
from tests.test_monitor import FakeEnv, _agent, _episodes


def test_imag_eval_end_to_end(tmp_path, monkeypatch):
	agent = _agent()
	torch.save(_episodes(T=30), tmp_path / "ep.pt")
	monkeypatch.setattr(rp, "load_agent", lambda *a, **k: agent)
	import src.envs.maniskill3 as ms
	def env(task, seed):
		e = FakeEnv(T=26); e.max_episode_steps = 26; return e
	monkeypatch.setattr(ms, "ManiSkill3Env", env)
	ie.main(["--model", "x", "--task", "PushCube-v1", "--episodes_file", str(tmp_path / "ep.pt"), "--out", str(tmp_path / "o"), "--starts", "4"])
	rows = json.loads((tmp_path / "o" / "rows.json").read_text())
	assert len(rows) == 4 * len(ie.POLICIES) and all(k in rows[0] for k in ("LBA", "D", "M", "B", "real_return", "imagined_return"))
	text = rep.report(rep.load([str(tmp_path / "o")]))
	assert "AURC" in text and "Policy evaluation" in text and "oracle" in text
