"""CPU test for Track 2 (Horde) with a mock robot: heads train, all three tests produce numbers."""
import json

import torch

from src import horde as hd
from src.preflight import run_preflight as rp
from tests.test_monitor import A, OBS, _agent, _episodes


def test_horde_end_to_end(tmp_path, monkeypatch):
	agent = _agent()
	monkeypatch.setattr(rp, "load_agent", lambda *a, **k: agent)
	g = torch.Generator().manual_seed(1)
	E, T = 40, 30
	ck = dict(episodes=dict(obs=torch.randn(E, T + 1, OBS, generator=g),
	                        action=torch.cat([torch.full((E, 1, A), float("nan")), torch.rand(E, T, A, generator=g) * 2 - 1], 1),
	                        reward=torch.rand(E, T + 1, generator=g)))
	torch.save(ck, tmp_path / "ck.pt")
	torch.save(_episodes(E=12, T=30), tmp_path / "ep.pt")
	hd.main(["--model", "x", "--task", "PushCube-v1", "--train_ckpt", str(tmp_path / "ck.pt"), "--train_episodes", "30",
	         "--episodes_file", str(tmp_path / "ep.pt"), "--updates", "50", "--out", str(tmp_path / "o")])
	r = json.loads((tmp_path / "o" / "horde.json").read_text())
	assert {"per_signal_auroc_mean", "localisation_top1", "reward_free_auroc", "reward_audit_auroc", "combined_auroc"} <= set(r)
	assert r["localisation_chance"] == 1 / OBS
