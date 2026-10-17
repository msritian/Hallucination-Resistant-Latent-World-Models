"""CPU tests for detector-guided replay: scoring windows, the priority set, batch format and content."""
from types import SimpleNamespace

import pytest
import torch

from src.training.audit_replay import AuditReplay, episodes_to_data, imagination_error
from tests.test_monitor import A, OBS, _agent

T, h, B = 40, 3, 16


def _episodes(E=60, seed=0):
	g = torch.Generator().manual_seed(seed)
	eps = []
	for _ in range(E):
		act = torch.cat([torch.full((1, A), float("nan")), torch.rand(T, A, generator=g) * 2 - 1])
		rew = torch.cat([torch.tensor([float("nan")]), torch.rand(T, generator=g)])
		eps.append(dict(obs=torch.randn(T + 1, OBS, generator=g), action=act, reward=rew))
	return eps


class FakeBuffer:
	"""Uniform sampler with Buffer.sample()'s output format; marks its sequences with obs = -100."""

	def sample(self):
		return (torch.full((h + 1, B, OBS), -100.0), torch.zeros(h, B, A), torch.zeros(h, B, 1), torch.zeros(h, B, 1), None)


@pytest.mark.parametrize("mode", ["lba", "D"])
def test_refresh_and_mixed_batches(mode):
	torch.manual_seed(0)
	agent, eps = _agent(), _episodes()
	rep = AuditReplay(FakeBuffer(), mode, frac=0.5, top=0.3, fit_episodes=50)
	assert rep.sample()[0].eq(-100).all()                       # before the first refresh: plain uniform batches
	rep.refresh(agent, eps)
	n_windows = len(eps) * (T - 12 + 1)
	assert rep.stats["windows"] == n_windows and len(rep.windows) == round(0.3 * n_windows)
	obs, action, reward, terminated, task = rep.sample()
	assert obs.shape == (h + 1, B, OBS) and action.shape == (h, B, A) and reward.shape == (h, B, 1)
	assert terminated.shape == (h, B, 1) and task is None
	assert obs[:, B // 2:].eq(-100).all() and not obs[:, :B // 2].eq(-100).any()   # half from the priority set
	# priority sequences are real consecutive slices of stored episodes, inside a flagged 12-step window
	data = episodes_to_data(eps)
	for j in range(B // 2):
		matches = [(e, s) for e in range(len(eps)) for s in range(T - h + 1) if torch.equal(data["obs"][e, s:s + h + 1], obs[:, j])]
		assert len(matches) == 1
		e, s = matches[0]
		assert torch.equal(data["action"][e, s:s + h], action[:, j]) and torch.equal(data["reward"][e, s:s + h], reward[:, j, 0])
		assert any(int(w[0]) == e and int(w[1]) <= s <= int(w[1]) + 12 - h for w in rep.windows)


def test_imagination_error():
	out = imagination_error(_agent(), _episodes(E=4))
	assert len(out["return_error_by_step"]) == 11 and out["return_error_last"] >= 0
