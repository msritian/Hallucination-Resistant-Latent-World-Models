import math

import torch

from src.auditor.trust import trust_weighted_return, trust_weighted_return_backward, trust_weights

H, N = 6, 4
G = 0.95


def _data(seed=0):
	g = torch.Generator().manual_seed(seed)
	return (torch.randn(H, N, 1, generator=g), torch.randn(H, N, 1, generator=g), torch.randn(N, 1, generator=g))


def test_full_trust_equals_standard_score():
	r, q, qT = _data()
	score, omega = trust_weighted_return(r, q, qT, torch.ones(H, N, 1), G)
	disc = G ** torch.arange(H).view(H, 1, 1)
	assert torch.allclose(score, (disc * r).sum(0) + G ** H * qT, atol=1e-6)
	assert torch.all(omega == 1)


def test_hard_loss_at_k_equals_truncation():
	r, q, qT = _data()
	k = 3
	w = torch.ones(H, N, 1)
	w[k] = 0
	score, _ = trust_weighted_return(r, q, qT, w, G)
	disc = G ** torch.arange(k).view(k, 1, 1)
	assert torch.allclose(score, (disc * r[:k]).sum(0) + G ** k * q[k], atol=1e-6)


def test_distrust_at_step0_gives_q_of_first_action():
	r, q, qT = _data()
	w = torch.ones(H, N, 1)
	w[0] = 0
	score, _ = trust_weighted_return(r, q, qT, w, G)
	assert torch.allclose(score, q[0], atol=1e-6)


def test_forward_equals_backward_lambda_return():
	r, q, qT = _data(1)
	w = torch.rand(H, N, 1, generator=torch.Generator().manual_seed(2))
	fwd, _ = trust_weighted_return(r, q, qT, w, G)
	assert torch.allclose(fwd, trust_weighted_return_backward(r, q, qT, w, G), atol=1e-5)


def test_worked_example_43_25_7():
	r = torch.tensor([1, 1, 1, 10, 10, 10.0]).view(6, 1, 1)
	q = torch.zeros(6, 1, 1)
	q[3] = 4
	qT = torch.tensor([[10.0]])
	for trust, expected in [(1.0, 43.0), (0.5, 25.0), (0.0, 7.0)]:
		w = torch.ones(6, 1, 1)
		w[3] = trust
		score, _ = trust_weighted_return(r, q, qT, w, 1.0)
		assert abs(score.item() - expected) < 1e-5


def test_trust_weights_values_and_extremes():
	s = torch.tensor([0.5, 1.0, 2.0, 4.0, 1e6, float("inf"), float("nan")])
	w = trust_weights(s)
	assert torch.allclose(w[:4], torch.tensor([1.0, 1.0, math.exp(-1), math.exp(-3)]), atol=1e-6)
	assert torch.isfinite(w).all() and (w >= 0).all() and (w <= 1).all()
	assert torch.equal(trust_weights(s, kappa=float("inf")), torch.tensor([1.0, 1, 0, 0, 0, 0, 0]))
	r, q, qT = _data()
	score, _ = trust_weighted_return(r, q, qT, trust_weights(torch.full((H, N, 1), float("inf"))), G)
	assert torch.isfinite(score).all()
