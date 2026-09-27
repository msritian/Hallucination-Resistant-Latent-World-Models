import torch

from src.auditor.elvis import RunningNorm, elvis_lambdas, elvis_return

H, N = 5, 3
G = 0.95


def _data():
	g = torch.Generator().manual_seed(0)
	return torch.randn(H, N, 1, generator=g), torch.randn(H + 1, N, 1, generator=g)


def test_lambda_one_is_standard_score():
	r, v = _data()
	score = elvis_return(r, v, torch.ones(H + 1, N, 1), G)
	disc = G ** torch.arange(H).view(H, 1, 1)
	assert torch.allclose(score, (disc * r).sum(0) + G ** H * v[-1], atol=1e-6)


def test_lambda_zero_is_one_step_bootstrap():
	r, v = _data()
	score = elvis_return(r, v, torch.zeros(H + 1, N, 1), G)
	assert torch.allclose(score, r[0] + G * v[1], atol=1e-6)


def test_norm_and_lambdas_stay_in_range():
	norm = RunningNorm()
	for scale in [1.0, 10.0, 1e4]:
		ucb = torch.randn(H + 1, N, 1) * scale
		norm.update(ucb)
		n = norm(ucb * 100)
		assert (n >= 0).all() and (n <= 1).all()
		lam = elvis_lambdas(ucb, norm, lambda_min=0.0, lambda_max=1.0)
		assert (lam >= 0).all() and (lam <= 1).all()
