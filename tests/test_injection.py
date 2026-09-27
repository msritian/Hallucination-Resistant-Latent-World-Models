import torch

from src.auditor.injection import goal_interpolation, simnorm, structured_noise, value_neutral_match


def _latents(n, d=32, seed=0):
	g = torch.Generator().manual_seed(seed)
	return simnorm(torch.randn(n, d, generator=g))


def _on_simplex(z, dim=8):
	groups = z.view(*z.shape[:-1], -1, dim)
	return (groups >= 0).all() and torch.allclose(groups.sum(-1), torch.ones(groups.shape[:-1]), atol=1e-5)


def test_structured_noise_stays_on_simplex_and_changes_latent():
	z = _latents(10)
	for sigma in [0.5, 1.0, 2.0]:
		out = structured_noise(z, sigma, generator=torch.Generator().manual_seed(1))
		assert _on_simplex(out)
		assert not torch.allclose(out, z)


def test_goal_interpolation_stays_on_simplex():
	z, goal = _latents(10, seed=0), _latents(10, seed=1)
	for alpha in [0.1, 0.5, 1.0]:
		assert _on_simplex(goal_interpolation(z, goal, alpha))
	assert torch.allclose(goal_interpolation(z, goal, 1.0), goal)


def test_value_neutral_match_respects_tolerance_and_distance():
	z_t, z_pool = _latents(4, seed=0), _latents(50, seed=1)
	q_t, q_pool = torch.tensor([0.0, 1.0, 2.0, 100.0]), torch.linspace(0, 3, 50)
	idx = value_neutral_match(q_t, z_t, q_pool, z_pool, q_tol=0.05, min_dist=0.1)
	assert idx[3] == -1
	for i in range(3):
		assert idx[i] >= 0
		assert (q_t[i] - q_pool[idx[i]]).abs() < 0.05
		assert (z_t[i] - z_pool[idx[i]]).norm() > 0.1
