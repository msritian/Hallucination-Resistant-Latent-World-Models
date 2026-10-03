import torch

from src.utils.safety import episodes_that_fit, model_is_finite


def test_episodes_that_fit_never_exceeds_capacity():
	# the grounded StackCube case: 9,887 episodes x 51 entries into a 500k buffer
	keep = episodes_that_fit(9_887, 51, 500_000)
	assert keep == 9_803 and keep * 51 <= 500_000
	assert episodes_that_fit(100, 51, 500_000) == 100
	assert episodes_that_fit(9_900, 101, 1_000_000) * 101 <= 1_000_000


def test_model_is_finite_detects_nan():
	m = torch.nn.Linear(3, 3)
	assert model_is_finite(m)
	with torch.no_grad():
		m.weight[0, 0] = float("nan")
	assert not model_is_finite(m)
