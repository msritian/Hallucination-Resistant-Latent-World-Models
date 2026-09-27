import torch

from src.auditor.calibrator import calibrate, clean_prefix_mask, error_thresholds


def test_clean_prefix_is_monotone():
	err = torch.tensor([[0.1, 0.5], [0.9, 0.1], [0.1, 0.1]])
	mask = clean_prefix_mask(err, eps_clean=0.4)
	assert mask.tolist() == [[True, False], [False, False], [False, False]]


def test_percentiles_on_clean_replays_and_carry_forward():
	H, M = 3, 1000
	g = torch.Generator().manual_seed(0)
	signal = torch.rand(H, M, generator=g)
	err = torch.zeros(H, M)
	err[1, 700:] = 1.0      # 300 replays turn dirty at step 1
	err[2, :] = 1.0         # every replay is dirty at step 2
	cal = calibrate(signal, err, eps_clean=0.5, tau_pct=95, min_clean_samples=200)
	assert torch.allclose(cal.tau[0], torch.quantile(signal[0], 0.95))
	assert torch.allclose(cal.tau[1], torch.quantile(signal[1, :700], 0.95))
	assert cal.carried.tolist() == [False, False, True]
	assert cal.tau[2] == cal.tau[1]
	assert cal.clean_counts.tolist() == [1000, 700, 0]


def test_floor_and_error_thresholds():
	cal = calibrate(torch.zeros(2, 300), torch.zeros(2, 300), eps_clean=1.0)
	assert (cal.tau >= 1e-6).all()
	err = torch.arange(1, 101).float().view(1, 100)
	th = error_thresholds(err)
	assert 89 < th.eps_clean < 92 and 98 < th.eps_hall < 100
