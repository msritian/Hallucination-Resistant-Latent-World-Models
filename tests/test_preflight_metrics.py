import torch

from src.preflight.analysis import natural_labels, normalize, replay_scores
from src.preflight.metrics import auroc, bootstrap_auroc, go_no_go
from src.auditor.calibrator import Calibration


def test_auroc_basic_cases():
	assert auroc(torch.tensor([0.1, 0.2, 0.8, 0.9]), torch.tensor([0, 0, 1, 1])) == 1.0
	assert auroc(torch.tensor([0.9, 0.8, 0.2, 0.1]), torch.tensor([0, 0, 1, 1])) == 0.0
	assert auroc(torch.tensor([0.5, 0.5, 0.5, 0.5]), torch.tensor([0, 1, 0, 1])) == 0.5
	assert auroc(torch.tensor([0.1, 0.4, 0.35, 0.8]), torch.tensor([0, 0, 1, 1])) == 0.75
	assert auroc(torch.tensor([1.0, 2.0]), torch.tensor([1, 1])) != auroc(torch.tensor([1.0, 2.0]), torch.tensor([1, 1]))


def test_auroc_matches_pairwise_definition_with_ties():
	g = torch.Generator().manual_seed(0)
	scores = torch.randint(0, 5, (200,), generator=g).float()
	labels = torch.rand(200, generator=g) > 0.6
	pos, neg = scores[labels], scores[~labels]
	pairwise = ((pos[:, None] > neg[None, :]).double() + 0.5 * (pos[:, None] == neg[None, :]).double()).mean().item()
	assert abs(auroc(scores, labels) - pairwise) < 1e-9


def test_bootstrap_ci_brackets_point():
	g = torch.Generator().manual_seed(1)
	labels = torch.rand(400, generator=g) > 0.5
	scores = labels.float() + torch.randn(400, generator=g)
	point, lo, hi = bootstrap_auroc(scores, labels, n_boot=200)
	assert lo <= point <= hi and 0.6 < point < 0.9


def test_go_no_go_rule():
	good = {"A": {"type2": 0.9, "type4": 0.9}, "B": {"type4": 0.8}, "E": {"type4": 0.82}}
	assert go_no_go(good)["decision"] == "GO"
	close = {"A": {"type2": 0.9, "type4": 0.85}, "B": {"type4": 0.84}, "E": {"type4": 0.7},
	         "C": {"type2": 0.9, "type4": 0.95}}
	r = go_no_go(close)
	assert r["decision"] == "GO (combined)" and r["signal"] == "C"
	weak = {"A": {"type2": 0.7, "type4": 0.9}, "B": {"type4": 0.5}, "E": {"type4": 0.5}, "C": {"type2": 0.75, "type4": 0.9}}
	assert go_no_go(weak)["decision"] == "NO-GO"


def test_normalize_scores_and_labels():
	H, M = 4, 5
	sig = {"A": torch.ones(H, M) * 2, "B": torch.ones(H, M)}
	taus = {k: Calibration(tau=torch.full((H,), 2.0), clean_counts=torch.zeros(H), carried=torch.zeros(H, dtype=torch.bool))
	        for k in sig}
	s = normalize(sig, taus)
	assert torch.allclose(s["A"], torch.ones(H, M)) and torch.allclose(s["C"], torch.ones(H, M))
	s["A"][3, 0] = 7.0
	assert replay_scores(s)["A"][0] == 7.0
	err = torch.tensor([[0.1, 0.5, 2.0]])
	labels, keep = natural_labels(err, eps_clean=0.2, eps_hall=1.0)
	assert labels.tolist() == [[False, False, True]] and keep.tolist() == [[True, False, True]]


def test_stratified_auroc_ignores_between_strata_differences():
	from src.preflight.metrics import stratified_auroc
	# stratum 0: early step, low scores, mostly negatives; stratum 1: late step, high scores, mostly positives.
	scores = torch.tensor([0.1, 0.2, 0.3, 0.9, 1.0, 1.1])
	labels = torch.tensor([0, 0, 1, 0, 1, 1]).bool()
	strata = torch.tensor([0, 0, 0, 1, 1, 1])
	assert stratified_auroc(scores, labels, strata) == 1.0
	# pooled AUROC is inflated by the step effect when within-step ranking is random
	scores2 = torch.tensor([0.3, 0.1, 0.2, 1.1, 0.9, 1.0])
	assert stratified_auroc(scores2, labels, strata) < auroc(scores2, labels)
