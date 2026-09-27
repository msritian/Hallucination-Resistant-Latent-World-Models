"""Detection metrics for the Phase 2 preflight (execution_final.md §6, Phase 2)."""
from typing import Optional

import torch


def auroc(scores: torch.Tensor, labels: torch.Tensor) -> float:
	"""Area under the ROC curve (probability a random positive scores above a random negative; ties count half).

	scores, labels: 1-D; labels are bool/0-1. Returns NaN if either class is empty.
	"""
	scores = scores.double().flatten()
	labels = labels.bool().flatten()
	n_pos, n_neg = int(labels.sum()), int((~labels).sum())
	if n_pos == 0 or n_neg == 0:
		return float("nan")
	order = torch.argsort(scores)
	sorted_scores = scores[order]
	ranks = torch.empty_like(scores)
	ranks[order] = torch.arange(1, len(scores) + 1, dtype=torch.float64)
	# average ranks over ties
	unique, inverse, counts = torch.unique(sorted_scores, return_inverse=True, return_counts=True)
	if len(unique) < len(scores):
		cum = torch.cumsum(counts, 0).double()
		avg_rank = cum - (counts.double() - 1) / 2
		ranks[order] = avg_rank[inverse]
	rank_sum_pos = ranks[labels].sum().item()
	return (rank_sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def stratified_auroc(scores: torch.Tensor, labels: torch.Tensor, strata: torch.Tensor) -> float:
	"""AUROC over positive/negative pairs from the same stratum only (e.g. the same imagined step).

	Equals the average of per-stratum AUROCs weighted by their number of pairs, so a signal cannot score well
	just by telling strata apart.
	"""
	num, den = 0.0, 0
	for s in torch.unique(strata):
		m = strata == s
		n_pos, n_neg = int(labels[m].sum()), int((~labels[m].bool()).sum())
		if n_pos and n_neg:
			num += auroc(scores[m], labels[m]) * n_pos * n_neg
			den += n_pos * n_neg
	return num / den if den else float("nan")


def bootstrap_auroc(
	scores: torch.Tensor, labels: torch.Tensor, n_boot: int = 1000, seed: int = 0, groups: Optional[torch.Tensor] = None,
	strata: Optional[torch.Tensor] = None,
) -> tuple[float, float, float]:
	"""AUROC (step-stratified if ``strata`` is given) with a 95% percentile bootstrap CI.

	If ``groups`` is given, whole groups (e.g. episodes) are resampled.
	"""
	metric = (lambda sc, lb, idx: stratified_auroc(sc, lb, strata[idx])) if strata is not None else (lambda sc, lb, idx: auroc(sc, lb))
	point = metric(scores, labels, torch.arange(len(scores)))
	g = torch.Generator().manual_seed(seed)
	if groups is None:
		groups = torch.arange(len(scores))
	uniq = torch.unique(groups)
	members = [torch.nonzero(groups == u).flatten() for u in uniq]
	stats = []
	for _ in range(n_boot):
		pick = torch.randint(len(uniq), (len(uniq),), generator=g)
		idx = torch.cat([members[i] for i in pick.tolist()])
		v = metric(scores[idx], labels[idx], idx)
		if v == v:
			stats.append(v)
	if not stats:
		return point, float("nan"), float("nan")
	s = torch.tensor(stats)
	return point, torch.quantile(s, 0.025).item(), torch.quantile(s, 0.975).item()


def go_no_go(auc: dict, threshold: float = 0.80, margin: float = 0.05) -> dict:
	"""Apply the Phase 2 decision rule to grounded-model AUROCs.

	``auc[signal][test]`` with signals in {A, B, C, E, ...} and tests {"type2", "type4"}.
	A signal qualifies if AUROC >= threshold on type2 and type4, and type4 AUROC >= max(B, E on type4) + margin.
	"""
	ce = max(auc.get("B", {}).get("type4", float("nan")), auc.get("E", {}).get("type4", float("nan")))

	def qualifies(sig):
		a = auc.get(sig, {})
		t2, t4 = a.get("type2", float("nan")), a.get("type4", float("nan"))
		return bool(t2 >= threshold and t4 >= threshold and t4 >= ce + margin)

	if qualifies("A"):
		decision, signal = "GO", "A"
	elif qualifies("C"):
		decision, signal = "GO (combined)", "C"
	else:
		decision, signal = "NO-GO", None
	return dict(decision=decision, signal=signal, critic_ensemble_best_type4=ce,
	            A_qualifies=qualifies("A"), C_qualifies=qualifies("C"))
