"""Offline analyses of the learned Bellman audit from saved per-step signals (features_<model>.pt dumps).

	python -m src.tools.audit_ablations path/to/feat_dir_or_tarballs ... > docs/meeting/audit_ablations.md

Analyses (each per robot, then averaged), all with the same tree settings as `learned_audits`:
	1. Signal ablations, including a "no Bellman" control (value level, predicted reward and step only).
	2. Data efficiency: audit fitted on 5 / 10 / 15 / 25 calibration episodes.
	3. Transfer: audit fitted on one robot and applied to another (same task, other task), raw or with the target's
	   unlabeled features standardised.
	4. Statistics: paired episode bootstrap of ours minus each published ensemble signal (D, M, used as is), per robot.
	5. Where the losses come from: AUROC on windows before vs after the task was already solved.
Labels: the simulator answer key (imagined vs real return; positive > P99, negative <= P90 of calibration step 0).
"""
import math
import sys
import tarfile
import tempfile
from pathlib import Path

import torch

from src.preflight.metrics import stratified_auroc

CRITIC = ["A", "Aa", "Ac", "Ab", "Acb", "P", "B", "E", "At", "Ao", "A2", "A3", "A5", "A8"]
ABLATIONS = {
	"Full audit (ours)": (CRITIC, True),
	"One-step δ only": (["A"], False),
	"Anchored δ only": (["Aa"], False),
	"One-step + anchored δ": (["A", "Aa"], False),
	"All residuals, no head disagreement": ([k for k in CRITIC if k not in ("P", "B")], True),
	"All residuals, no multi-step δ": ([k for k in CRITIC if k not in ("A2", "A3", "A5", "A8")], True),
	"All residuals, no anchored δ": ([k for k in CRITIC if k not in ("Aa", "Ac", "Acb")], True),
	"No Bellman: value, reward, step only": ([], True),
	"No Bellman: critic spread + value, reward, step": (["B", "E"], True),
}


# ---------------------------------------------------------------- data
def load_dumps(paths) -> dict:
	"""{robot: dump} from directories or tarballs containing features_<model>.pt."""
	out = {}
	for p in map(Path, paths):
		files = []
		if p.is_dir():
			files = list(p.rglob("features_*.pt"))
		elif p.suffixes[-2:] == [".tar", ".gz"]:
			tmp = Path(tempfile.mkdtemp())
			with tarfile.open(p) as tf:
				members = [m for m in tf.getmembers() if Path(m.name).name.startswith("features_")]
				tf.extractall(tmp, members=members)
			files = list(tmp.rglob("features_*.pt"))
		else:
			files = [p]
		base = p.name.replace(".tar.gz", "").replace("feat_", "")
		for f in files:
			critic = f.stem.replace("features_", "")
			out[f"{base}/{critic}"] = torch.load(f, weights_only=False)
	return out


def task_of(robot: str) -> str:
	return "Push" if "push" in robot else "StackCube" if "stack" in robot else "YCB" if "ycb" in robot else "Peg" if "peg" in robot else "?"


def labels(part: dict, E_ref: torch.Tensor):
	"""(positive mask, kept mask, sudden mask, gradual mask) over [L, M] with thresholds from the calibration error."""
	E = part["E"]
	e1 = E_ref[0].flatten().float()
	clean, hall = torch.quantile(e1, 0.9).item(), torch.quantile(e1, 0.99).item()
	inc = lambda v: torch.cat([v[:1], v[1:] - v[:-1]], 0)
	big = torch.quantile(inc(E_ref).flatten().float(), 0.99).item()
	any_big = torch.cummax((inc(E) > big).int(), 0).values.bool()
	pos, neg = E > hall, E <= clean
	return pos, pos | neg, pos & any_big, pos & ~any_big


def features(part: dict, keys, extras: bool, L: int, episodes=None) -> torch.Tensor:
	cols = []
	for k in keys:
		if k in part["signals"]:
			x = part["signals"][k][:L].float()
			cols += [x, torch.cummax(x, 0).values]
	if extras:
		cols += [part["v_hat"][:L].float(), part["r_hat"][:L].float()]
	M = part["E"].shape[1]
	cols.append(torch.arange(L, dtype=torch.float32).view(-1, 1).expand(L, M))
	X = torch.stack(cols, -1)
	if episodes is not None:
		X = X[:, episodes]
	return X


def gbt():
	from sklearn.ensemble import HistGradientBoostingClassifier
	return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0,
	                                      class_weight="balanced", random_state=0)


def fit(dump: dict, keys, extras, episode_subset=None):
	cal = dump["cal"]
	L = cal["E"].shape[0]
	pos, keep, _, _ = labels(cal, cal["E"])
	cols = torch.ones(cal["E"].shape[1], dtype=torch.bool)
	if episode_subset is not None:
		cols = torch.isin(cal["episode"], episode_subset)
	X = features(cal, keys, extras, L)[:, cols]
	F = X.shape[-1]
	X = X.reshape(-1, F)
	y, k = pos[:, cols].reshape(-1), keep[:, cols].reshape(-1)
	if y[k].sum() < 5 or (~y[k]).sum() < 5:
		return None
	return gbt().fit(X[k].numpy(), y[k].numpy())


def score(model, dump: dict, keys, extras, standardize_with=None) -> torch.Tensor:
	ev = dump["ev"]
	L = ev["E"].shape[0]
	X = features(ev, keys, extras, L)
	F = X.shape[-1]
	if standardize_with is not None:  # target robot's own unlabeled calibration features
		ref = features(standardize_with["cal"], keys, extras, L).reshape(-1, F)
		X = (X - ref.mean(0)) / (ref.std(0) + 1e-6)
	return torch.as_tensor(model.predict_proba(X.reshape(-1, F).numpy())[:, 1], dtype=torch.float32).reshape(L, -1)


def standardized_fit(dump, keys, extras):
	cal = dump["cal"]
	L = cal["E"].shape[0]
	X = features(cal, keys, extras, L)
	F = X.shape[-1]
	X = X.reshape(-1, F)
	X = (X - X.mean(0)) / (X.std(0) + 1e-6)
	pos, keep, _, _ = labels(cal, cal["E"])
	y, k = pos.reshape(-1), keep.reshape(-1)
	if y[k].sum() < 5:
		return None
	return gbt().fit(X[k].numpy(), y[k].numpy())


def aurocs(s: torch.Tensor, dump: dict, mask_extra=None) -> dict:
	ev = dump["ev"]
	pos, keep, sudden, gradual = labels(ev, dump["cal"]["E"])
	steps = torch.arange(s.shape[0]).view(-1, 1).expand_as(s)
	out = {}
	for name, p in [("all", pos), ("gradual", gradual), ("sudden", sudden)]:
		k = (p | (keep & ~pos))
		if mask_extra is not None:
			k = k & mask_extra
		out[name] = stratified_auroc(s[k], p[k], steps[k]) if p[k].any() and (~p[k]).any() else float("nan")
	return out


def mean(xs):
	xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
	return sum(xs) / len(xs) if xs else float("nan")


def fmt(v):
	return "–" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.3f}"


def paired_bootstrap(so: torch.Tensor, se: torch.Tensor, d: dict, n: int = 300):
	"""(AUROC ours, AUROC rival, difference, 2.5%, 97.5%) with evaluation episodes resampled jointly for both scores."""
	pos, keep, _, _ = labels(d["ev"], d["cal"]["E"])
	steps = torch.arange(so.shape[0]).view(-1, 1).expand_as(so)
	a_o = stratified_auroc(so[keep], pos[keep], steps[keep])
	a_r = stratified_auroc(se[keep], pos[keep], steps[keep])
	eps = d["ev"]["episode"]
	uniq = torch.unique(eps)
	g = torch.Generator().manual_seed(0)
	boots = []
	for _ in range(n):
		pick = uniq[torch.randint(len(uniq), (len(uniq),), generator=g)]
		cols = torch.cat([torch.nonzero(eps == e).flatten() for e in pick])
		k, p, st = keep[:, cols], pos[:, cols], steps[:, cols]
		if p[k].any() and (~p[k]).any():
			boots.append(stratified_auroc(so[:, cols][k], p[k], st[k]) - stratified_auroc(se[:, cols][k], p[k], st[k]))
	b = torch.tensor(boots)
	return a_o, a_r, a_o - a_r, torch.quantile(b, 0.025).item(), torch.quantile(b, 0.975).item()


# ---------------------------------------------------------------- analyses
def run(dumps: dict) -> str:
	robots = sorted(dumps)
	grounded = [r for r in robots if r.endswith("/grounded")]
	L = []
	L += ["# Learned Bellman audit: ablations, data needs, transfer and statistics", "",
	      f"{len(robots)} robots ({len(grounded)} with our critic). Simulator labels; AUROC (0.5 guessing, 1.0 perfect). "
	      "All audits use the same gradient-boosted tree settings as the main results.", ""]

	# 1. ablations
	res = {name: {r: aurocs(score(m, dumps[r], keys, ex), dumps[r]) if (m := fit(dumps[r], keys, ex)) else None
	              for r in robots} for name, (keys, ex) in ABLATIONS.items()}
	L += ["## 1. Which signals matter (average over all robots)", "",
	      "| Audit reads… | All | Gradual | Sudden |", "| :-- | :-: | :-: | :-: |"]
	for name in ABLATIONS:
		vals = [mean([res[name][r][c] for r in robots if res[name][r]]) for c in ("all", "gradual", "sudden")]
		L.append(f"| {name} | " + " | ".join(fmt(v) for v in vals) + " |")
	L.append("")

	# 2. data efficiency
	L += ["## 2. How many episodes the audit needs (average over all robots, all hallucinations)", "",
	      "| Calibration episodes | AUROC |", "| :-: | :-: |"]
	keys, ex = ABLATIONS["Full audit (ours)"]
	for n in (5, 10, 15, 25):
		vals = []
		for r in robots:
			eps = torch.unique(dumps[r]["cal"]["episode"])[:n]
			m = fit(dumps[r], keys, ex, episode_subset=eps)
			if m is not None:
				vals.append(aurocs(score(m, dumps[r], keys, ex), dumps[r])["all"])
		L.append(f"| {n} | {fmt(mean(vals))} |")
	L.append("")

	# 3. transfer
	L += ["## 3. Transfer: audit fitted on one robot, used on another (all hallucinations)", "",
	      "| Fitted on | Raw features | Standardised with the target's unlabeled data |", "| :-- | :-: | :-: |"]
	raw_models = {r: fit(dumps[r], keys, ex) for r in robots}
	std_models = {r: standardized_fit(dumps[r], keys, ex) for r in robots}
	groups = {"the same robot": [], "another robot, same task": [], "a robot of another task": []}
	groups_std = {k: [] for k in groups}
	for src in robots:
		for tgt in robots:
			if src.split("/")[1] != tgt.split("/")[1]:
				continue  # keep critic type fixed (ours→ours, standard→standard)
			g = "the same robot" if src == tgt else "another robot, same task" if task_of(src) == task_of(tgt) else "a robot of another task"
			if raw_models[src] is not None:
				groups[g].append(aurocs(score(raw_models[src], dumps[tgt], keys, ex), dumps[tgt])["all"])
			if std_models[src] is not None:
				groups_std[g].append(aurocs(score(std_models[src], dumps[tgt], keys, ex, standardize_with=dumps[tgt]), dumps[tgt])["all"])
	for g in groups:
		L.append(f"| {g} | {fmt(mean(groups[g]))} ({len(groups[g])} pairs) | {fmt(mean(groups_std[g]))} |")
	L.append("")

	# 4. statistics vs the published ensemble signals, used as is (paired episode bootstrap)
	for rival, rlabel in [("M", "MOBILE-style"), ("D", "Dynamics ensemble")]:
		L += [f"## 4{'ab'[rival == 'D']}. Ours minus {rlabel} (as published), per robot (paired bootstrap over episodes, all hallucinations)", "",
		      f"| Robot | Ours | {rlabel} | Difference | 95% interval | Significant? |", "| :-- | :-: | :-: | :-: | :-: | :-: |"]
		sig_count, worse_count, diffs = 0, 0, []
		for r in grounded:
			d = dumps[r]
			mo = fit(d, CRITIC, True)
			if mo is None or rival not in d["ev"]["signals"]:
				continue
			so = score(mo, d, CRITIC, True)
			se = d["ev"]["signals"][rival][:so.shape[0]].float()
			a_o, a_r, base, lo, hi = paired_bootstrap(so, se, d)
			sig_count += lo > 0
			worse_count += hi < 0
			diffs.append(base)
			L.append(f"| {r.split('/')[0]} | {fmt(a_o)} | {fmt(a_r)} | {base:+.3f} | [{lo:+.3f}, {hi:+.3f}] | "
			         f"{'yes' if lo > 0 else ('no (rival better)' if hi < 0 else 'no')} |")
		L += ["", f"Ours significantly better on {sig_count} of {len(diffs)} robots, significantly worse on {worse_count}; "
		      f"mean difference {mean(diffs):+.3f}.", ""]

	# 5. losses: before vs after the task is solved
	L += ["## 5. Before vs after the task is already solved (robots with our critic, all hallucinations)", "",
	      "A window counts as 'after' if the real episode had already reached success at the window's start.", "",
	      "| Robot | Ours: before | Ours: after | Dynamics ensemble: before | Dynamics ensemble: after |", "| :-- | :-: | :-: | :-: | :-: |"]
	for r in grounded:
		d = dumps[r]
		ev = d["ev"]
		succ = torch.cummax(d["success"].float(), 1).values  # [E, T]
		solved = succ[ev["episode"], ev["start"]].bool()      # [M]
		L_ = ev["E"].shape[0]
		after = solved.view(1, -1).expand(L_, -1)
		mo = fit(d, CRITIC, True)
		if mo is None:
			continue
		so = score(mo, d, CRITIC, True)
		sd = ev["signals"]["D"][:L_].float() if "D" in ev["signals"] else None
		a1, a2 = aurocs(so, d, ~after)["all"], aurocs(so, d, after)["all"]
		d1 = aurocs(sd, d, ~after)["all"] if sd is not None else float("nan")
		d2 = aurocs(sd, d, after)["all"] if sd is not None else float("nan")
		L.append(f"| {r.split('/')[0]} | {fmt(a1)} | {fmt(a2)} | {fmt(d1)} | {fmt(d2)} |")
	L.append("")
	return "\n".join(L)


def main(argv=None):
	paths = argv or sys.argv[1:]
	print(run(load_dumps(paths)))


if __name__ == "__main__":
	main()
