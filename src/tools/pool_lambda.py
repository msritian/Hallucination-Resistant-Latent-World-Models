"""Pool `--preset lambda` evaluations across models and test the pre-registered claims C1-C3.

Each input is an evaluation result (a directory with episodes.csv, or its .tar.gz from the cluster). Differences are
paired by episode seed within each model, averaged within each model, then averaged over models. 95% CIs resample
episodes within each model (stratified bootstrap).

	C1  const0.8 - ELVIS          at H12 and at H24     pass if mean >= -0.03 at both
	C2  const0.8 H12 - standard_H3                       pass if CI lower bound > 0
	C3  const0.8 + penalty 1.0 - const0.8, H12 and H24 pooled   pass if CI lower bound > 0

Example:
	python -m src.tools.pool_lambda results/eval_seed_grounded_stack_s20_lambda.tar.gz ...
"""
import argparse
import csv
import io
import json
import tarfile
from pathlib import Path

import torch

ELVIS = "elvis_H{H}_lmin0.0_lmax1.0_b1.0"
CONST = "elvis_H{H}_const0.8"
PEN = "elvis_H{H}_const0.8_pen{b}"
H3 = "standard_H3"


def read_episodes(path: str) -> dict:
	"""{arm: {seed: success}} from a result directory or tarball."""
	p = Path(path)
	if p.is_dir():
		text = (p / "episodes.csv").read_text() if (p / "episodes.csv").exists() else (p / "evalout" / "episodes.csv").read_text()
	else:
		with tarfile.open(p) as tf:
			member = next(m for m in tf.getmembers() if m.name.endswith("episodes.csv"))
			text = tf.extractfile(member).read().decode()
	out = {}
	for row in csv.DictReader(io.StringIO(text)):
		out.setdefault(row["arm"], {})[int(row["seed"])] = float(row["success"])
	return out


def paired(models: dict, pairs: list) -> list:
	"""Per model: tensor of per-episode differences, concatenated over the (a, b) arm pairs; models missing an arm skip."""
	out = []
	for name, eps in models.items():
		diffs = []
		for a, b in pairs:
			if a in eps and b in eps:
				seeds = sorted(set(eps[a]) & set(eps[b]))
				diffs.append(torch.tensor([eps[a][s] - eps[b][s] for s in seeds]))
		if diffs:
			out.append(torch.cat(diffs))
	return out


def pooled_ci(per_model: list, n_boot: int = 5000, seed: int = 0) -> dict:
	"""Mean over models of the within-model mean difference, with a stratified bootstrap 95% CI."""
	if not per_model:
		return dict(mean=float("nan"), ci95=[float("nan")] * 2, models=0)
	g = torch.Generator().manual_seed(seed)
	point = torch.stack([d.mean() for d in per_model]).mean()
	boots = torch.zeros(n_boot)
	for d in per_model:
		boots += d[torch.randint(len(d), (n_boot, len(d)), generator=g)].mean(1)
	boots /= len(per_model)
	return dict(mean=float(point), ci95=[float(torch.quantile(boots, 0.025)), float(torch.quantile(boots, 0.975))],
	            models=len(per_model))


def claims(models: dict) -> dict:
	c1 = {H: pooled_ci(paired(models, [(CONST.format(H=H), ELVIS.format(H=H))])) for H in (12, 24)}
	c2 = pooled_ci(paired(models, [(CONST.format(H=12), H3)]))
	c3 = pooled_ci(paired(models, [(PEN.format(H=H, b=1.0), CONST.format(H=H)) for H in (12, 24)]))
	c3_half = pooled_ci(paired(models, [(PEN.format(H=H, b=0.5), CONST.format(H=H)) for H in (12, 24)]))
	return dict(
		C1=dict(H12=c1[12], H24=c1[24], passed=all(c1[H]["mean"] >= -0.03 for H in (12, 24))),
		C2=dict(**c2, passed=c2["ci95"][0] > 0),
		C3=dict(**c3, passed=c3["ci95"][0] > 0),
		C3_penalty0_5_reported_only=c3_half,
	)


def success_table(models: dict) -> str:
	arms = sorted({a for eps in models.values() for a in eps}, key=lambda a: (a != "stock_tdmpc2_H3", a))
	lines = ["| arm | " + " | ".join(models) + " | mean |", "| :- |" + " :-: |" * (len(models) + 1)]
	for a in arms:
		vals = [sum(models[m][a].values()) / len(models[m][a]) if a in models[m] else float("nan") for m in models]
		ok = [v for v in vals if v == v]
		lines.append(f"| {a} | " + " | ".join(f"{v:.2f}" for v in vals) + f" | {sum(ok) / max(len(ok), 1):.3f} |")
	return "\n".join(lines)


STOCK_ARM = "stock_tdmpc2_H3"


def model_key(name: str) -> tuple:
	"""'seed_grounded_stack_s20' -> ('grounded', 'stack_s20'); run type and (task, seed) key for pairing."""
	for run in ("grounded", "stock"):
		for prefix in (f"seed_{run}_", f"p4_{run}_", f"bk_{run}_", f"p1_{run}_"):
			if name.startswith(prefix):
				return run, name[len(prefix):].replace("stackcube", "stack")
	return None, name


def grounding(models: dict) -> dict:
	"""Grounded vs stock success with TD-MPC2's own planner, per (task, seed) pair and pooled over pairs.

	Different models see the same episode seeds, so differences are paired by seed within each pair."""
	by_key = {}
	for name, eps in models.items():
		run, key = model_key(name)
		if run and STOCK_ARM in eps:
			by_key.setdefault(key, {})[run] = eps[STOCK_ARM]
	pairs, per_pair = [], {}
	for key, runs in sorted(by_key.items()):
		if {"grounded", "stock"} <= set(runs):
			g, s_ = runs["grounded"], runs["stock"]
			seeds = sorted(set(g) & set(s_))
			d = torch.tensor([g[x] - s_[x] for x in seeds])
			pairs.append(d)
			per_pair[key] = dict(grounded=sum(g[x] for x in seeds) / len(seeds), stock=sum(s_[x] for x in seeds) / len(seeds))
	return dict(per_pair=per_pair, pooled_grounded_minus_stock=pooled_ci(pairs))


def pessimism(models: dict) -> dict:
	"""Min-over-heads value vs mean value, and long horizons vs H3, from `--preset pessimism` runs."""
	out = {}
	for label, a, b in [
		("H12_min_vs_mean", "standard_H12_vmin", "standard_H12"),
		("H24_min_vs_mean", "standard_H24_vmin", "standard_H24"),
		("H12min_vs_H3", "standard_H12_vmin", "standard_H3"),
		("H12min_vs_H3min", "standard_H12_vmin", "standard_H3_vmin"),
		("const0.8_H12_min_vs_mean", "elvis_H12_const0.8_vmin", "elvis_H12_const0.8"),
		("const0.8_H12min_vs_H3", "elvis_H12_const0.8_vmin", "standard_H3"),
		("const0.8_H24min_vs_H3", "elvis_H24_const0.8_vmin", "standard_H3"),
	]:
		out[label] = pooled_ci(paired(models, [(a, b)]))
	return out


def load(paths: list, strip: str) -> dict:
	models = {}
	for r in paths:
		name = Path(r).name.replace(".tar.gz", "").replace("eval_", "").replace(strip, "")
		models[name] = read_episodes(r)
	return models


def main(argv=None):
	p = argparse.ArgumentParser()
	p.add_argument("results", nargs="+", help="eval_*_lambda.tar.gz (claims C1-C3) and optionally eval_*_pessimism.tar.gz")
	args = p.parse_args(argv)
	lam = load([r for r in args.results if "_pessimism" not in r], "_lambda")
	pes = load([r for r in args.results if "_pessimism" in r], "_pessimism")
	if lam:
		print("## Lambda preset: success per arm\n")
		print(success_table(lam), "\n")
		print("## Pre-registered claims C1-C3\n")
		print(json.dumps(claims(lam), indent=1), "\n")
		print("## Grounded vs stock (TD-MPC2's own planner)\n")
		print(json.dumps(grounding(lam), indent=1), "\n")
	if pes:
		print("## Pessimism preset: success per arm\n")
		print(success_table(pes), "\n")
		print(json.dumps(pessimism(pes), indent=1))


if __name__ == "__main__":
	main()
