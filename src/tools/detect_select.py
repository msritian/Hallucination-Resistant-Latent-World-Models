"""Apply the pre-registered detection rules (execution_final.md, 2026-09-28 21:40) to preflight re-analysis results.

	select   on development results: the candidate with the highest mean of [mean gradual, mean sudden_natural]
	confirm  on unseen results: the chosen detector's mean AUROC vs every external baseline's mean, per test

Inputs are detect_<name>.tar.gz files (or directories) containing reanalysis/report.json; each yields the grounded
and (if present) stock model.

	python -m src.tools.detect_select select results_local/detect_*.tar.gz
	python -m src.tools.detect_select confirm --detector A+Aa+P results_local/detect_*.tar.gz
"""
import argparse
import json
import math
import tarfile
from pathlib import Path

CANDIDATES = ("A+Aa", "Ab+Acb", "A+A5", "A+Aa+P", "A+A5+P")
BASELINES = ("B", "D", "E", "M")
TESTS = ("gradual", "sudden_natural", "type2_effective")
REAL_TESTS = ("real_gradual", "real_sudden", "real_all")


def load(path: str) -> dict:
	"""{model_label: results} where results[signal][test] = auroc."""
	p = Path(path)
	if p.is_dir():
		report = json.loads(next(p.rglob("report.json")).read_text())
	else:
		with tarfile.open(p) as tf:
			member = next(m for m in tf.getmembers() if m.name.endswith("report.json"))
			report = json.load(tf.extractfile(member))
	name = p.name.replace(".tar.gz", "").replace("detect_", "")
	out = {}
	for model in ("grounded", "stock"):
		if model in report:
			out[f"{name}/{model}"] = {s: {t: v["auroc"] for t, v in r.items()} for s, r in report[model]["results"].items()}
	return out


def mean(values) -> float:
	v = [x for x in values if x is not None and not math.isnan(x)]
	return sum(v) / len(v) if v else float("nan")


def score(models: dict, signal: str, test: str) -> float:
	return mean(m.get(signal, {}).get(test) for m in models.values())


def select(models: dict) -> dict:
	table = {c: dict(gradual=score(models, c, "gradual"), sudden_natural=score(models, c, "sudden_natural")) for c in CANDIDATES}
	for v in table.values():
		v["selection_score"] = (v["gradual"] + v["sudden_natural"]) / 2
	chosen = max(table, key=lambda c: table[c]["selection_score"])
	return dict(candidates=table, chosen=chosen)


def confirm(models: dict, detector: str, tests=TESTS) -> dict:
	out = {}
	for test in tests:
		ours = score(models, detector, test)
		base = {b: score(models, b, test) for b in BASELINES}
		best = max((v for v in base.values() if not math.isnan(v)), default=float("nan"))
		out[test] = dict(ours=ours, baselines=base, passed=bool(ours >= best))
	out["passed_all"] = all(out[t]["passed"] for t in tests)
	out["per_model"] = {m: {t: dict(ours=r.get(detector, {}).get(t), **{b: r.get(b, {}).get(t) for b in BASELINES})
	                        for t in tests} for m, r in models.items()}
	return out


def label_table(models: dict, tests: tuple, signals=("LBA", "LBA_lin", "LENS", "LALL", "A+Aa+P", "Aa", "A", "P") + BASELINES) -> str:
	"""Mean AUROC per signal and test (rows: signals)."""
	lines = ["signal  " + "  ".join(f"{t:>16s}" for t in tests)]
	for sig in signals:
		lines.append(f"{sig:7s} " + "  ".join(f"{score(models, sig, t):16.3f}" for t in tests))
	return "\n".join(lines)


def margins(models: dict, detector: str, rivals: tuple, tests: tuple) -> dict:
	"""Per test and rival: mean AUROC difference over models, a bootstrap 95% CI over models, and models won."""
	import random
	out = {}
	for t in tests:
		out[t] = {}
		for r in rivals:
			diffs = [m[detector][t] - m[r][t] for m in models.values()
			         if t in m.get(detector, {}) and t in m.get(r, {})
			         and not math.isnan(m[detector][t]) and not math.isnan(m[r][t])]
			if not diffs:
				continue
			rng = random.Random(0)
			boots = sorted(sum(rng.choice(diffs) for _ in diffs) / len(diffs) for _ in range(5000))
			out[t][r] = dict(mean_diff=sum(diffs) / len(diffs), ci95=[boots[125], boots[4874]],
			                 models_won=sum(d > 0 for d in diffs), models=len(diffs))
	return out


def main(argv=None):
	p = argparse.ArgumentParser()
	p.add_argument("mode", choices=["select", "confirm", "table", "margins"])
	p.add_argument("results", nargs="+")
	p.add_argument("--detector", default=None)
	p.add_argument("--real", action="store_true", help="use the critic-free (simulator) labels")
	p.add_argument("--tests", default=None, help="comma-separated test names (overrides --real)")
	args = p.parse_args(argv)
	models = {}
	for r in args.results:
		models.update(load(r))
	print(f"{len(models)} models: {', '.join(models)}\n")
	tests = tuple(args.tests.split(",")) if args.tests else (REAL_TESTS if args.real else TESTS)
	if args.mode == "table":
		print(label_table(models, tests))
		return
	if args.mode == "margins":
		res = margins(models, args.detector, ("LENS", "Aa", "M", "B", "D", "E"), tests)
		for t, rs in res.items():
			print(f"\n{t}  ({args.detector} minus rival; 95% CI over models; models won)")
			for r, v in rs.items():
				print(f"  vs {r:5s} {v['mean_diff']:+.3f}  [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}]  {v['models_won']}/{v['models']}")
		return
	result = select(models) if args.mode == "select" else confirm(models, args.detector, tests)
	print(json.dumps(result, indent=1))


if __name__ == "__main__":
	main()
