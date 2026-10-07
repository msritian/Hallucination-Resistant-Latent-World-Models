"""Analysis for Track 1 (docs/preregistration_imag_eval.md).

	python -m src.tools.imag_eval_report results/ie_*.tar.gz > docs/imag_eval_results.md

Per robot: (1) Spearman between each detector's episode score and the imagination gap |R_hat - R|; (2) risk-coverage:
mean gap of the episodes each detector trusts most (lowest score) at 25/50/75% coverage, and the area under that curve
(AURC; random selection = mean gap); (3) policy evaluation: per policy, the imagined value estimated from its most-trusted
50% of episodes vs its real value (all episodes): Spearman over policies and mean absolute error, vs using all episodes.
"""
import json
import math
import sys
import tarfile
import tempfile
from pathlib import Path

DET = ("LBA", "D", "M", "B")


def load(paths):
	out = {}
	for p in map(Path, paths):
		if p.suffixes[-2:] == [".tar", ".gz"]:
			d = Path(tempfile.mkdtemp())
			with tarfile.open(p) as tf:
				tf.extractall(d)
			p2 = next(d.rglob("rows.json"))
		else:
			p2 = p if p.name == "rows.json" else next(p.rglob("rows.json"))
		out[p.name.replace(".tar.gz", "")] = json.loads(p2.read_text())
	return out


def rank(x):
	order = sorted(range(len(x)), key=lambda i: x[i])
	r = [0.0] * len(x)
	for k, i in enumerate(order):
		r[i] = k
	return r


def spearman(x, y):
	rx, ry = rank(x), rank(y)
	n = len(x)
	mx, my = sum(rx) / n, sum(ry) / n
	cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
	vx = math.sqrt(sum((a - mx) ** 2 for a in rx)); vy = math.sqrt(sum((b - my) ** 2 for b in ry))
	return cov / (vx * vy) if vx and vy else float("nan")


def analyse(rows):
	gap = [abs(r["imagined_return"] - r["real_return"]) for r in rows]
	res = {"mean_gap": sum(gap) / len(gap)}
	dets = [d for d in DET if all(not math.isnan(r.get(d, float("nan"))) for r in rows)] + ["oracle"]
	for d in dets:
		s = gap if d == "oracle" else [r[d] for r in rows]
		order = sorted(range(len(rows)), key=lambda i: s[i])          # most trusted first
		cov = {c: sum(gap[i] for i in order[:max(1, int(c * len(rows)))]) / max(1, int(c * len(rows))) for c in (0.25, 0.5, 0.75)}
		aurc = sum(sum(gap[i] for i in order[:k]) / k for k in range(1, len(rows) + 1)) / len(rows)
		res[d] = dict(spearman_score_gap=spearman(s, gap) if d != "oracle" else 1.0, risk_at=cov, aurc=aurc)
	# policy evaluation from each policy's most-trusted half
	pols = sorted({r["policy"] for r in rows})
	real = {p: sum(r["real_return"] for r in rows if r["policy"] == p) / sum(r["policy"] == p for r in rows) for p in pols}
	def est(det):
		e = {}
		for p in pols:
			m = [r for r in rows if r["policy"] == p]
			if det is not None:
				key = (lambda r: abs(r["imagined_return"] - r["real_return"])) if det == "oracle" else (lambda r: r[det])
				m = sorted(m, key=key)[:max(1, len(m) // 2)]
			e[p] = sum(r["imagined_return"] for r in m) / len(m)
		return e
	res["policy_eval"] = {}
	for det in [None] + dets:
		e = est(det)
		res["policy_eval"]["all episodes" if det is None else det] = dict(
			spearman=spearman([e[p] for p in pols], [real[p] for p in pols]),
			mae=sum(abs(e[p] - real[p]) for p in pols) / len(pols))
	res["real_value"] = real
	return res


def report(data):
	L = ["# When to trust imagined policy evaluation: results", ""]
	for name, rows in data.items():
		a = analyse(rows)
		L += [f"## {name} ({len(rows)} imagined episodes; mean gap |imagined - real| = {a['mean_gap']:.3f})", "",
		      "| Detector | Spearman(score, gap) | gap @25% trusted | @50% | @75% | AURC (lower better; random = mean gap) |",
		      "| :-- | :-: | :-: | :-: | :-: | :-: |"]
		for d in [k for k in a if k in DET or k == "oracle"]:
			v = a[d]
			L.append(f"| {d} | {v['spearman_score_gap']:+.2f} | {v['risk_at'][0.25]:.3f} | {v['risk_at'][0.5]:.3f} | "
			         f"{v['risk_at'][0.75]:.3f} | {v['aurc']:.3f} |")
		L += ["", "| Policy evaluation from | Spearman over policies | Mean abs. error |", "| :-- | :-: | :-: |"]
		for k, v in a["policy_eval"].items():
			L.append(f"| {k} | {v['spearman']:+.2f} | {v['mae']:.3f} |")
		L.append("")
	return "\n".join(L)


if __name__ == "__main__":
	print(report(load(sys.argv[1:])))
